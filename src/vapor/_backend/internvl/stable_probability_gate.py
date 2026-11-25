# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Keep the fast pipeline; avoid exp(logit-logsumexp) reconstruction.

Diagnostics use canonical selected-head operators on identical Q/K/V and
pre-step anchor state. They never run during timed generation.
"""
import torch
import triton
import triton.language as tl
from .cached_triton_dispatch import CachedDispatch
from . import parallel_conditional_kernels as auxiliary
from .stable_probability_kernels import stable_attention_values, stable_dense_normalizer
ORIGINAL_AV = auxiliary.attention_values_all
ORIGINAL_NORMALIZER = auxiliary.dense_normalizer
ENABLED = False
VALIDATE = False
VALIDATION_ROWS = []
_CLASSES = {}

def configure(enabled):
    global ENABLED
    ENABLED = bool(enabled)
    auxiliary.attention_values_all = CachedDispatch(stable_attention_values) if enabled else ORIGINAL_AV
    auxiliary.dense_normalizer = CachedDispatch(stable_dense_normalizer) if enabled else ORIGINAL_NORMALIZER

def set_validation(enabled):
    global VALIDATE
    VALIDATE = bool(enabled)

def metric(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return dict(elements=a.numel(), different_elements=int((a != b).sum()), max_abs=float(d.max()) if d.numel() else 0.0, abs_error_sum=float(d.sum()))

@triton.jit(do_not_specialize=['n_keys', 'scratch_stride'])
def saved_probabilities(Scratch, Routes, RF, P, n_keys, scratch_stride, H: tl.constexpr=32, B: tl.constexpr=256, STABLE: tl.constexpr=False):
    h = tl.program_id(0)
    i = tl.program_id(1) * B + tl.arange(0, B)
    if tl.load(Routes + h) != 0:
        x = tl.load(Scratch + h * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
        if STABLE:
            maximum = tl.load(RF + 4 * H + h)
            total = tl.load(RF + 5 * H + h)
            p = tl.exp(x - maximum) / total
        else:
            z = tl.load(RF + 2 * H + h)
            p = tl.exp(x - z)
        tl.store(P + h * n_keys + i, p.to(tl.bfloat16), i < n_keys)

class StableProbabilityMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._stable_probability_enabled = ENABLED

    def statistics(self):
        r = super().statistics()
        r.update(stable_probability_reconstruction=getattr(self, '_stable_probability_enabled', ENABLED), numerical_implementation='Fast conditional kernels with max/sum probability reconstruction; identical QK/proxy and selected-head canonical diagnostics' if getattr(self, '_stable_probability_enabled', ENABLED) else 'Unchanged fast GPU-eigen conditional kernels', no_extra_hot_loop_host_selection=not getattr(self, '_native_bootstrap_enabled', False) or getattr(self, '_static_native_bootstrap_enabled', False), ordinary_HOLD_visual_Dense_computed=False)
        return r

    def attention(self, layer, query, keys, values, mask=None):
        if not VALIDATE or self.decode_tokens + 1 > 4:
            return super().attention(layer, query, keys, values, mask)
        if self.work_counts is None:
            self._prepare_buffers(query)
        state = self._state(layer, self.head_dim, query.device)
        decomp = self.decompositions.get(layer)
        if decomp is None:
            decomp = self._prepare_decomposition(layer, keys)
        before = [v.clone() for v in [state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base]]
        token = self.decode_tokens + 1
        actual = super().attention(layer, query, keys, values, mask).clone()
        n = keys.shape[-2]
        H = self.num_heads
        D = self.head_dim
        from .native_selective_hold import load, step, threshold_table
        textpos = torch.nonzero(~self.visual_flags, as_tuple=False).flatten()
        ref, _ = step(load(), query, keys, values, mask, self.visual_positions, textpos, decomp.basis, decomp.low_keys, before, token, self.max_anchor_distance, self.groups, float(self.model.model.language_model.layers[layer].self_attn.scaling), self.m3_base, lut=threshold_table(self.max_anchor_distance, self.m3_base, query.device))
        ref_output, ref_proxy, ref_mass, ref_anchor, ref_stay, ref_checked = ref
        if getattr(self, '_native_bootstrap_enabled', False) and token <= 2:
            assert not bool(ref_stay.any()) and (not bool(ref_checked.any()))
            anchor_metric = metric(state.anchor_readout, ref_anchor)
            output_metric = metric(actual, ref_output)
            assert torch.equal(actual, ref_output), 'Native bootstrap output changed relative to canonical operators on the same QKV'
            assert anchor_metric['different_elements'] == 0, 'Native bootstrap anchor differs on the same QKV'
            VALIDATION_ROWS.append(dict(layer=layer, token=token, n_keys=n, dense_heads=H, stable_probability=True, tensorcore_QK=True, native_dense_bootstrap=True, actual_native_bootstrap=True, proxy_checked_structure_differences=0, route_differences_vs_canonical=0, whole_output_vs_canonical=output_metric, same_Dense_output_vs_canonical=output_metric, same_HOLD_output_vs_canonical=metric(actual[:, :0], ref_output[:, :0]), proxy_vs_canonical=metric(ref_proxy[:0], ref_proxy[:0]), anchor_vs_canonical_selected=anchor_metric, native_bootstrap_output_and_anchor_same_QKV_exact=True, component_scope='Actual canonical native bootstrap; QK/softmax/AV substage comparisons not reconstructed from unused fast scratch buffers'))
            return actual
        routes = self.route_flags[0].bool()
        checked = self.route_flags[1].bool()
        dense = torch.nonzero(routes, as_tuple=False).flatten()
        same_dense = routes & ~ref_stay
        same_hold = ~routes & ref_stay
        entry = dict(layer=layer, token=token, n_keys=n, dense_heads=int(routes.sum()), stable_probability=self._stable_probability_enabled, tensorcore_QK=getattr(self, '_tensorcore_QK_enabled', False), native_dense_bootstrap=getattr(self, '_native_bootstrap_enabled', False), actual_native_bootstrap=False, proxy_checked_structure_differences=int((checked != ref_checked).sum()), route_differences_vs_canonical=int((routes != ~ref_stay).sum()), whole_output_vs_canonical=metric(actual, ref_output), same_Dense_output_vs_canonical=metric(actual[:, same_dense], ref_output[:, same_dense]), same_HOLD_output_vs_canonical=metric(actual[:, same_hold], ref_output[:, same_hold]), proxy_vs_canonical=metric(self.route_float[0, checked], ref_proxy[checked]))
        assert entry['proxy_checked_structure_differences'] == 0
        if dense.numel():
            ki = dense // self.groups
            q = query[0, :, 0].index_select(0, dense).unsqueeze(1)
            kk = keys[0].index_select(0, ki)
            vv = values[0].index_select(0, ki)
            logits = torch.matmul(q, kk.transpose(1, 2)) * float(self.model.model.language_model.layers[layer].self_attn.scaling)
            if mask is not None:
                logits += mask[0, 0, :1, :n]
            native_probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            fast_logits = self.scratch_logits.index_select(0, dense)[:, :n].unsqueeze(1).to(query.dtype)
            same_logits_native_probs = torch.softmax(fast_logits, dim=-1, dtype=torch.float32).to(query.dtype)
            all_fast_probs = torch.empty((H, n), device=query.device, dtype=query.dtype)
            saved_probabilities[H, triton.cdiv(n, 256)](self.scratch_logits, self.route_flags, self.route_float, all_fast_probs, n, self.scratch_logits.stride(0), STABLE=self._stable_probability_enabled, num_warps=4, enable_fp_fusion=False)
            fp = all_fast_probs.index_select(0, dense).unsqueeze(1)
            exact_AV_given_fast_P = torch.matmul(fp, vv)
            vp = fp[:, 0].index_select(1, self.visual_positions).float()
            vvisual = vv.index_select(1, self.visual_positions).float()
            mass = vp.sum(-1)
            vr = torch.einsum('hv,hvd->hd', vp, vvisual)
            anchor_given_fast_P = torch.where(mass[:, None] > 0, vr / mass.clamp_min(1e-20)[:, None], 0.0)
            native_mass = native_probs[:, 0].index_select(1, self.visual_positions).float().sum(-1)
            native_vp = native_probs[:, 0].index_select(1, self.visual_positions).float()
            native_vr = torch.einsum('hv,hvd->hd', native_vp, vvisual)
            native_anchor = torch.where(native_mass[:, None] > 0, native_vr / native_mass.clamp_min(1e-20)[:, None], 0.0)
            entry.update(QK_vs_canonical_selected=metric(fast_logits, logits), softmax_given_same_fast_logits=metric(fp, same_logits_native_probs), probabilities_vs_canonical_selected=metric(fp, native_probs), AV_given_same_fast_probabilities=metric(actual[0, :, 0].index_select(0, dense), exact_AV_given_fast_P[:, 0]), anchor_given_same_fast_probabilities=metric(state.anchor_readout.index_select(0, dense), anchor_given_fast_P), anchor_vs_canonical_selected=metric(state.anchor_readout.index_select(0, dense), native_anchor), BF16_probability_mass_vs_canonical=metric(mass, native_mass))
        VALIDATION_ROWS.append(entry)
        return actual

def diagnostic_gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('StableProbability' + base.__name__, (StableProbabilityMixin, base), {})
    return _CLASSES[base]
