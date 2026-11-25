# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Dense-only native-order softmax inside the existing eight method nodes."""
import torch
import triton
from .method_pipeline_graph_hold_gate import Recorder, record_eight
from .parallel_conditional_hold_gate import ParallelConditionalMixin
from .method_pipeline_graph_dispatch import MethodBatch
from . import native_dense_probability_kernels as fixed_kernels
from . import native_dense_probability_legacy_kernels as legacy_kernels
from .native_probability_AV_kernels import native_probability_attention_values
VALIDATE = False
ALIGNED_FIX = True
VALIDATION_ROWS = []
_CLASSES = {}

def configure(validate=False, aligned_fix=True):
    global VALIDATE, ALIGNED_FIX
    VALIDATE = bool(validate)
    ALIGNED_FIX = bool(aligned_fix)

def bucket(n):
    if n <= 2048:
        return (0, max(1, triton.next_power_of_2(n) // 32))
    if n <= 9216:
        return (1, triton.cdiv(n, 1024))
    return (2, 0)

class NativeSoftmaxMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._softmax_alignment_fixed = ALIGNED_FIX
        self._native_probabilities = torch.empty_like(self.scratch_logits)
        self._native_probability_work = torch.zeros((2, len(self.layers), self.num_heads), device=query.device, dtype=torch.int64)

    def _batch_attention(self, layer, query, keys, values, mask=None):
        if self.work_counts is None:
            self._prepare_buffers(query)
        token = self.decode_tokens + 1
        n = keys.shape[-2]
        off = self.layer_offset[layer]
        if token > self.max_new_tokens or n > self.scratch_logits.shape[-1]:
            raise RuntimeError('Frozen allocation exceeded')
        state = self._state(layer, self.head_dim, query.device)
        decomp = self.decompositions.get(layer)
        if decomp is None:
            decomp = self._prepare_decomposition(layer, keys)
        if not hasattr(state, 'kernel_pointer_args'):
            state.kernel_pointer_args = (state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base)
        sp = state.kernel_pointer_args
        actual_mask = query if mask is None else mask
        kind, parameter = bucket(n)
        signature = (mask is not None, actual_mask.dtype, decomp.basis.shape[-1], decomp.low_keys.stride(0), float(self.model.model.language_model.layers[layer].self_attn.scaling), kind, parameter, self._softmax_alignment_fixed)
        key = (layer, signature)
        batch = self._pipeline_batches.get(key)
        if batch is None:
            proto = self._pipeline_prototypes.get(signature)
            if proto is None:
                cpu = tuple((getattr(self, k) for k in ['decode_tokens', 'last_nonvisual_tokens', 'text_key_rows', 'total_key_rows']))
                entries = []
                with record_eight(entries):
                    ParallelConditionalMixin.attention(self, layer, query, keys, values, mask)
                for k, v in zip(['decode_tokens', 'last_nonvisual_tokens', 'text_key_rows', 'total_key_rows'], cpu):
                    setattr(self, k, v)
                assert len(entries) == 8
                replacement = []
                H, D = (self.num_heads, self.head_dim)
                family = fixed_kernels if self._softmax_alignment_fixed else legacy_kernels
                function = [family.persistent, family.register_block, family.ilp4_block][kind]
                special = {'ITER': parameter} if kind == 0 else {'REG': parameter} if kind == 1 else {}
                Recorder('native_softmax_Dense_only', function, replacement)[H,](self.scratch_logits, self.route_flags, self._native_probabilities, self._native_probability_work[0], self._native_probability_work[1], n, self.scratch_logits.stride(0), off, H=H, num_warps=1 if kind == 0 else 32, enable_fp_fusion=False, **special)
                Recorder('attention_values_native_P', native_probability_attention_values, replacement)[H, triton.cdiv(n, 32)](values, self.visual_flags, self.scratch_logits, self.route_flags, self.route_float, self.partial_readout, self.partial_visual_readout, self.partial_mass, self.partial_visual_rows, n, values.stride(1), self.scratch_logits.stride(0), self.value_partition_stride, self._native_probabilities, D=D, H=H, GROUPS=self.groups, BLOCK=32, num_warps=4, enable_fp_fusion=False)
                entries = entries[:5] + replacement + entries[7:]
                assert len(entries) == 8
                batch = MethodBatch([v[1] for v in entries], [v[2] for v in entries])
                self._pipeline_prototypes[signature] = batch
            else:
                batch = proto.clone(sp, off)
            self._pipeline_batches[key] = batch
        self.last_nonvisual_tokens = n - self.visual_positions.numel()
        self.text_key_rows += self.num_heads * self.last_nonvisual_tokens
        self.total_key_rows += self.num_heads * n
        batch.launch_values((query.data_ptr(), keys.data_ptr(), values.data_ptr(), actual_mask.data_ptr(), decomp.basis.data_ptr(), decomp.low_keys.data_ptr(), n, keys.stride(1), values.stride(1), token))
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return self.output_buffer

    def attention(self, layer, query, keys, values, mask=None):
        token = self.decode_tokens + 1
        actual = super().attention(layer, query, keys, values, mask)
        if VALIDATE and token > 2:
            selected = torch.nonzero(self.route_flags[0], as_tuple=False).flatten()
            n = keys.shape[-2]
            if selected.numel():
                x = self.scratch_logits.index_select(0, selected)[:, :n]
                expected = torch.softmax(x, dim=-1, dtype=torch.float32).to(query.dtype)
                got = self._native_probabilities.index_select(0, selected)[:, :n]
                assert torch.equal(got, expected), 'Integrated native softmax differs on identical real logits'
            VALIDATION_ROWS.append(dict(layer=layer, token=token, n_keys=n, selected_Dense_heads=int(selected.numel()), native_BF16_probability_bit_exact=True, method_graph_kernel_count=8, no_extra_hotloop_Python_launch=True))
        return actual

    def statistics(self):
        r = super().statistics()
        r.update(native_softmax_aligned_fix=self._softmax_alignment_fixed, method_native_softmax=True, native_softmax_dense_head_rows=int(self._native_probability_work[0].sum()), native_softmax_key_rows=int(self._native_probability_work[1].sum()), native_softmax_graph_kernel_count=8, native_softmax_extra_hotloop_Python_launch=False, ordinary_HOLD_native_softmax_visual_work=False, numerical_implementation='Native-order FP32 softmax and BF16 probabilities for selected Dense heads; original fast HOLD text AV/proxy/QK/anchor reduction and common modules')
        return r

def gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('NativeSoftmax' + base.__name__, (NativeSoftmaxMixin, base), {})
    return _CLASSES[base]
