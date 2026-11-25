# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Batch the native selective eager algorithm's ATen calls in one C++ call.

No visual Dense QK/AV runs on an ordinary HOLD head. All compaction,
normalization, AV and state arithmetic use the independent eager oracle's
native operators and shapes, including its measured synchronization costs.
"""
import hashlib
import os
from pathlib import Path
import torch
SOURCE = '\n#include <torch/extension.h>\n#include <optional>\n#include <tuple>\n#include <vector>\nstd::tuple<std::vector<at::Tensor>,int64_t> attention(\n    const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,\n    const std::optional<at::Tensor>& mask,const at::Tensor& positions,const at::Tensor& alltextpos,\n    const at::Tensor& basis,const at::Tensor& lowkeys,\n    std::vector<at::Tensor> state,\n    int64_t token,int64_t horizon,int64_t groups,\n    double scaling,double m3,const at::Tensor& threshold,const at::Tensor& checked) {\n    auto H=query.size(1),D=query.size(3),N=key.size(2),NV=positions.numel();\n    auto proxyheads=at::nonzero(checked).flatten();\n    auto proxy=at::full({H},NAN,query.options().dtype(at::kFloat));\n    auto textread=at::zeros({H,D},query.options().dtype(at::kFloat));\n    if(proxyheads.numel()) {\n        auto textpos=alltextpos.slice(0,0,N-NV);\n        auto ki=at::floor_divide(proxyheads,groups);\n        auto tk=key.select(0,0).index_select(1,textpos).index_select(0,ki);\n        auto q=query.select(0,0).select(1,0).index_select(0,proxyheads).unsqueeze(1);\n        auto logits=at::matmul(q,tk.transpose(1,2)).squeeze(1)*scaling;\n        if(mask.has_value()&&mask->defined())logits+=mask->select(0,0).select(0,0).select(0,0).index_select(0,textpos);\n        logits=logits.to(at::kFloat);\n        auto ztext=at::logsumexp(logits,{-1});\n        auto lowq=at::einsum("hd,hdr->hr",{q.squeeze(1).to(at::kFloat),basis.index_select(0,ki)});\n        auto approximate=at::einsum("hr,hvr->hv",{lowq,lowkeys.index_select(0,ki)})*scaling;\n        if(mask.has_value()&&mask->defined())approximate+=mask->select(0,0).select(0,0).select(0,0).index_select(0,positions).to(at::kFloat);\n        auto pm=at::sigmoid(at::logsumexp(approximate,{-1})-ztext);\n        proxy.index_copy_(0,proxyheads,pm);\n        auto localhold=pm<threshold.index_select(0,proxyheads);\n        auto holdindices=at::nonzero(localhold).flatten();\n        if(holdindices.numel()) {\n            auto heads=proxyheads.index_select(0,holdindices);\n            auto tv=value.select(0,0).index_select(1,textpos).index_select(0,at::floor_divide(heads,groups)).to(at::kFloat);\n            auto probs=at::softmax(logits.index_select(0,holdindices),-1,at::kFloat);\n            textread.index_copy_(0,heads,at::einsum("hs,hsd->hd",{probs,tv}));\n        }\n    }\n    auto stay=at::logical_and(checked,proxy<threshold);\n    auto dense=at::logical_not(stay),heads=at::nonzero(dense).flatten();\n    auto mass=at::full_like(proxy,NAN),visual=at::full_like(textread,NAN);\n    auto denseout=at::zeros_like(query);\n    if(heads.numel()) {\n        auto ki=at::floor_divide(heads,groups);\n        auto keys=key.select(0,0).index_select(0,ki),values=value.select(0,0).index_select(0,ki);\n        auto q=query.select(0,0).select(1,0).index_select(0,heads).unsqueeze(1);\n        auto logits=at::matmul(q,keys.transpose(1,2))*scaling;\n        if(mask.has_value()&&mask->defined())logits+=mask->select(0,0).select(0,0).slice(0,0,1).slice(1,0,N);\n        auto probs=at::softmax(logits,-1,at::kFloat).to(query.scalar_type());\n        denseout.select(0,0).select(1,0).index_copy_(0,heads,at::matmul(probs,values).squeeze(1));\n        auto vp=probs.select(1,0).index_select(1,positions).to(at::kFloat);\n        auto vv=values.index_select(1,positions).to(at::kFloat);\n        auto vm=vp.sum(-1),vr=at::einsum("hv,hvd->hd",{vp,vv});\n        mass.index_copy_(0,heads,vm);\n        visual.index_copy_(0,heads,at::where(vm.unsqueeze(1)>0,vr/at::clamp_min(vm,1.e-20).unsqueeze(1),at::zeros_like(vr)));\n    }\n    return {std::vector<at::Tensor>{denseout,proxy,mass,visual,stay,checked,textread},proxyheads.numel()};\n}\n'
_MODULE = None
_CLASSES = {}

def load():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline
        if not os.environ.get('CXX'):
            for p in ['/opt/rh/gcc-toolset-12/root/usr/bin/g++', '/opt/rh/gcc-toolset-11/root/usr/bin/g++']:
                if Path(p).exists():
                    os.environ['CXX'] = p
                    break
        os.environ['MAX_JOBS'] = '1'
        _MODULE = load_inline(name='phase_native_selective_hold_' + hashlib.sha256(SOURCE.encode()).hexdigest()[:16], cpp_sources=SOURCE, functions=['attention'], extra_cflags=['-O3'], with_cuda=False, verbose=False)
    return _MODULE

def threshold_table(horizon, m3, device):
    distance = torch.arange(horizon + 1, device=device, dtype=torch.int16)
    phase = torch.clamp(torch.clamp_min(distance - 2, 0).float() / float(horizon - 2), 0.0, 1.0)
    return m3 * torch.cos(0.5 * 3.141592653589793 * phase)

def step(module, query, keys, values, mask, positions, textpos, basis, lowkeys, state, token, horizon, groups, scaling, m3, lut=None):
    from .native_selective_gate_output_kernel import select_gate, combine_output
    H = query.shape[1]
    D = query.shape[-1]
    if lut is None:
        lut = threshold_table(horizon, m3, query.device)
    threshold = torch.empty((H,), device=query.device, dtype=torch.float32)
    checked = torch.empty((H,), device=query.device, dtype=torch.bool)
    select_gate[1,](state[0], state[1], state[2], lut, threshold, checked, token, horizon, H=H, B=32, num_warps=1, enable_fp_fusion=False)
    raw, np = module.attention(query, keys, values, mask, positions, textpos, basis, lowkeys, state, token, horizon, groups, scaling, m3, threshold, checked)
    denseout, proxy, mass, visual, stay, checked, textread = raw
    output = torch.empty_like(query)
    combine_output[H,](denseout, proxy, textread, state[3], stay, output, D=D, num_warps=4, enable_fp_fusion=False)
    return ([output, proxy, mass, visual, stay, checked], np)

class NativeSelectiveMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._selective_cpp = load()
        self._energy_used = set()
        self._all_text_positions = torch.nonzero(~self.visual_flags, as_tuple=False).flatten()
        self.backend = 'native_eager_ATen_selective_HOLD_cached_method_launch_v1'
        self._age_threshold_lut = threshold_table(self.max_anchor_distance, self.m3_base, query.device)

    def attention(self, layer, query, keys, values, mask=None):
        if self.work_counts is None:
            self._prepare_buffers(query)
        if self.entry_mass_m3_gate:
            raise ValueError('Direct-transfer fixed initial tau only')
        token = self.decode_tokens + 1
        n = keys.shape[-2]
        nv = self.visual_positions.numel()
        off = self.layer_offset[layer]
        if token > self.max_new_tokens or n > self.scratch_logits.shape[-1]:
            raise RuntimeError('Frozen allocation exceeded')
        self.last_nonvisual_tokens = n - nv
        self.text_key_rows += self.num_heads * (n - nv)
        self.total_key_rows += self.num_heads * n
        state = self._state(layer, self.head_dim, query.device)
        decomp = self.decompositions.get(layer)
        if decomp is None:
            decomp = self._prepare_decomposition(layer, keys)
        state_args = (state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base)
        parts, np = step(self._selective_cpp, query, keys, values, mask, self.visual_positions, self._all_text_positions, decomp.basis, decomp.low_keys, list(state_args), token, self.max_anchor_distance, self.groups, float(self.model.model.language_model.layers[layer].self_attn.scaling), self.m3_base, lut=self._age_threshold_lut)
        output, proxy, mass, visual, stay, checked = parts
        from .native_selective_state_kernel import finish_state
        finish_state[self.num_heads,](state_args, self.kernel_counters, self.kernel_floats, self.kernel_work, proxy, mass, visual, stay, checked, token, off, n, nv, self.m1, self.max_anchor_distance, H=self.num_heads, D=self.head_dim, num_warps=4, enable_fp_fusion=False)
        if np and layer not in self._energy_used:
            self._energy_used.add(layer)
            self.energy_sum[off] += decomp.explained_energy.mean().double()
            self.energy_count[off] += 1
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return output

    def statistics(self):
        result = super().statistics()
        result.update(execution_backend=self.backend, numerical_implementation='Native eager reductions unchanged; exact pointwise age gate, HOLD output selection, and state/counters in scoped kernels', hot_loop_host_head_selection=True, all_compaction_cost_in_timing=True, ordinary_HOLD_visual_Dense_computed=False)
        return result

def native_selective_gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('NativeSelective' + base.__name__, (NativeSelectiveMixin, base), {})
    return _CLASSES[base]
