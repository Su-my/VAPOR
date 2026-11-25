# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Reuse the canonical Dense branch with known all-head bootstrap routing."""
from pathlib import Path
import hashlib
import os
import torch
from .native_selective_hold import SOURCE as ORIGINAL_SOURCE
_start = ORIGINAL_SOURCE.index('    if(heads.numel()) {\n') + len('    if(heads.numel()) {\n')
_end = ORIGINAL_SOURCE.index('    }\n    return ', _start)
ORIGINAL_DENSE_BODY = ORIGINAL_SOURCE[_start:_end]
assert ORIGINAL_DENSE_BODY.count('auto ki=at::floor_divide(heads,groups);') == 1
DENSE_BODY = ORIGINAL_DENSE_BODY.replace('auto ki=at::floor_divide(heads,groups);', 'auto ki=cached_kv_heads;')
SOURCE = '\n#include <torch/extension.h>\n#include <optional>\n#include <vector>\nstd::vector<at::Tensor> attention(\n    const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,\n    const std::optional<at::Tensor>& mask,const at::Tensor& positions,\n    const at::Tensor& heads,const at::Tensor& cached_kv_heads,\n    const at::Tensor& proxy,int64_t groups,double scaling) {\n    auto H=query.size(1),D=query.size(3),N=key.size(2);\n    TORCH_CHECK(heads.numel()==H && cached_kv_heads.numel()==H,"Expected all bootstrap heads");\n    auto textread=at::zeros({H,D},query.options().dtype(at::kFloat));\n    auto mass=at::full_like(proxy,NAN),visual=at::full_like(textread,NAN);\n    auto denseout=at::zeros_like(query);\n' + DENSE_BODY + '\n    return {denseout,mass,visual};\n}\n'
ENABLED = False
_MODULE = None
_CLASSES = {}

def configure(enabled):
    global ENABLED
    ENABLED = bool(enabled)

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
        _MODULE = load_inline(name='phase_static_native_bootstrap_' + hashlib.sha256(SOURCE.encode()).hexdigest()[:16], cpp_sources=SOURCE, functions=['attention'], extra_cflags=['-O3'], with_cuda=False, verbose=False)
    return _MODULE

class StaticNativeBootstrapMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._static_native_bootstrap_enabled = ENABLED
        if ENABLED:
            self._static_bootstrap_cpp = load()
            self._bootstrap_all_heads = torch.arange(self.num_heads, device=query.device, dtype=torch.int64)
            self._bootstrap_kv_heads = self._bootstrap_all_heads // self.groups
            self._bootstrap_proxy = torch.full((self.num_heads,), float('nan'), device=query.device, dtype=torch.float32)
            self._bootstrap_false = torch.zeros((self.num_heads,), device=query.device, dtype=torch.bool)

    def attention(self, layer, query, keys, values, mask=None):
        if not (getattr(self, '_static_native_bootstrap_enabled', ENABLED) and self.decode_tokens + 1 <= 2):
            return super().attention(layer, query, keys, values, mask)
        if self.work_counts is None:
            self._prepare_buffers(query)
        assert self._native_bootstrap_enabled
        token, n, nv = (self.decode_tokens + 1, keys.shape[-2], self.visual_positions.numel())
        if token > self.max_new_tokens or n > self.scratch_logits.shape[-1]:
            raise RuntimeError('Frozen allocation exceeded')
        self.last_nonvisual_tokens = n - nv
        self.text_key_rows += self.num_heads * (n - nv)
        self.total_key_rows += self.num_heads * n
        state = self._state(layer, self.head_dim, query.device)
        if self.decompositions.get(layer) is None:
            self._prepare_decomposition(layer, keys)
        args = (state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base)
        scaling = float(self.model.model.language_model.layers[layer].self_attn.scaling)
        dense, mass, visual = self._static_bootstrap_cpp.attention(query, keys, values, mask, self.visual_positions, self._bootstrap_all_heads, self._bootstrap_kv_heads, self._bootstrap_proxy, self.groups, scaling)
        from .native_selective_state_kernel import finish_state
        finish_state[self.num_heads,](args, self.kernel_counters, self.kernel_floats, self.kernel_work, self._bootstrap_proxy, mass, visual, self._bootstrap_false, self._bootstrap_false, token, self.layer_offset[layer], n, nv, self.m1, self.max_anchor_distance, H=self.num_heads, D=self.head_dim, num_warps=4, enable_fp_fusion=False)
        self._native_bootstrap_calls += 1
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return dense

    def statistics(self):
        r = super().statistics()
        enabled = getattr(self, '_static_native_bootstrap_enabled', ENABLED)
        r.update(native_bootstrap_static=enabled, hot_loop_host_head_selection=False if enabled else r.get('hot_loop_host_head_selection', True), native_bootstrap_static_float_ops_unchanged=enabled, host_selection_scope='Statically all Dense for the first2 already-mandatory cached queries; no data-dependent compaction' if enabled else r.get('host_selection_scope'))
        return r

def gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('StaticNativeBootstrap' + base.__name__, (StaticNativeBootstrapMixin, base), {})
    return _CLASSES[base]
