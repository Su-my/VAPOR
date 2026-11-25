# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Keep actual Dense heads on native eager; fuse only routing and HOLD work."""
import torch
from .merged_text_kernel import merged_text_logits
from .native_dense_hold_kernel import route_decode, attention_values, finalize_decode, parallel_visual_proxy, parallel_low_query
from .cached_triton_dispatch import CachedDispatch
merged_text_logits = CachedDispatch(merged_text_logits)
_CLASSES = {}

class ParallelConditionalMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self.route_float = torch.empty((6, self.num_heads), device=query.device, dtype=torch.float32)
        self._native_map = torch.arange(self.num_heads, device=query.device, dtype=torch.int32)
        import triton
        self._proxy_partition_stride = triton.cdiv(self.visual_positions.numel(), 64)
        self._proxy_part_block = triton.next_power_of_2(self._proxy_partition_stride)
        self._proxy_max = torch.empty((self.num_heads, self._proxy_partition_stride), device=query.device, dtype=torch.float32)
        self._proxy_sum = torch.empty_like(self._proxy_max)
        self._proxy_low_query = torch.empty((self.num_heads, self.rank), device=query.device, dtype=torch.float32)
        from .native_hybrid_dense_cpp import load
        self._hybrid_dense_cpp = load()
        self._dense_visual_max = torch.empty_like(self._proxy_max)
        self._dense_visual_sum = torch.empty_like(self._proxy_sum)
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        self._native_eager = eager_attention_forward
        self.backend = 'method_fused_selected_Dense_parallel_visual_QK_proxy_HOLD_v1'

    def attention(self, layer, query, kv_keys, kv_values, attention_mask=None):
        if query.shape[:3] != (1, self.num_heads, 1) or query.dtype != torch.bfloat16:
            raise ValueError('Native Dense/HOLD requires one BF16 query on one GPU')
        if self.work_counts is None:
            self._prepare_buffers(query)
        token = self.decode_tokens + 1
        n = kv_keys.shape[-2]
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
            decomp = self._prepare_decomposition(layer, kv_keys)
        if not hasattr(state, 'kernel_pointer_args'):
            state.kernel_pointer_args = (state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base)
        sp = state.kernel_pointer_args
        mask = query if attention_mask is None else attention_mask
        owner = self.model.model.language_model.layers[layer].self_attn
        scale = float(owner.scaling)
        common = dict(D=self.head_dim, H=self.num_heads, GROUPS=self.groups, enable_fp_fusion=False)
        merged_text_logits[self.num_heads // self.groups, (n + 63) // 64](query, kv_keys, mask, self.visual_flags, self.scratch_logits, self.rounding_flags, self.text_partition_max, self.text_partition_sum, self.text_partition_rows, n, kv_keys.stride(1), self.scratch_logits.stride(0), self.text_partition_max.stride(0), scale, HAS_MASK=attention_mask is not None, BLOCK=64, num_warps=4, **common)
        parallel_low_query[self.num_heads,](query, decomp.basis, sp, self._proxy_low_query, token, self.max_anchor_distance, D=self.head_dim, GROUPS=self.groups, RANK=decomp.basis.shape[-1], num_warps=8, enable_fp_fusion=False)
        parallel_visual_proxy[self.num_heads, self._proxy_partition_stride](query, mask, self.visual_positions, decomp.basis, decomp.low_keys, sp, self._proxy_max, self._proxy_sum, self._proxy_low_query, nv, decomp.low_keys.stride(0), self._proxy_partition_stride, token, self.max_anchor_distance, scale, D=self.head_dim, H=self.num_heads, GROUPS=self.groups, RANK=decomp.basis.shape[-1], HAS_MASK=attention_mask is not None, BLOCK=64, num_warps=8, enable_fp_fusion=False)
        route_decode[self.num_heads,](query, kv_keys, kv_values, mask, self.visual_flags, self.visual_positions, decomp.basis, decomp.low_keys, self.scratch_logits, self.output_buffer, self.text_partition_max, self.text_partition_sum, self.text_partition_rows, self._proxy_max, self._proxy_sum, self.route_flags, self.route_float, self.route_work, sp, self.kernel_counters, self.kernel_floats, self.kernel_work, n, nv, kv_keys.stride(1), kv_values.stride(1), decomp.low_keys.stride(0), self.scratch_logits.stride(0), self.text_partition_max.stride(0), self._proxy_partition_stride, token, off, scale, self.m1, self.confirmation_m1, self.m3_base, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max, self.max_anchor_distance, RANK=decomp.basis.shape[-1], PROXY_PART_BLOCK=self._proxy_part_block, HAS_MASK=attention_mask is not None, ENTRY_M3=self.entry_mass_m3_gate, EXTERNAL_DENSE=True, AGE_DECAY=getattr(self, 'fixed_age_decay', True), USE_HORIZON=getattr(self, 'fixed_use_horizon', True), SHARED_AGE_GATES=False, DECAY_KIND=getattr(self, 'fixed_decay_kind', 0), BLOCK=64, num_warps=8, **common)
        from .parallel_conditional_kernels import dense_visual_logits, dense_normalizer, attention_values_all, finalize_all
        dense_visual_logits[self.num_heads, self._proxy_partition_stride](query, kv_keys, mask, self.visual_positions, self.route_flags, self.scratch_logits, self._dense_visual_max, self._dense_visual_sum, nv, kv_keys.stride(1), self.scratch_logits.stride(0), self._proxy_partition_stride, scale, D=self.head_dim, H=self.num_heads, GROUPS=self.groups, HAS_MASK=attention_mask is not None, BLOCK=64, num_warps=8, enable_fp_fusion=False)
        dense_normalizer[self.num_heads,](self.route_flags, self.route_float, self._dense_visual_max, self._dense_visual_sum, nv, self._proxy_partition_stride, H=self.num_heads, PART_BLOCK=self._proxy_part_block, BLOCK=64, num_warps=4, enable_fp_fusion=False)
        attention_values_all[self.num_heads, (n + 31) // 32](kv_values, self.visual_flags, self.scratch_logits, self.route_flags, self.route_float, self.partial_readout, self.partial_visual_readout, self.partial_mass, self.partial_visual_rows, n, kv_values.stride(1), self.scratch_logits.stride(0), self.value_partition_stride, BLOCK=32, num_warps=4, **common)
        finalize_all[self.num_heads,](self.output_buffer, sp, self.kernel_counters, self.kernel_floats, self.kernel_work, self.route_flags, self.route_float, self.route_work, self.partial_readout, self.partial_visual_readout, self.partial_mass, self.partial_visual_rows, n, nv, self.value_partition_stride, token, off, self.m1, self.confirmation_m1, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max, D=self.head_dim, H=self.num_heads, ENTRY_M3=self.entry_mass_m3_gate, SHARED_AGE_GATES=False, SHARED_HORIZON=self.max_anchor_distance, BLOCK=32, PART_BLOCK=16, num_warps=4, enable_fp_fusion=False)
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return self.output_buffer

def native_dense_gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('ParallelConditional' + base.__name__, (ParallelConditionalMixin, base), {})
    return _CLASSES[base]
