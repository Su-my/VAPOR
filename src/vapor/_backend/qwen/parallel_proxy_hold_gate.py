# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Batch only the five existing HOLD kernel launches; unchanged CUDA math."""
import torch
from .pinned_stream_prepared_hold_gate import PinnedStreamPreparedHoldGate
from .prepared_launch_hold_gate import text_logits, correct_text_logits, route_decode, attention_values, finalize_decode
from .parallel_decomposition_hold_gate import install_parallel_hold_gate
from .parallel_proxy_batch_dispatch import HoldBatch
from .prepared_triton_dispatch import PreparedDispatch
from .merged_text_kernel import merged_text_logits
from .async_packed_batched_hold_gate import AsyncPackedBatchedHoldGate
text_logits = PreparedDispatch(merged_text_logits)
from .explicit_layout_route import route_decode as parallel_route
from .proxy_prepare import proxy_logits
route_decode = PreparedDispatch(parallel_route)
proxy_dispatch = PreparedDispatch(proxy_logits)

class ParallelProxyHoldGate(AsyncPackedBatchedHoldGate):

    def reset(self):
        super().reset()
        self._host_batches = {}
        self._host_prototypes = {}
        self.backend = 'triton_av32_merged_text_exact_correction_candidate_v1'

    def attention(self, layer, query, kv_keys, kv_values, attention_mask=None):
        if query.device.type != 'cuda':
            raise RuntimeError('Fused backend is GPU-only; no local model execution')
        if query.shape[:3] != (1, self.num_heads, 1) or query.dtype != torch.bfloat16:
            raise ValueError('Fused execution requires batch=1, one query, BF16')
        if self.head_dim != 128 or self.num_heads != 32 or self.rank not in (4, 8, 16, 32, 64, 128):
            raise ValueError('This frozen kernel targets the admitted Qwen3-VL-8B dimensions')
        if self.work_counts is None:
            self._prepare_buffers(query)
        token = self.decode_tokens + 1
        n_keys = kv_keys.shape[-2]
        if token > self.max_new_tokens or n_keys > self.scratch_logits.shape[-1]:
            raise RuntimeError('Generation escaped its frozen allocation or output cap')
        offset = self.layer_offset[layer]
        n_visual = self.visual_positions.numel()
        if not hasattr(self, 'proxy_workspace'):
            self.proxy_workspace = torch.empty((self.num_heads, n_visual), device=query.device, dtype=torch.float32)
        self.last_nonvisual_tokens = n_keys - n_visual
        self.text_key_rows += self.num_heads * (n_keys - n_visual)
        self.total_key_rows += self.num_heads * n_keys
        state = self._state(layer, self.head_dim, query.device)
        decomposition = self.decompositions.get(layer)
        if decomposition is None:
            decomposition = self._prepare_decomposition(layer, kv_keys)
        kernel_rank = decomposition.basis.shape[-1]
        if kv_keys.stride(-1) != 1 or kv_keys.stride(-2) != self.head_dim:
            raise ValueError('Unexpected key-cache stride')
        if kv_values.stride(-1) != 1 or kv_values.stride(-2) != self.head_dim:
            raise ValueError('Unexpected value-cache stride')
        if attention_mask is not None and (attention_mask.ndim != 4 or attention_mask.shape[1:3] != (1, 1)):
            raise ValueError('Expected one shared additive decode mask')
        mask = attention_mask if attention_mask is not None else query
        if not hasattr(state, 'kernel_pointer_args'):
            state.kernel_pointer_args = (state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base)
        state_ptrs = state.kernel_pointer_args
        counts = self.kernel_counters
        floats = self.kernel_floats
        plan_key = (layer, attention_mask is not None, mask.dtype, kernel_rank)
        if not hasattr(decomposition, 'kernel_pointer_args'):
            decomposition.kernel_pointer_args = (decomposition.basis.data_ptr(), decomposition.low_keys.data_ptr())
        updates = (query.data_ptr(), kv_keys.data_ptr(), kv_values.data_ptr(), mask.data_ptr(), *decomposition.kernel_pointer_args, n_keys, kv_keys.stride(1), kv_values.stride(1), token)
        batch = self._host_batches.get(plan_key)
        if batch is None:
            template_key = (attention_mask is not None, mask.dtype, kernel_rank, decomposition.low_keys.stride(0), float(self.model.model.language_model.layers[layer].self_attn.scaling))
            prototype = self._host_prototypes.get(template_key)
            if prototype is None:
                launches = (text_logits.prepare((query, kv_keys, mask, self.visual_flags, self.scratch_logits, self.rounding_flags, self.text_partition_max, self.text_partition_sum, self.text_partition_rows, n_keys, kv_keys.stride(1), self.scratch_logits.stride(0), self.text_partition_max.stride(0), float(self.model.model.language_model.layers[layer].self_attn.scaling)), (self.num_heads // self.groups, (n_keys + 63) // 64), {'D': self.head_dim, 'H': self.num_heads, 'GROUPS': self.groups, 'HAS_MASK': attention_mask is not None, 'BLOCK': 64, 'num_warps': 4, 'enable_fp_fusion': False}), route_decode.prepare((query, kv_keys, kv_values, mask, self.visual_flags, self.visual_positions, decomposition.basis, decomposition.low_keys, self.scratch_logits, self.proxy_workspace, self.text_partition_max, self.text_partition_sum, self.text_partition_rows, self.route_flags, self.route_float, self.route_work, state_ptrs, counts, floats, self.kernel_work, n_keys, n_visual, kv_keys.stride(1), kv_values.stride(1), decomposition.low_keys.stride(0), self.scratch_logits.stride(0), self.text_partition_max.stride(0), token, offset, float(self.model.model.language_model.layers[layer].self_attn.scaling), self.m1, self.confirmation_m1, self.m3_base, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max, self.max_anchor_distance), (self.num_heads,), {'D': self.head_dim, 'H': self.num_heads, 'GROUPS': self.groups, 'RANK': kernel_rank, 'HAS_MASK': attention_mask is not None, 'ENTRY_M3': self.entry_mass_m3_gate, 'AGE_DECAY': getattr(self, 'fixed_age_decay', True), 'USE_HORIZON': getattr(self, 'fixed_use_horizon', True), 'SHARED_AGE_GATES': getattr(self, 'fixed_shared_age_gates', False), 'BLOCK': 64, 'num_warps': 8, 'enable_fp_fusion': False}), attention_values.prepare((kv_values, self.visual_flags, self.scratch_logits, self.route_flags, self.route_float, self.partial_readout, self.partial_visual_readout, self.partial_mass, self.partial_visual_rows, n_keys, kv_values.stride(1), self.scratch_logits.stride(0), self.value_partition_stride), (self.num_heads, (n_keys + 31) // 32), {'D': self.head_dim, 'H': self.num_heads, 'GROUPS': self.groups, 'BLOCK': 32, 'num_warps': 4, 'enable_fp_fusion': False}), finalize_decode.prepare((self.output_buffer, state_ptrs, counts, floats, self.kernel_work, self.route_flags, self.route_float, self.route_work, self.partial_readout, self.partial_visual_readout, self.partial_mass, self.partial_visual_rows, n_keys, n_visual, self.value_partition_stride, token, offset, self.m1, self.confirmation_m1, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max), (self.num_heads,), {'D': self.head_dim, 'H': self.num_heads, 'ENTRY_M3': self.entry_mass_m3_gate, 'SHARED_AGE_GATES': getattr(self, 'fixed_shared_age_gates', False), 'SHARED_HORIZON': self.max_anchor_distance, 'BLOCK': 32, 'PART_BLOCK': 16, 'num_warps': 4, 'enable_fp_fusion': False}))
                proxy_launch = proxy_dispatch.prepare((query, mask, self.visual_positions, decomposition.basis, decomposition.low_keys, self.proxy_workspace, state_ptrs, self.route_float, n_visual, decomposition.low_keys.stride(0), token, float(self.model.model.language_model.layers[layer].self_attn.scaling), self.max_anchor_distance, self.m3_base), (self.num_heads * ((n_visual + 63) // 64),), dict(D=self.head_dim, H=self.num_heads, GROUPS=self.groups, RANK=kernel_rank, HAS_MASK=attention_mask is not None, USE_HORIZON=self.fixed_use_horizon, BLOCK=64, num_warps=8, enable_fp_fusion=False))
                launches = (launches[0], proxy_launch, *launches[1:])
                batch = HoldBatch(launches, self.groups, n_visual)
                self._host_prototypes[template_key] = batch
            else:
                batch = prototype.clone(state_ptrs, offset)
            self._host_batches[plan_key] = batch
        batch.launch_values(updates)
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return self.output_buffer
