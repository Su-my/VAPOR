# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Host wrapper for the fused on-policy GPU attention/state kernel."""
from types import SimpleNamespace
import torch
from .real_hold_gate import RealHoldGate
from .lean_stats_gate import VisualDecomposition
from .triton_hold_kernel import route_decode, COUNTER_NAMES, text_logits, correct_text_logits, attention_values, finalize_decode
from .prepared_triton_dispatch import PreparedDispatch as CachedDispatch
text_logits = CachedDispatch(text_logits)
correct_text_logits = CachedDispatch(correct_text_logits)
route_decode = CachedDispatch(route_decode)
attention_values = CachedDispatch(attention_values)
finalize_decode = CachedDispatch(finalize_decode)

class PreparedLaunchHoldGate(RealHoldGate):

    def reset(self):
        super().reset()
        self.backend = 'triton_cached_values32_v13'
        self.work_counts = None
        self.kernel_counters = None
        self.kernel_floats = None
        self.kernel_work = None
        self.energy_recorded = None
        self.raw_energy = None
        self.visual_flags = None
        self.scratch_logits = None
        self.rounding_flags = None
        self.output_buffer = None
        self.text_partition_max = None
        self.text_partition_sum = None
        self.text_partition_rows = None
        self.route_flags = None
        self.route_float = None
        self.route_work = None
        self.partial_readout = None
        self.value_partition_stride = 0
        self.partial_visual_readout = None
        self.partial_mass = None
        self.partial_visual_rows = None

    def original(self, hidden_states, cache_position):
        if self.prompt_length <= 1:
            return super().original(hidden_states, cache_position)
        return not self.active or hidden_states.shape[-2] != 1 or cache_position is None or (cache_position.numel() != 1)

    def _state(self, layer, dim, device):
        state = self.states.get(layer)
        if state is None:
            state = SimpleNamespace(holding=torch.zeros(self.num_heads, dtype=torch.bool, device=device), anchor_ready=torch.zeros(self.num_heads, dtype=torch.bool, device=device), anchor_token=torch.full((self.num_heads,), -1, dtype=torch.int16, device=device), anchor_readout=torch.zeros((self.num_heads, dim), dtype=torch.float32, device=device), anchor_nonvisual_tokens=torch.zeros(self.num_heads, dtype=torch.int32, device=device), episode_m3_base=torch.full((self.num_heads,), self.m3_base, dtype=torch.float32, device=device), episode_confirmation_threshold=torch.full((self.num_heads,), self.confirmation_m1, dtype=torch.float32, device=device))
            self.states[layer] = state
        return state

    def _prepare_buffers(self, query):
        device = query.device
        self.work_counts = torch.zeros((3, len(self.layers), self.num_heads), dtype=torch.int64, device=device)
        self.energy_recorded = torch.zeros(len(self.layers), dtype=torch.int32, device=device)
        self.raw_energy = torch.zeros(len(self.layers), dtype=torch.float32, device=device)
        capacity = self.prompt_length + self.max_new_tokens + 1
        self.visual_flags = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.visual_flags[self.visual_positions] = True
        self.scratch_logits = torch.empty((self.num_heads, capacity), dtype=query.dtype, device=device)
        self.rounding_flags = torch.empty_like(self.scratch_logits, dtype=torch.bool)
        self.output_buffer = torch.empty_like(query, memory_format=torch.contiguous_format)
        partitions = (capacity + 63) // 64
        self.text_partition_max = torch.empty((self.num_heads, partitions), dtype=torch.float32, device=device)
        self.text_partition_sum = torch.empty_like(self.text_partition_max)
        self.text_partition_rows = torch.empty((self.num_heads, partitions), dtype=torch.int32, device=device)
        self.route_flags = torch.empty((3, self.num_heads), dtype=torch.int32, device=device)
        self.route_float = torch.empty((4, self.num_heads), dtype=torch.float32, device=device)
        self.route_work = torch.empty((2, self.num_heads), dtype=torch.int64, device=device)
        value_partitions = (capacity + 31) // 32
        self.value_partition_stride = value_partitions
        self.partial_readout = torch.empty((self.num_heads, value_partitions, self.head_dim), dtype=torch.float32, device=device)
        self.partial_visual_readout = torch.empty_like(self.partial_readout)
        self.partial_mass = torch.empty((self.num_heads, value_partitions), dtype=torch.float32, device=device)
        self.partial_visual_rows = torch.empty((self.num_heads, value_partitions), dtype=torch.int32, device=device)
        self.kernel_counters = tuple((getattr(self, name) for name in COUNTER_NAMES))
        self.kernel_floats = (self.proxy_abs_error_sum, self.entry_mass_m3_value_sum, self.first_token_exact_mass, self.energy_sum, self.energy_count, self.energy_recorded, self.raw_energy)
        self.kernel_work = tuple(self.work_counts.unbind(0))

    def _prepare_decomposition(self, layer, kv_keys):
        visual = kv_keys[0, :, self.visual_positions, :].float()
        eigenvalues, basis = torch.linalg.eigh(visual.transpose(-1, -2) @ visual)
        eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
        rank = min(self.rank, visual.shape[-2], visual.shape[-1])
        basis = basis.flip(-1)[..., :rank].contiguous()
        low_keys = (visual @ basis).contiguous()
        padded_rank = 1 << (rank - 1).bit_length()
        if padded_rank != rank:
            basis = torch.nn.functional.pad(basis, (0, padded_rank - rank)).contiguous()
            low_keys = torch.nn.functional.pad(low_keys, (0, padded_rank - rank)).contiguous()
        energy = eigenvalues[..., :rank].sum(-1) / eigenvalues.sum(-1).clamp_min(1e-20)
        self.raw_energy[self.layer_offset[layer]] = energy.mean()
        result = VisualDecomposition(basis=basis, low_keys=low_keys, explained_energy=energy, effective_rank=rank)
        self.decompositions[layer] = result
        return result

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
        launches = self._prepared_launches.get(plan_key)
        updates = dict(Q=query.data_ptr(), K=kv_keys.data_ptr(), V=kv_values.data_ptr(), Mask=mask.data_ptr(), Basis=decomposition.basis.data_ptr(), LowKeys=decomposition.low_keys.data_ptr(), n_keys=n_keys, k_stride=kv_keys.stride(1), v_stride=kv_values.stride(1), token=token)
        if launches is None:
            launches = (text_logits.prepare((query, kv_keys, mask, self.visual_flags, self.scratch_logits, self.rounding_flags, self.text_partition_max, self.text_partition_sum, self.text_partition_rows, n_keys, kv_keys.stride(1), self.scratch_logits.stride(0), self.text_partition_max.stride(0), float(self.model.model.language_model.layers[layer].self_attn.scaling)), (self.num_heads // self.groups, (n_keys + 63) // 64), {'D': self.head_dim, 'H': self.num_heads, 'GROUPS': self.groups, 'HAS_MASK': attention_mask is not None, 'BLOCK': 64, 'num_warps': 4, 'enable_fp_fusion': False}), correct_text_logits.prepare((query, kv_keys, mask, self.visual_flags, self.scratch_logits, self.rounding_flags, self.text_partition_max, self.text_partition_sum, self.text_partition_rows, n_keys, kv_keys.stride(1), self.scratch_logits.stride(0), self.text_partition_max.stride(0), float(self.model.model.language_model.layers[layer].self_attn.scaling)), (self.num_heads, (n_keys + 63) // 64), {'D': self.head_dim, 'H': self.num_heads, 'GROUPS': self.groups, 'HAS_MASK': attention_mask is not None, 'BLOCK': 64, 'num_warps': 4, 'enable_fp_fusion': False}), route_decode.prepare((query, kv_keys, kv_values, mask, self.visual_flags, self.visual_positions, decomposition.basis, decomposition.low_keys, self.scratch_logits, self.output_buffer, self.text_partition_max, self.text_partition_sum, self.text_partition_rows, self.route_flags, self.route_float, self.route_work, state_ptrs, counts, floats, self.kernel_work, n_keys, n_visual, kv_keys.stride(1), kv_values.stride(1), decomposition.low_keys.stride(0), self.scratch_logits.stride(0), self.text_partition_max.stride(0), token, offset, float(self.model.model.language_model.layers[layer].self_attn.scaling), self.m1, self.confirmation_m1, self.m3_base, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max, self.max_anchor_distance), (self.num_heads,), {'D': self.head_dim, 'H': self.num_heads, 'GROUPS': self.groups, 'RANK': kernel_rank, 'HAS_MASK': attention_mask is not None, 'ENTRY_M3': self.entry_mass_m3_gate, 'AGE_DECAY': getattr(self, 'fixed_age_decay', True), 'USE_HORIZON': getattr(self, 'fixed_use_horizon', True), 'SHARED_AGE_GATES': getattr(self, 'fixed_shared_age_gates', False), 'BLOCK': 64, 'num_warps': 8, 'enable_fp_fusion': False}), attention_values.prepare((kv_values, self.visual_flags, self.scratch_logits, self.route_flags, self.route_float, self.partial_readout, self.partial_visual_readout, self.partial_mass, self.partial_visual_rows, n_keys, kv_values.stride(1), self.scratch_logits.stride(0), self.value_partition_stride), (self.num_heads, (n_keys + 31) // 32), {'D': self.head_dim, 'H': self.num_heads, 'GROUPS': self.groups, 'BLOCK': 32, 'num_warps': 4, 'enable_fp_fusion': False}), finalize_decode.prepare((self.output_buffer, state_ptrs, counts, floats, self.kernel_work, self.route_flags, self.route_float, self.route_work, self.partial_readout, self.partial_visual_readout, self.partial_mass, self.partial_visual_rows, n_keys, n_visual, self.value_partition_stride, token, offset, self.m1, self.confirmation_m1, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max), (self.num_heads,), {'D': self.head_dim, 'H': self.num_heads, 'ENTRY_M3': self.entry_mass_m3_gate, 'SHARED_AGE_GATES': getattr(self, 'fixed_shared_age_gates', False), 'SHARED_HORIZON': self.max_anchor_distance, 'BLOCK': 32, 'PART_BLOCK': 16, 'num_warps': 4, 'enable_fp_fusion': False}))
            self._prepared_launches[plan_key] = launches
        launches[0].launch((self.num_heads // self.groups, (n_keys + 63) // 64), updates)
        launches[1].launch((self.num_heads, (n_keys + 63) // 64), updates)
        launches[2].launch((self.num_heads,), updates)
        launches[3].launch((self.num_heads, (n_keys + 31) // 32), updates)
        launches[4].launch((self.num_heads,), updates)
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return self.output_buffer

    def statistics(self):
        self.execution_dense_visual_rows = int(self.dense_rows.sum().item())
        self.execution_skipped_visual_rows = int(self.hold_rows.sum().item())
        self.execution_text_proxy_rows = int(self.proxy_check_rows.sum().item())
        result = super().statistics()
        if self.work_counts is not None:
            qk, av, text = [int(v.sum().item()) for v in self.work_counts]
            result.update(actual_dense_visual_qk_key_rows=qk, actual_dense_visual_av_key_rows=av, actual_text_qk_key_rows=text, actual_attention_qk_macs=self.head_dim * (qk + text) + self.execution_text_proxy_rows * (1 << (min(self.rank, self.visual_positions.numel(), self.head_dim) - 1).bit_length()) * (self.head_dim + self.visual_positions.numel()), hot_loop_host_head_selection=False, proxy_kernel_rank=1 << (min(self.rank, self.visual_positions.numel(), self.head_dim) - 1).bit_length(), visual_decomposition_preparation='once per targeted layer, before first decode attention', numerical_implementation='BF16 rounded logits/probabilities, BF16 Tensor Core text-QK plus separate sparse single-key FP64 rounding correction, FP64 visual-QK, FP32 AV; no TF32', kernel_block=64, kernel_num_warps=8, text_qk_parallel_partitions=True, text_qk_tensorcore=True, text_qk_mma_query_rows=16, text_qk_valid_query_rows=self.groups, text_qk_partition_block=64, text_qk_partition_warps=4, attention_values_parallel=True, value_partition_block=32, value_finalize_part_block=16)
        return result
