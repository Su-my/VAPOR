# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Avoid unused SVD for natural short outputs; retain exact HOLD computation."""
import torch
from .batched_host_hold_gate import BatchedHostHoldGate
from .async_first_decode_hold_gate import AsyncFirstDecodeHoldGate
from .lean_stats_gate import VisualDecomposition
from .parallel_decomposition_hold_gate import install_parallel_hold_gate
from .prefill_schedule import PendingDecompositions
from .packed_hold_state import PackedHoldState

class LazyPackedBatchedHoldGate(BatchedHostHoldGate):

    def reset(self):
        pending = getattr(self, '_prefill_pending', None)
        if pending is not None:
            pending.clear(lambda e: e.synchronize())
        super().reset()
        self._prefill_pending = PendingDecompositions()
        self._dummy_decomposition = None
        self._packed_state = None
        self.backend = 'triton_av32_lazy3_packed_state_batched_host_candidate_v1'

    def _state(self, layer, dim, device):
        state = self.states.get(layer)
        if state is None:
            if self._packed_state is None:
                self._packed_state = PackedHoldState(len(self.layers), self.num_heads, dim, self.m3_base, self.confirmation_m1, device)
            state = self._packed_state.row(self.layer_offset[layer])
            self.states[layer] = state
        return state

    def _prepare_decomposition(self, layer, kv_keys):
        if self.decode_tokens + 1 < 3:
            if self._dummy_decomposition is None:
                rank = min(self.rank, self.visual_positions.numel(), self.head_dim)
                padded = 1 << (rank - 1).bit_length()
                heads = self.num_heads // self.groups
                self._dummy_decomposition = VisualDecomposition(basis=torch.empty((heads, self.head_dim, padded), dtype=torch.float32, device=kv_keys.device), low_keys=torch.empty((heads, self.visual_positions.numel(), padded), dtype=torch.float32, device=kv_keys.device), explained_energy=torch.zeros(heads, dtype=torch.float32, device=kv_keys.device), effective_rank=rank)
            return self._dummy_decomposition
        if layer in self.decompositions:
            return self.decompositions[layer]
        AsyncFirstDecodeHoldGate._start_decompositions(self, layer, kv_keys)
        if layer not in self._prefill_pending.futures:
            return super()._prepare_decomposition(layer, kv_keys)
        decomposition, event = self._prefill_pending.futures[layer].result()
        torch.cuda.current_stream().wait_event(event)
        self.raw_energy[self.layer_offset[layer]] = decomposition.explained_energy.mean()
        self.decompositions[layer] = decomposition
        return decomposition

    def finish_async_preparation(self):
        self._prefill_pending.join(torch.cuda.current_stream().wait_event)
