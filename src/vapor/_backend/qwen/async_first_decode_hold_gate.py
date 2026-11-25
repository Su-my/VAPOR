# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Run original SVD during the two mandatory Dense anchor queries."""
import torch
from .lean_stats_gate import VisualDecomposition
from .pinned_stream_prepared_hold_gate import PinnedStreamPreparedHoldGate, resources
from .parallel_decomposition_hold_gate import install_parallel_hold_gate
from .prefill_schedule import PendingDecompositions

class AsyncFirstDecodeHoldGate(PinnedStreamPreparedHoldGate):

    def reset(self):
        pending = getattr(self, '_prefill_pending', None)
        if pending is not None:
            pending.clear(lambda event: event.synchronize())
        super().reset()
        self._prefill_pending = PendingDecompositions()
        self._dummy_decomposition = None
        self.backend = 'triton_av32_async_first_decode_svd_candidate_v1'

    def _start_decompositions(self, layer, keys):
        if self._prefill_pending.futures:
            return
        snapshot = {i: self.decode_cache.layers[i].keys for i in self.layers} if self.decode_cache is not None else {layer: keys}
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        bank = resources(keys.device)
        for i, immutable_keys in snapshot.items():
            executor, stream = bank[self.layer_offset[i] % len(bank)]

            def submit(value, _executor=executor, _stream=stream):
                return _executor.submit(self._prepare_one, value, _stream, ready, self.visual_positions, self.rank)
            self._prefill_pending.enqueue(i, immutable_keys, submit)

    def _prepare_decomposition(self, layer, kv_keys):
        token = self.decode_tokens + 1
        if token < 3:
            if self.max_new_tokens > 3:
                self._start_decompositions(layer, kv_keys)
            if self._dummy_decomposition is None:
                rank = min(self.rank, self.visual_positions.numel(), self.head_dim)
                padded = 1 << (rank - 1).bit_length()
                kv_heads = self.num_heads // self.groups
                self._dummy_decomposition = VisualDecomposition(basis=torch.empty((kv_heads, self.head_dim, padded), dtype=torch.float32, device=kv_keys.device), low_keys=torch.empty((kv_heads, self.visual_positions.numel(), padded), dtype=torch.float32, device=kv_keys.device), explained_energy=torch.zeros(kv_heads, dtype=torch.float32, device=kv_keys.device), effective_rank=rank)
            return self._dummy_decomposition
        if layer in self.decompositions:
            return self.decompositions[layer]
        self._start_decompositions(layer, kv_keys)
        if layer not in self._prefill_pending.futures:
            return super()._prepare_decomposition(layer, kv_keys)
        decomposition, event = self._prefill_pending.futures[layer].result()
        torch.cuda.current_stream().wait_event(event)
        self.raw_energy[self.layer_offset[layer]] = decomposition.explained_energy.mean()
        self.decompositions[layer] = decomposition
        return decomposition

    def finish_async_preparation(self):
        self._prefill_pending.join(torch.cuda.current_stream().wait_event)
