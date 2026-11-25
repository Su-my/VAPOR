# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Reuse eight SVD threads, each pinned to one CUDA stream; identical math."""
from concurrent.futures import ThreadPoolExecutor
import os
import torch
from .triton_hold_gate import TritonHoldGate
from .parallel_decomposition_hold_gate import ParallelDecompositionHoldGate, install_parallel_hold_gate
from .prepared_launch_hold_gate import PreparedLaunchHoldGate
_RESOURCES = {}

def resources(device):
    key = (os.getpid(), device.index)
    if key not in _RESOURCES:
        _RESOURCES[key] = [(ThreadPoolExecutor(max_workers=1), torch.cuda.Stream(device=device)) for _ in range(8)]
    return _RESOURCES[key]

class PinnedStreamPreparedHoldGate(TritonHoldGate):
    attention = PreparedLaunchHoldGate.attention
    _prepare_one = ParallelDecompositionHoldGate._prepare_one

    def reset(self):
        for future in getattr(self, '_decomposition_futures', {}).values():
            future.result()
        super().reset()
        self._prepared_launches = {}
        self.decode_cache = None
        self._decomposition_futures = {}
        self._decomposition_keys = None
        self.backend = 'triton_av32_pinned_stream_prepared_candidate_v1'

    def _prepare_decomposition(self, layer, kv_keys):
        if self.decode_cache is None:
            return super()._prepare_decomposition(layer, kv_keys)
        if layer in self.decompositions:
            return self.decompositions[layer]
        if not self._decomposition_futures:
            self._decomposition_keys = {i: self.decode_cache.layers[i].keys for i in self.layers}
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
            bank = resources(kv_keys.device)
            for index, i in enumerate(self.layers):
                executor, stream = bank[index % len(bank)]
                self._decomposition_futures[i] = executor.submit(self._prepare_one, self._decomposition_keys[i], stream, ready, self.visual_positions, self.rank)
        decomposition, event = self._decomposition_futures[layer].result()
        torch.cuda.current_stream().wait_event(event)
        self.raw_energy[self.layer_offset[layer]] = decomposition.explained_energy.mean()
        self.decompositions[layer] = decomposition
        return decomposition
