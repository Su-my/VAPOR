# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Prepare original per-layer SVD concurrently; attention arithmetic unchanged."""
from concurrent.futures import ThreadPoolExecutor
from types import MethodType
import torch
from .triton_hold_gate import TritonHoldGate
from .lean_stats_gate import VisualDecomposition
from .real_hold_gate import install_real_hold_gate

class ParallelDecompositionHoldGate(TritonHoldGate):

    def reset(self):
        pool = getattr(self, '_decomposition_pool', None)
        if pool is not None:
            pool.shutdown(wait=True)
        super().reset()
        self.backend = 'triton_av32_parallel_svd_candidate_v1'
        self.decode_cache = None
        self._decomposition_pool = None
        self._decomposition_futures = {}
        self._decomposition_keys = None

    def _prepare_one(self, keys, stream, ready, positions, rank):
        with torch.cuda.device(keys.device), torch.cuda.stream(stream), torch.inference_mode():
            stream.wait_event(ready)
            keys.record_stream(stream)
            visual = keys[0, :, positions, :].float()
            eigenvalues, basis = torch.linalg.eigh(visual.transpose(-1, -2) @ visual)
            eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
            effective = min(rank, visual.shape[-2], visual.shape[-1])
            basis = basis.flip(-1)[..., :effective].contiguous()
            low_keys = (visual @ basis).contiguous()
            padded = 1 << (effective - 1).bit_length()
            if padded != effective:
                basis = torch.nn.functional.pad(basis, (0, padded - effective)).contiguous()
                low_keys = torch.nn.functional.pad(low_keys, (0, padded - effective)).contiguous()
            energy = eigenvalues[..., :effective].sum(-1) / eigenvalues.sum(-1).clamp_min(1e-20)
            end = torch.cuda.Event()
            end.record(stream)
            return (VisualDecomposition(basis=basis, low_keys=low_keys, explained_energy=energy, effective_rank=effective), end)

    def _prepare_decomposition(self, layer, kv_keys):
        if self.decode_cache is None:
            return super()._prepare_decomposition(layer, kv_keys)
        if layer in self.decompositions:
            return self.decompositions[layer]
        if self._decomposition_pool is None:
            self._decomposition_keys = {i: self.decode_cache.layers[i].keys for i in self.layers}
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
            self._decomposition_pool = ThreadPoolExecutor(max_workers=8)
            for i in self.layers:
                keys = self._decomposition_keys[i]
                stream = torch.cuda.Stream(device=keys.device)
                self._decomposition_futures[i] = self._decomposition_pool.submit(self._prepare_one, keys, stream, ready, self.visual_positions, self.rank)
        decomposition, event = self._decomposition_futures[layer].result()
        torch.cuda.current_stream().wait_event(event)
        self.raw_energy[self.layer_offset[layer]] = decomposition.explained_energy.mean()
        self.decompositions[layer] = decomposition
        return decomposition

def install_parallel_hold_gate(model, manager):
    install_real_hold_gate(model, manager)
    for layer in manager.layers:
        attention = model.model.language_model.layers[layer].self_attn
        bound = attention.forward

        def capture(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, _bound=bound, **kwargs):
            manager.decode_cache = past_key_values
            return _bound(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, cache_position=cache_position, **kwargs)
        attention.forward = MethodType(capture, attention)
