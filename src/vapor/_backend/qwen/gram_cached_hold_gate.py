# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Small exact Gram/eigenvector cache; no K, V, low-key or state caching."""
import weakref
import torch
from .merged_text_hold_gate import MergedTextHoldGate
from .lazy_batched_hold_gate import LazyPackedBatchedHoldGate
from .pinned_stream_prepared_hold_gate import resources
from .parallel_decomposition_hold_gate import install_parallel_hold_gate
from .lean_stats_gate import VisualDecomposition
from .svd_cache_keys import unique_storage_bytes
from .gram_basis_cache import decompose_visual
_CACHE = weakref.WeakKeyDictionary()
MAX_CACHE_BYTES = 128 * 1024 * 1024

def clear_cache(model):
    _CACHE.pop(model, None)

class GramCachedHoldGate(MergedTextHoldGate):

    def reset(self):
        pending = getattr(self, '_prefill_pending', None)
        if pending is not None and pending.futures and (not getattr(self, 'cache_published', False)):
            self.finish_async_preparation()
        super().reset()
        self._basis_cache_entry = None
        self._cache_preparations = {}
        self._cache_hits = set()
        self.cache_hit = False
        self.cache_published = False
        self.cache_match_bytes = 0
        self.gram_cache_hit_layers = 0
        self.cache_bytes = 0
        self.backend = 'triton_av32_merged_text_exact_gram_basis_cache_candidate_v1'

    def _signature(self, keys):
        return (self.layers, self.rank, str(keys.device), str(keys.dtype), self.num_heads, self.head_dim, min(self.rank, self.visual_positions.numel(), self.head_dim), torch.backends.cuda.matmul.allow_tf32, str(torch.backends.cuda.preferred_linalg_library()))

    def _prepare_one_cached(self, layer, keys, stream, ready, positions, rank):
        cached = self._basis_cache_entry['layers'][layer] if self._basis_cache_entry is not None else None
        with torch.cuda.device(keys.device), torch.cuda.stream(stream), torch.inference_mode():
            stream.wait_event(ready)
            keys.record_stream(stream)
            if cached is not None:
                for tensor in (cached['gram'], cached['core_basis'], cached['basis'], cached['energy']):
                    tensor.record_stream(stream)
            visual = keys[0, :, positions, :].float()
            basis, low_keys, energy, effective, entry, hit = decompose_visual(torch, visual, rank, cached)
            self._cache_preparations[layer] = entry
            if hit:
                self._cache_hits.add(layer)
            end = torch.cuda.Event()
            end.record(stream)
            return (VisualDecomposition(basis=basis, low_keys=low_keys, explained_energy=energy, effective_rank=effective), end)

    def _start_decompositions(self, layer, keys):
        if self._prefill_pending.futures:
            return
        entry = _CACHE.get(self.model)
        if entry is not None and entry['signature'] == self._signature(keys):
            self._basis_cache_entry = entry
            self.cache_match_bytes = 2 * sum((x['gram'].numel() * x['gram'].element_size() for x in entry['layers'].values()))
        snapshot = {i: self.decode_cache.layers[i].keys for i in self.layers} if self.decode_cache is not None else {layer: keys}
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        bank = resources(keys.device)
        for i, immutable_keys in snapshot.items():
            executor, stream = bank[self.layer_offset[i] % len(bank)]

            def submit(value, _executor=executor, _stream=stream, _layer=i):
                return _executor.submit(self._prepare_one_cached, _layer, value, _stream, ready, self.visual_positions, self.rank)
            self._prefill_pending.enqueue(i, immutable_keys, submit)

    def _prepare_decomposition(self, layer, kv_keys):
        if self.decode_tokens + 1 < 3:
            if self.max_new_tokens > 3:
                self._start_decompositions(layer, kv_keys)
            return LazyPackedBatchedHoldGate._prepare_decomposition(self, layer, kv_keys)
        if layer in self.decompositions:
            return self.decompositions[layer]
        self._start_decompositions(layer, kv_keys)
        if layer not in self._prefill_pending.futures:
            raise RuntimeError('The admitted decode cache must contain every target layer')
        decomposition, event = self._prefill_pending.futures[layer].result()
        torch.cuda.current_stream().wait_event(event)
        self.raw_energy[self.layer_offset[layer]] = decomposition.explained_energy.mean()
        self.decompositions[layer] = decomposition
        return decomposition

    def finish_async_preparation(self):
        self._prefill_pending.join(torch.cuda.current_stream().wait_event)
        for layer, future in self._prefill_pending.futures.items():
            self.decompositions.setdefault(layer, future.result()[0])
        self.gram_cache_hit_layers = len(self._cache_hits)
        self.cache_hit = bool(self.layers) and self._cache_hits == set(self.layers)
        if self.cache_published or set(self._cache_preparations) != set(self.layers):
            return
        tensors = [tensor for entry in self._cache_preparations.values() for tensor in (entry['gram'], entry['core_basis'], entry['basis'], entry['energy'])]
        self.cache_bytes = unique_storage_bytes(tensors)
        if self.cache_bytes <= MAX_CACHE_BYTES:
            keys = next(iter(self._prefill_pending.keys.values()))
            _CACHE[self.model] = dict(signature=self._signature(keys), layers=dict(self._cache_preparations))
            self.cache_published = True
