# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Reuse only a validated SVD with identical visual K; no model/state caching."""
import weakref
import torch
from .tuple_merged_text_hold_gate import TupleMergedTextHoldGate as MergedTextHoldGate
from .lazy_batched_hold_gate import LazyPackedBatchedHoldGate
from .parallel_decomposition_hold_gate import install_parallel_hold_gate
from .fast_svd_cache_keys import exact_visual_keys_match
from .svd_cache_keys import unique_storage_bytes
_CACHE = weakref.WeakKeyDictionary()
MAX_CACHE_BYTES = 128 * 1024 * 1024

def clear_cache(model):
    _CACHE.pop(model, None)

class FastCachedSVDHoldGate(MergedTextHoldGate):

    def reset(self):
        pending = getattr(self, '_prefill_pending', None)
        if pending is not None and pending.futures and (not getattr(self, 'cache_published', False)):
            self.finish_async_preparation()
        super().reset()
        self.cache_hit = False
        self.cache_checked = False
        self.cache_published = False
        self.cache_match_bytes = 0
        self._cached_entry = None
        self.backend = 'triton_av32_merged_text_tuple_launch_fast_svd_cache_candidate_v1'

    def _signature(self, keys):
        return (self.layers, self.rank, str(keys.device), str(keys.dtype), self.num_heads, self.head_dim, torch.backends.cuda.matmul.allow_tf32, str(torch.backends.cuda.preferred_linalg_library()))

    def _try_cache(self, keys):
        if self.cache_checked:
            return
        self.cache_checked = True
        entry = _CACHE.get(self.model)
        if entry is None or entry['signature'] != self._signature(keys) or self.decode_cache is None:
            return
        current = {layer: self.decode_cache.layers[layer].keys for layer in self.layers}
        self.cache_match_bytes = 2 * sum((key.shape[1] * self.visual_positions.numel() * key.shape[-1] * key.element_size() for key in current.values()))
        if exact_visual_keys_match(torch, entry, current, self.visual_positions):
            self.cache_hit = True
            self._cached_entry = entry

    def _prepare_decomposition(self, layer, kv_keys):
        if self.max_new_tokens > 3:
            self._try_cache(kv_keys)
        if not self.cache_hit:
            return super()._prepare_decomposition(layer, kv_keys)
        if self.decode_tokens + 1 < 3:
            return LazyPackedBatchedHoldGate._prepare_decomposition(self, layer, kv_keys)
        if layer not in self.decompositions:
            decomposition = self._cached_entry['decompositions'][layer]
            self.raw_energy[self.layer_offset[layer]] = decomposition.explained_energy.mean()
            self.decompositions[layer] = decomposition
        return self.decompositions[layer]

    def finish_async_preparation(self):
        super().finish_async_preparation()
        pending = self._prefill_pending
        if self.cache_hit or self.cache_published or set(pending.futures) != set(self.layers):
            return
        decompositions = {layer: future.result()[0] for layer, future in pending.futures.items()}
        tensors = list(pending.keys.values()) + [self.visual_positions]
        for value in decompositions.values():
            tensors.extend((value.basis, value.low_keys, value.explained_energy))
        if unique_storage_bytes(tensors) <= MAX_CACHE_BYTES:
            _CACHE[self.model] = dict(signature=self._signature(next(iter(pending.keys.values()))), keys=dict(pending.keys), positions=self.visual_positions, decompositions=decompositions)
            self.cache_published = True

    def statistics(self):
        result = super().statistics()
        if self.cache_hit:
            result['svd_gram_and_kb_macs'] = 0
            result['intended_attention_qk_ratio_including_gram_and_kb'] = result['intended_attention_qk_macs_excluding_svd_preprocess'] / max(result['baseline_attention_qk_macs'], 1)
        return result
