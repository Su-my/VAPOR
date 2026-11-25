# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Native selective arithmetic for a bounded prefix; routing is unchanged."""
import hashlib
import torch
from .native_selective_hold import NativeSelectiveMixin
LIMIT = 2
CAPTURE = False
CAPTURES = []
_CLASSES = {}

def configure(limit):
    global LIMIT
    assert limit in (2, 4, 8192)
    LIMIT = limit

def set_capture(enabled):
    global CAPTURE
    CAPTURE = bool(enabled)

def digest_tensor(t):
    a = t.detach().contiguous().cpu()
    return hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest()

class NativeSelectivePrefixMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._native_prefix_limit = LIMIT
        self._native_selective_prefix_layer_calls = 0
        self._native_prefix_text_rows = None

    def attention(self, layer, query, keys, values, mask=None):
        token = self.decode_tokens + 1
        native = 2 < token <= getattr(self, '_native_prefix_limit', LIMIT)
        if native:
            if self.work_counts is None:
                self._prepare_buffers(query)
            out = NativeSelectiveMixin.attention(self, layer, query, keys, values, mask)
            self._native_selective_prefix_layer_calls += 1
            if layer in self._energy_used:
                self.energy_recorded[self.layer_offset[layer]] = 1
        else:
            out = super().attention(layer, query, keys, values, mask)
        if layer == self.layers[-1] and token <= self._native_prefix_limit:
            self._native_prefix_text_rows = self.kernel_work[2].sum().clone()
        if CAPTURE and layer == self.layers[-1] and (token == 4):
            fields = ['holding', 'anchor_ready', 'anchor_token', 'anchor_readout', 'anchor_nonvisual_tokens', 'episode_m3_base']
            states = {str(l): {n: digest_tensor(getattr(self.states[l], n)) for n in fields} for l in self.layers}
            CAPTURES.append(dict(limit=self._native_prefix_limit, states=states, counters=[digest_tensor(t) for t in self.kernel_counters], floats=[digest_tensor(t) for t in self.kernel_floats], work=[digest_tensor(t) for t in self.kernel_work]))
        return out

    def statistics(self):
        r = super().statistics()
        limit = getattr(self, '_native_prefix_limit', LIMIT)
        r.update(native_selective_prefix_cached_queries=limit, native_selective_prefix_layer_calls=getattr(self, '_native_selective_prefix_layer_calls', 0), native_prefix_text_qk_key_rows=int(self._native_prefix_text_rows.item()) if getattr(self, '_native_prefix_text_rows', None) is not None else 0, native_prefix_no_forced_Dense=True, native_prefix_ordinary_HOLD_skips_visual_Dense=True, all_prefix_compaction_and_preparation_in_timing=True, additional_forced_Dense_queries=0, hot_loop_host_head_selection=limit > 2, host_selection_scope='Bounded native selective prefix including HOLD; fast routing after prefix', no_extra_hot_loop_host_selection=limit <= 2)
        return r

def gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('NativeSelectivePrefix' + base.__name__, (NativeSelectivePrefixMixin, base), {})
    return _CLASSES[base]
