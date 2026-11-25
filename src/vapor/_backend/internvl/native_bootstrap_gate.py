# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Use canonical eager only for the two already-mandatory Dense queries."""
from .native_selective_hold import NativeSelectiveMixin, load, threshold_table
import torch
ENABLED = False
_CLASSES = {}

def configure(enabled):
    global ENABLED
    ENABLED = bool(enabled)

class NativeBootstrapMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._native_bootstrap_enabled = ENABLED
        self._native_bootstrap_calls = 0
        if ENABLED:
            self._selective_cpp = load()
            self._energy_used = set()
            self._all_text_positions = torch.nonzero(~self.visual_flags, as_tuple=False).flatten()
            self._age_threshold_lut = threshold_table(self.max_anchor_distance, self.m3_base, query.device)

    def attention(self, layer, query, keys, values, mask=None):
        if getattr(self, '_native_bootstrap_enabled', ENABLED) and self.decode_tokens + 1 <= 2:
            if self.work_counts is None:
                self._prepare_buffers(query)
            state = self._state(layer, self.head_dim, query.device)
            self._native_bootstrap_calls += 1
            return NativeSelectiveMixin.attention(self, layer, query, keys, values, mask)
        return super().attention(layer, query, keys, values, mask)

    def statistics(self):
        r = super().statistics()
        enabled = getattr(self, '_native_bootstrap_enabled', ENABLED)
        r.update(native_mandatory_Dense_bootstrap=enabled, native_bootstrap_layer_calls=getattr(self, '_native_bootstrap_calls', 0), native_bootstrap_cached_queries=2 if enabled else 0, hot_loop_host_head_selection=enabled, host_selection_scope='Only the two already-mandatory Dense cached queries; fast GPU routing thereafter' if enabled else 'No hot-loop host selection', ordinary_HOLD_visual_Dense_computed=False, additional_forced_Dense_queries=0)
        return r

def gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('NativeBootstrap' + base.__name__, (NativeBootstrapMixin, base), {})
    return _CLASSES[base]
