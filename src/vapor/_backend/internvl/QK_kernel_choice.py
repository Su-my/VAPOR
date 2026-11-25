# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Change only method QK arithmetic; keep common model operators untouched."""
from .cached_triton_dispatch import CachedDispatch
from . import parallel_conditional_hold_gate as reference
from . import parallel_conditional_kernels as auxiliary
from .tensorcore_text_QK import merged_text_logits as tensor_text
from .tensorcore_visual_QK import dense_visual_logits as tensor_visual
tensor_visual = tensor_visual.function
ORIGINAL_TEXT = reference.merged_text_logits
ORIGINAL_VISUAL = auxiliary.dense_visual_logits
ENABLED = False
_CLASSES = {}

def configure(enabled):
    global ENABLED
    ENABLED = bool(enabled)
    reference.merged_text_logits = ORIGINAL_TEXT
    auxiliary.dense_visual_logits = CachedDispatch(tensor_visual) if enabled else ORIGINAL_VISUAL

class QKChoiceMixin:

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._tensorcore_QK_enabled = ENABLED

    def statistics(self):
        r = super().statistics()
        enabled = getattr(self, '_tensorcore_QK_enabled', ENABLED)
        r.update(method_text_QK_without_absolute_dot_and_FP64_correction=False, method_visual_QK_tensorcore=enabled, mandatory_text_absolute_dot_executed=True, sparse_text_FP64_correction_executed=True, visual_Dense_QK_FP64_executed=not enabled, QK_common_projection_unchanged=True)
        return r

def gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('QKChoice' + base.__name__, (QKChoiceMixin, base), {})
    return _CLASSES[base]
