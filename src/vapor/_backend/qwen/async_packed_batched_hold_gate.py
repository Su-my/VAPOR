# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Original SVD overlaps mandatory Dense queries; exact batched HOLD launch."""
from .lazy_batched_hold_gate import LazyPackedBatchedHoldGate
from .async_first_decode_hold_gate import AsyncFirstDecodeHoldGate
from .parallel_decomposition_hold_gate import install_parallel_hold_gate

class AsyncPackedBatchedHoldGate(LazyPackedBatchedHoldGate):

    def _prepare_decomposition(self, layer, kv_keys):
        if self.decode_tokens + 1 < 3 and self.max_new_tokens > 3:
            AsyncFirstDecodeHoldGate._start_decompositions(self, layer, kv_keys)
        return super()._prepare_decomposition(layer, kv_keys)
