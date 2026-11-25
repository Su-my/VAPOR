# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Own pending per-sample decompositions until every GPU consumer is safe."""

def eligible_prefill(active, prompt_length, max_new_tokens, query_length, keys):
    return active and max_new_tokens > 1 and (query_length > 1) and (keys is not None) and (keys.shape[-2] >= prompt_length)

class PendingDecompositions:

    def __init__(self):
        self.futures = {}
        self.keys = {}

    def enqueue(self, layer, keys, submit):
        if layer not in self.futures:
            self.keys[layer] = keys
            self.futures[layer] = submit(keys)
        return self.futures[layer]

    def join(self, wait_event):
        for future in self.futures.values():
            _, event = future.result()
            wait_event(event)

    def clear(self, wait_event):
        self.join(wait_event)
        self.futures.clear()
        self.keys.clear()
