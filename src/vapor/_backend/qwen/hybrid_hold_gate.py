# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Choose exact visual-K or exact Gram caching using prompt storage size."""
from .fast_cached_svd_hold_gate import FastCachedSVDHoldGate, clear_cache as clear_visual_cache
from .gram_cached_hold_gate import GramCachedHoldGate, clear_cache as clear_gram_cache
from .hold_cache_policy import cache_family

class CorrectedVisualCachedHoldGate(FastCachedSVDHoldGate):

    def finish_async_preparation(self):
        super().finish_async_preparation()
        if not self.cache_hit:
            for layer, future in self._prefill_pending.futures.items():
                self.decompositions.setdefault(layer, future.result()[0])

def select_hold_gate(model, setting, cpu_input_ids):
    if cpu_input_ids.device.type != 'cpu':
        raise ValueError('Select cache storage using the already prepared CPU prompt')
    config = model.config
    visual = int(((cpu_input_ids == config.image_token_id) | (cpu_input_ids == config.video_token_id)).sum())
    text = config.text_config
    family = cache_family(cpu_input_ids.shape[-1], visual, setting['layers'][1] - setting['layers'][0] + 1, text.num_key_value_heads, text.hidden_size // text.num_attention_heads, setting['rank'])
    if family == 'visual_keys':
        clear_gram_cache(model)
        return CorrectedVisualCachedHoldGate
    clear_visual_cache(model)
    return GramCachedHoldGate
