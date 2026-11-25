# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Historical backend guards: text eager, vision SDPA; true-HOLD decode is fused."""
from collections import Counter

def assert_eager_model(model):
    expected = {'text': 'eager', 'vision': 'sdpa'}
    actual = {name: getattr(config, '_attn_implementation', None) for name, config in {'text': model.config.text_config, 'vision': model.config.vision_config}.items()}
    if actual != expected:
        raise RuntimeError('Historical attention backend mismatch: ' + str(actual))
    layers = model.model.language_model.layers
    if len(layers) != 36 or any((getattr(layer.self_attn.config, '_attn_implementation', None) != 'eager' for layer in layers)):
        raise RuntimeError('All 36 text attention layers must use eager')
    if getattr(model.model.visual.config, '_attn_implementation', None) != 'sdpa':
        raise RuntimeError('Vision model must retain historical SDPA implementation')
    return actual

def set_historical_backends(model):
    model.config.vision_config._attn_implementation = 'sdpa'
    model.model.visual.config._attn_implementation = 'sdpa'
    return assert_eager_model(model)

def assert_eager_row(row):
    if row.get('model_attention_backend') != {'text': 'eager', 'vision': 'sdpa'}:
        raise RuntimeError('Missing historical text-eager/vision-SDPA provenance')
    expected = 'native_eager' if row['setting'] == 'native_eager' else 'legacy_eager_dense_reference' if row['setting'] == 'legacy_adaptive_m030_reference' else 'triton_cached_values32_v13'
    if row.get('prefill_implementation') != 'native_eager' or row.get('decode_implementation') != expected:
        raise RuntimeError('Text prefill/decode backend does not match eager protocol')

def probe_native_eager(model, batch):
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as modeling
    counts = Counter()
    interface = modeling.ALL_ATTENTION_FUNCTIONS
    original_eager = modeling.eager_attention_forward
    original_sdpa = interface['sdpa']

    def measured_eager(module, query, key, value, attention_mask, *args, **kwargs):
        phase = 'decode' if query.shape[-2] == 1 else 'prefill'
        counts['eager:' + type(module).__name__ + '/' + phase] += 1
        return original_eager(module, query, key, value, attention_mask, *args, **kwargs)

    def measured_sdpa(module, query, key, value, attention_mask, *args, **kwargs):
        phase = 'decode' if query.shape[-2] == 1 else 'prefill'
        counts['sdpa:' + type(module).__name__ + '/' + phase] += 1
        return original_sdpa(module, query, key, value, attention_mask, *args, **kwargs)
    modeling.eager_attention_forward = measured_eager
    interface['sdpa'] = measured_sdpa
    try:
        with torch.inference_mode():
            output = model.generate(**batch, min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, cache_implementation='dynamic')
        text_decode = sum((v for k, v in counts.items() if k.startswith('eager:') and 'TextAttention/decode' in k))
        text_prefill = sum((v for k, v in counts.items() if k.startswith('eager:') and 'TextAttention/prefill' in k))
        vision = sum((v for k, v in counts.items() if k.startswith('sdpa:') and 'VisionAttention/' in k))
        wrong_text = sum((v for k, v in counts.items() if k.startswith('sdpa:') and 'TextAttention/' in k))
        wrong_vision = sum((v for k, v in counts.items() if k.startswith('eager:') and 'VisionAttention/' in k))
        if text_decode < 36 or text_prefill < 36 or vision == 0 or wrong_text or wrong_vision:
            raise RuntimeError('Historical backend function probe failed: ' + str(dict(counts)))
        return dict(passed=True, actual_attention_calls=dict(counts), actual_eager_calls=dict(counts), generated_tokens=2, backend=assert_eager_model(model), accuracy_evaluated=False, throughput_ranked=False)
    finally:
        modeling.eager_attention_forward = original_eager
        interface['sdpa'] = original_sdpa
