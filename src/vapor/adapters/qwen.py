# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-8B-Instruct adapter for the extracted five-kernel backend."""
from contextlib import contextmanager
from .common import REQUEST_LOCK, RequestState, validate, begin
from ..config import VAPORConfig

@contextmanager
def vapor(model, tokenizer, input_ids, config=None):
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as module
    from .._backend.qwen.hybrid_hold_gate import select_hold_gate, clear_visual_cache, clear_gram_cache
    from .._backend.qwen.parallel_proxy_hold_gate import ParallelProxyHoldGate
    from .._backend.qwen.parallel_decomposition_hold_gate import install_parallel_hold_gate
    from .._backend.qwen.query_consistency_online import QueryConsistencyPrefill, set_before_first_decode
    from ..prefill import make_query_chunked_eager
    config = config or VAPORConfig()
    with REQUEST_LOCK:
        layers = model.model.language_model.layers
        validate(model, model.config.text_config, layers, input_ids, config)
        original_eager = module.eager_attention_forward
        originals = [(x.self_attn, x.self_attn.forward) for x in layers]
        manager = observer = None
        clear_visual_cache(model); clear_gram_cache(model)
        try:
            base = select_hold_gate(model, config.legacy_setting(), input_ids.cpu())
            cls = type('VAPORQwen', (base,), {'attention': ParallelProxyHoldGate.attention})
            manager = cls(model, 0, 35)
            install_parallel_hold_gate(model, manager)
            begin(manager, input_ids, config)
            state = RequestState(manager, config)
            plain = make_query_chunked_eager(original_eager, config.prefill_query_block,
                                            config.prefill_chunk_threshold)
            module.eager_attention_forward = plain
            if config.calibrate:
                def calibrated(_, info):
                    state.consistency = info['query_consistency_median']
                    state.threshold = config.select_threshold(state.consistency)
                    set_before_first_decode(manager, state.threshold)
                observer = QueryConsistencyPrefill(model, tokenizer, original_eager, plain, None)
                observer.observed_eager = make_query_chunked_eager(
                    observer.observed_eager, config.prefill_query_block, config.prefill_chunk_threshold)
                observer.arm(input_ids.cpu(), torch.device('cuda:0'), calibrated)
            yield state
            if observer is not None and not observer.completed:
                raise RuntimeError("No complete multimodal prefill was observed")
        finally:
            try:
                if manager is not None:
                    manager.active = False
                    manager.finish_async_preparation()
                    torch.cuda.synchronize()
            finally:
                if observer is not None: observer.close()
                module.eager_attention_forward = original_eager
                for attention, forward in originals:
                    attention.forward = forward
                    for name in ('_svd_checkpoint_hold_manager', '_svd_checkpoint_hold_original_forward',
                                 '_svd_checkpoint_hold_wrapped'):
                        if hasattr(attention, name): delattr(attention, name)
                clear_visual_cache(model); clear_gram_cache(model)
