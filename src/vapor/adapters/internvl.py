# SPDX-License-Identifier: Apache-2.0
"""InternVL3.5-8B adapter with the native-softmax research backend."""
from contextlib import contextmanager
from .common import REQUEST_LOCK, RequestState, validate, begin
from ..config import VAPORConfig

def build_gate(base):
    from .._backend.internvl.parallel_conditional_hold_gate import native_dense_gate
    from .._backend.internvl.CPU_eigen_prepare import cpu_eigen_gate
    from .._backend.internvl.method_pipeline_graph_hold_gate import batched_gate
    from .._backend.internvl import stable_probability_gate, native_bootstrap_gate
    from .._backend.internvl import static_native_bootstrap, QK_kernel_choice, native_selective_prefix
    from .._backend.internvl import native_softmax_pipeline
    cls = batched_gate(cpu_eigen_gate(native_dense_gate(base)))
    stable_probability_gate.configure(True)
    native_bootstrap_gate.configure(True); cls = native_bootstrap_gate.gate(cls)
    static_native_bootstrap.configure(True); cls = static_native_bootstrap.gate(cls)
    cls = stable_probability_gate.diagnostic_gate(cls)
    QK_kernel_choice.configure(True); cls = QK_kernel_choice.gate(cls)
    native_selective_prefix.configure(2); cls = native_selective_prefix.gate(cls)
    native_softmax_pipeline.configure(False, True)
    return native_softmax_pipeline.gate(cls)

@contextmanager
def vapor(model, fast_tokenizer, input_ids, rendered_prompt, config=None):
    import torch
    from transformers.models.qwen3 import modeling_qwen3 as module
    from .._backend.internvl.internvl_hold_adapter import ModelFacade, install, restore, NativeQueryConsistency
    from .._backend.internvl.hybrid_hold_gate import select_hold_gate, clear_visual_cache, clear_gram_cache
    from .._backend.internvl.query_consistency_online import set_before_first_decode
    from ..prefill import make_query_chunked_eager
    config = config or VAPORConfig.internvl()
    with REQUEST_LOCK:
        validate(model, model.config.llm_config, model.language_model.model.layers, input_ids, config)
        if model.img_context_token_id is None: raise ValueError("Set model.img_context_token_id first")
        facade = ModelFacade(model)
        original_eager = module.eager_attention_forward
        originals = [(x.self_attn, x.self_attn.forward) for x in model.language_model.model.layers]
        manager = observer = None
        try:
            clear_visual_cache(facade); clear_gram_cache(facade)
            base = select_hold_gate(facade, config.legacy_setting(), input_ids.cpu())
            manager = build_gate(base)(facade, 0, 35)
            install(model, manager); begin(manager, input_ids, config)
            state = RequestState(manager, config)
            module.eager_attention_forward = make_query_chunked_eager(
                original_eager, config.prefill_query_block, config.prefill_chunk_threshold)
            if config.calibrate:
                def calibrated(_, info):
                    state.consistency = info['query_consistency_median']
                    state.threshold = config.select_threshold(state.consistency)
                    set_before_first_decode(manager, state.threshold)
                prepared = dict(batch={'input_ids': input_ids.cpu()}, native_rendered_prompt=rendered_prompt)
                observer = NativeQueryConsistency(module, model, fast_tokenizer, prepared, calibrated,
                              original_eager, config.prefill_query_block, config.prefill_chunk_threshold)
            yield state
            if observer is not None and not observer.done:
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
                restore(originals)
                clear_visual_cache(facade); clear_gram_cache(facade)
