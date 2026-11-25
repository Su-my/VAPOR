# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Same36/32/8/128 HOLD math; native Qwen3 RoPE and module boundary."""
from types import SimpleNamespace, MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

class ModelFacade:

    def __init__(self, model):
        self.model = SimpleNamespace(language_model=model.language_model.model)
        self.config = SimpleNamespace(text_config=model.config.llm_config, image_token_id=model.img_context_token_id, video_token_id=model.img_context_token_id)

def install(model, manager):
    originals = []
    for index in manager.layers:
        att = model.language_model.model.layers[index].self_attn
        original = att.forward
        originals.append((att, original))

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, _original=original, **kwargs):
            manager.decode_cache = past_key_values
            if manager.original(hidden_states, cache_position):
                return _original(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, cache_position=cache_position, **kwargs)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            if past_key_values is not None:
                key, value = past_key_values.update(key, value, self.layer_idx, dict(sin=sin, cos=cos, cache_position=cache_position))
            output = manager.attention(self.layer_idx, query, key, value, attention_mask)
            return (self.o_proj(output.transpose(1, 2).contiguous().reshape(*input_shape, -1)), None)
        att.forward = MethodType(forward, att)
    return originals

def restore(originals):
    for att, original in originals:
        att.forward = original

class NativeQueryConsistency:
    """Original conditional-visual Bhattacharyya statistic, native user span."""

    def __init__(self, module, model, tokenizer, prepared, callback, native, chunk_size=512, threshold=8192):
        from .native_query_chunked_prefill import make_query_chunked_eager
        self.module = module
        self.callback = callback
        self.done = False
        self.native = native
        self.plain = make_query_chunked_eager(native, chunk_size, threshold)
        ids = prepared['batch']['input_ids'][0].tolist()
        rendered = prepared['native_rendered_prompt']
        encoded = tokenizer(rendered, return_offsets_mapping=True)
        assert encoded['input_ids'] == ids, 'Native slow/fast tokenization must match exactly before using offsets'
        header = '<|im_start|>user\n'
        left = rendered.index(header) + len(header)
        end = rendered.index('<|im_end|>', left)
        visual_end = rendered.rfind('</img>', left, end) + len('</img>')
        assert visual_end >= left
        specials = set(tokenizer.all_special_ids)
        self.positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > a and a >= visual_end and (b <= end) and (ids[i] not in specials)]
        assert len(self.positions) >= 2
        self.prompt = len(ids)
        self.visual = torch.tensor([i for i, x in enumerate(ids) if x == model.img_context_token_id], device='cuda', dtype=torch.long)
        self.cursors = [0] * 36
        self.seen = [0] * 36
        self.layers = []
        self.acc = None
        self.current_layer = None
        self.support = torch.zeros(len(self.visual), device='cuda', dtype=torch.float64)

        def observed(owner, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            out, p = native(owner, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
            if not self.done:
                self.observe(owner.layer_idx, query, p)
            return (out, p)
        module.eager_attention_forward = make_query_chunked_eager(observed, chunk_size, threshold)

    def observe(self, layer, query, p):
        from .query_consistency_online import QueryConsistencyPrefill
        QueryConsistencyPrefill.observe(self, layer, query, p)

    def finish(self):
        assert len(self.layers) == 36 and self.cursors == [self.prompt] * 36
        values = torch.stack([v[0] for v in self.layers])
        valid = torch.stack([v[1] for v in self.layers])
        selected = values[valid]
        assert selected.numel() == 1152
        median = float(selected.median())
        neff = float(self.support.sum().square() / self.support.square().sum())
        result = dict(query_consistency_median=median, query_consistency_mean=float(selected.mean()), query_effective_support=neff, query_tokens=len(self.positions), visual_tokens=len(self.visual), valid_pair_heads=1152, same_single_prefill_no_extra_forward=True, same_head_across_query_tokens=True, conditional_visual_distribution=True, exact_native_slow_fast_token_ID_offset_match=True, all_query_tokens_after_last_visual=True, no_generated_tokens_or_answers_used=True, no_prefill_anchor=True)
        self.done = True
        self.module.eager_attention_forward = self.plain
        self.callback(neff, result)

    def close(self):
        self.module.eager_attention_forward = self.plain
        self.acc = None
        self.layers = []
        self.support = None
        self.visual = None
