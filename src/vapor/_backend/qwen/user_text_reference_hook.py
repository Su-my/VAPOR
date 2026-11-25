# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Read-only original-user-text reference; never supplies an output anchor.

This module has CPU tensor fixtures only. It is not a model/GPU qualification.
"""

def build_user_text_mask(original_ids, tokenizer):
    """Fail closed unless decoded offsets reproduce the exact rendered input IDs."""
    ids = list(map(int, original_ids))
    rendered = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert list(encoded['input_ids']) == ids, 'Tokenizer offset round-trip changed input IDs'
    offsets = encoded['offset_mapping']
    assert len(offsets) == len(ids)
    start, end = ('<|im_start|>', '<|im_end|>')
    visual_start, visual_end = ('<|vision_start|>', '<|vision_end|>')
    user_spans, visual_spans = ([], [])
    position = 0
    while True:
        message_start = rendered.find(start, position)
        if message_start < 0:
            break
        header = message_start + len(start)
        newline = rendered.find('\n', header)
        assert newline >= 0, 'Incomplete role header'
        message_end = rendered.find(end, newline + 1)
        following_start = rendered.find(start, newline + 1)
        if message_end < 0:
            assert rendered[header:newline] == 'assistant'
            assert not rendered[newline + 1:], 'Only an empty assistant generation header may be open'
            break
        assert following_start < 0 or following_start > message_end, 'Nested/ambiguous role boundary'
        if rendered[header:newline] == 'user':
            user_spans.append((newline + 1, message_end))
        position = message_end + len(end)
    assert user_spans, 'No original user message'
    position = 0
    while True:
        left = rendered.find(visual_start, position)
        if left < 0:
            break
        right = rendered.find(visual_end, left + len(visual_start))
        assert right >= 0, 'Unclosed visual span'
        right += len(visual_end)
        visual_spans.append((left, right))
        position = right
    special = set(tokenizer.all_special_ids)
    mask = [token not in special and hi > lo and any((left <= lo and hi <= right for left, right in user_spans)) and (not any((lo < right and hi > left for left, right in visual_spans))) for token, (lo, hi) in zip(ids, offsets)]
    assert any(mask), 'No eligible user-text queries'
    return (mask, dict(prompt_tokens=len(ids), user_text_query_tokens=sum(mask), user_messages=len(user_spans), visual_spans=len(visual_spans), exact_input_ID_offset_roundtrip=True, excluded_system_assistant_visual_and_special_tokens=True))

class UserTextReferenceState:

    def __init__(self, manager, layer, query_mask):
        assert len(query_mask) > 1 and any(query_mask)
        self.manager = manager
        self.module = manager.model.model.language_model.layers[layer].self_attn
        self.query_mask = tuple(map(bool, query_mask))
        self.prompt_query_tokens = len(query_mask)
        self.seen_queries = self.chunk_calls = self.selected_queries = 0
        self.raw = None
        self.calls = 0

    def observe(self, module, query, result):
        if module is not self.module:
            return
        count = query.shape[-2]
        if self.seen_queries == self.prompt_query_tokens:
            assert count == 1, 'Repeated prefill is not a new reference'
            return
        assert 0 < count <= self.prompt_query_tokens - self.seen_queries
        import torch
        weights = result[1]
        assert weights is not None and weights.shape[-2] == count
        assert weights.shape[0] == 1 and weights.shape[1] == self.manager.num_heads
        if self.raw is None:
            self.raw = torch.zeros((self.manager.num_heads, self.manager.visual_positions.numel()), device=weights.device, dtype=torch.float32)
        local = self.query_mask[self.seen_queries:self.seen_queries + count]
        selected = [i for i, yes in enumerate(local) if yes]
        for offset in range(0, len(selected), 16):
            positions = torch.tensor(selected[offset:offset + 16], device=weights.device, dtype=torch.long)
            visual = weights[0].index_select(-2, positions).index_select(-1, self.manager.visual_positions)
            self.raw.add_(visual.float().sum(-2))
        self.selected_queries += len(selected)
        self.seen_queries += count
        self.chunk_calls += 1
        if self.seen_queries == self.prompt_query_tokens:
            assert self.selected_queries == sum(self.query_mask)
            self.calls = 1

    def probability(self):
        import torch
        assert self.calls == 1 and self.raw is not None
        mass = self.raw.sum(-1)
        valid = torch.isfinite(self.raw).all(-1) & torch.isfinite(mass) & (mass > 0)
        return (self.raw / mass[:, None].clamp_min(1e-30), mass, valid)

class UserTextReferenceTape:

    def __init__(self, modeling_module):
        self.modeling = modeling_module
        self.original = modeling_module.eager_attention_forward
        self.active = None

        def wrapped(module, query, key, value, *args, **kwargs):
            result = self.original(module, query, key, value, *args, **kwargs)
            if self.active is not None:
                self.active.observe(module, query, result)
            return result
        self.wrapped = wrapped
        modeling_module.eager_attention_forward = wrapped

    def activate(self, manager, layer, query_mask):
        assert self.active is None
        self.active = UserTextReferenceState(manager, layer, query_mask)
        return self.active

    def deactivate(self, state):
        assert self.active is state
        self.active = None

    def restore_unwrapped_owner(self):
        assert self.active is None
        assert self.modeling.eager_attention_forward is self.wrapped
        self.modeling.eager_attention_forward = self.original
