# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Exact C435 Bhattacharyya mean; no cosine/entropy work in production."""
import torch
from .user_text_reference_hook import build_user_text_mask

class QueryConsistencyPrefill:

    def __init__(self, model, tokenizer, base_eager, plain_eager, plain_prefill):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as module
        self.module = module
        self.model = model
        self.tokenizer = tokenizer
        self.attentions = [x.self_attn for x in model.model.language_model.layers]
        self.plain_eager = plain_eager
        self.plain_prefill = plain_prefill
        self.active = False
        self.completed = False
        self.callback = None

        def observed(owner, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            result = base_eager(owner, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
            layer = getattr(owner, 'layer_idx', None)
            if self.active and layer is not None and (self.attentions[layer] is owner):
                self.observe(layer, query, result[1])
            return result
        self.observed_eager = observed

    def arm(self, cpu_input_ids, device, callback):
        assert not self.active and self.module.eager_attention_forward is self.plain_eager
        assert cpu_input_ids.device.type == 'cpu'
        ids = cpu_input_ids[0].tolist()
        self.prompt = len(ids)
        cfg = self.model.config
        visual = [i for i, x in enumerate(ids) if x in [cfg.image_token_id, cfg.video_token_id]]
        assert visual
        mask, self.audit = build_user_text_mask(ids, self.tokenizer)
        self.positions = [i for i, yes in enumerate(mask) if yes and i > visual[-1]]
        assert self.positions
        self.visual = torch.tensor(visual, device=device, dtype=torch.long)
        self.support = torch.zeros(len(visual), device=device, dtype=torch.float64)
        self.cursors = [0] * 36
        self.seen = [0] * 36
        self.layers = []
        self.acc = None
        self.current_layer = None
        self.callback = callback
        self.completed = False
        self.last_result = None
        self.module.eager_attention_forward = self.observed_eager
        self.active = True

    def observe(self, layer, query, p):
        start = self.cursors[layer]
        end = start + query.shape[-2]
        self.cursors[layer] = end
        assert end <= self.prompt
        selected = [j for j in self.positions if start <= j < end]
        if selected:
            assert p is not None
            if self.acc is None:
                self.current_layer = layer
                self.acc = dict(root=torch.zeros((32, self.visual.numel()), device=p.device, dtype=torch.float64), diagonal=torch.zeros(32, device=p.device, dtype=torch.float64), n=torch.zeros(32, device=p.device, dtype=torch.float64))
            assert self.current_layer == layer
            for offset in range(0, len(selected), 8):
                loc = torch.tensor([j - start for j in selected[offset:offset + 8]], device=p.device)
                actual = p[0].index_select(1, loc)
                denominator = actual.sum(-1, dtype=torch.float32)
                visual = actual.index_select(2, self.visual).float() / denominator[:, :, None]
                self.support += visual.sum((0, 1), dtype=torch.float64)
                mass = visual.sum(-1)
                valid = mass > 0
                conditional = visual / torch.where(valid, mass, torch.ones_like(mass))[:, :, None]
                root = conditional.sqrt()
                self.acc['root'] += root.sum(1, dtype=torch.float64)
                self.acc['diagonal'] += root.square().sum((1, 2), dtype=torch.float64)
                self.acc['n'] += valid.sum(1)
            self.seen[layer] += len(selected)
        if end == self.prompt:
            assert self.current_layer == layer and self.seen[layer] == len(self.positions)
            a = self.acc
            n = a['n']
            value = ((a['root'].square().sum(-1) - a['diagonal']) / (n * (n - 1)).clamp_min(1)).clamp(0, 1)
            self.layers.append((value, n >= 2))
            self.acc = None
            self.current_layer = None
        if layer == 35 and self.cursors == [self.prompt] * 36:
            self.finish()

    def finish(self):
        assert self.active and len(self.layers) == 36
        values = torch.stack([x[0] for x in self.layers])
        valid = torch.stack([x[1] for x in self.layers])
        selected = values[valid]
        neff = float(self.support.sum().square() / self.support.square().sum())
        assert selected.numel() == 1152
        c = float(selected.mean())
        median = float(selected.median())
        result = dict(query_consistency_mean=c, query_consistency_median=median, query_effective_support=neff, query_tokens=len(self.positions), visual_tokens=self.visual.numel(), prompt_tokens=self.prompt, observed_query_layer_head_rows=36 * 32 * len(self.positions), valid_pair_heads=1152, same_single_prefill_no_extra_forward=True, native_attention_output_tuple_unchanged=True, same_head_across_query_tokens=True, conditional_visual_distribution=True, coverage_not_used_for_threshold=True, no_cosine_or_JS_entropy_computed=True, no_distribution_or_token_head_trace_saved=True, query_mask_audit=self.audit)
        callback = self.callback
        self.last_result = result
        self.completed = True
        self.close()
        callback(neff, result)

    def close(self):
        self.module.eager_attention_forward = self.plain_eager
        self.active = False
        self.callback = None
        for name in ['support', 'visual', 'acc', 'layers']:
            if hasattr(self, name):
                delattr(self, name)

def set_before_first_decode(manager, tau):
    assert 0.0 < tau < 1.0 and manager.decode_tokens == 0 and (not manager.states)
    for name in ['_prepared_launches', '_host_batches', '_host_prototypes']:
        assert not getattr(manager, name, {})
    for name in ['m1', 'confirmation_m1', 'm3_base', 'sample_effective_m1', 'sample_effective_confirmation_m1', 'sample_effective_m3_base']:
        setattr(manager, name, tau)
