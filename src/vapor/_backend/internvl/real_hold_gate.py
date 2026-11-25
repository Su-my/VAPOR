# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""True online HOLD: selective Dense, exact text, rank-r visual proxy.

This PyTorch backend is the correctness oracle for subsequent GPU kernels.
It does not compute full visual QK or AV on ordinary HOLD head rows.
Frozen reference files are never replaced. CPU tests use synthetic tensors only.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
from .lean_stats_gate import SVDCheckpointHoldGate as ReferenceGate, EPS
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

class RealHoldGate(ReferenceGate):

    def reset(self):
        super().reset()
        self.execution_dense_visual_rows = 0
        self.execution_skipped_visual_rows = 0
        self.execution_text_proxy_rows = 0
        self.backend = 'torch_selective_oracle'

    def statistics(self):
        result = super().statistics()
        result.update(actual_visual_qk_saving=True, execution_backend=self.backend, execution_dense_visual_rows=self.execution_dense_visual_rows, execution_skipped_visual_rows=self.execution_skipped_visual_rows, execution_text_proxy_rows=self.execution_text_proxy_rows, proxy_mae_scope='proxy_confirmation_rows_with_algorithm_required_exact_mass', ordinary_hold_exact_mass_computed=False, trace_file=None)
        return result

    def attention(self, layer, query, kv_keys, kv_values, attention_mask=None):
        if query.shape[:3] != (1, self.num_heads, 1):
            raise ValueError('Selective decode requires batch=1 and one query token')
        token = self.decode_tokens + 1
        if token > self.max_new_tokens:
            raise RuntimeError('Generation exceeds frozen output cap')
        offset = self.layer_offset[layer]
        positions = self.visual_positions.to(query.device)
        n_total_keys = kv_keys.shape[-2]
        n_text_keys = n_total_keys - positions.numel()
        self.last_nonvisual_tokens = n_text_keys
        self.text_key_rows += self.num_heads * n_text_keys
        self.total_key_rows += self.num_heads * n_total_keys
        state = self._state(layer, kv_values.shape[-1], query.device)
        holding_before = state.holding.clone()
        anchor_before = state.anchor_token.clone()
        distance = torch.where(anchor_before >= 0, token - anchor_before, -1)
        phase = ((distance - 2).clamp_min(0).float() / max(self.max_anchor_distance - 2, 1)).clamp(0.0, 1.0)
        current_m1 = self.m1
        current_confirmation_m1 = self.confirmation_m1
        current_m3_base = state.episode_m3_base if self.entry_mass_m3_gate else self.m3_base
        threshold = current_m3_base * torch.cos(0.5 * math.pi * phase)
        threshold = torch.where(distance >= 0, threshold, torch.zeros_like(threshold))
        checkpoint_due, _, _ = self._checkpoint_due(holding_before, state.anchor_ready, distance, token, offset, self.num_heads, self.max_anchor_distance, self.checkpoint_schedule)
        proxy_checked = self._proxy_mask(holding_before, state.anchor_ready, distance, self.max_anchor_distance, checkpoint_due)
        proxy_heads = torch.nonzero(proxy_checked, as_tuple=False).flatten()
        proxy_mass = torch.full((self.num_heads,), float('nan'), dtype=torch.float32, device=query.device)
        text_readout = torch.zeros((self.num_heads, kv_values.shape[-1]), dtype=torch.float32, device=query.device)
        scaling = float(self.model.model.language_model.layers[layer].self_attn.scaling)
        if proxy_heads.numel():
            text_mask = torch.ones(n_total_keys, dtype=torch.bool, device=query.device)
            text_mask[positions] = False
            text_positions = torch.nonzero(text_mask, as_tuple=False).flatten()
            kv_index = proxy_heads // self.groups
            text_keys = kv_keys[0, kv_index][:, text_positions, :]
            text_logits = (query[0, proxy_heads, 0, :].unsqueeze(1) @ text_keys.transpose(-1, -2)).squeeze(1) * scaling
            if attention_mask is not None:
                text_logits += attention_mask[0, 0, 0, text_positions]
            text_logits = text_logits.float()
            log_z_text = torch.logsumexp(text_logits, dim=-1)
            decomposition = self.decompositions.get(layer)
            if decomposition is None:
                decomposition = self._decompose(layer, kv_keys)
            low_query = torch.einsum('hd,hdr->hr', query[0, proxy_heads, 0, :].float(), decomposition.basis[kv_index])
            approximate_visual_logits = torch.einsum('hr,hvr->hv', low_query, decomposition.low_keys[kv_index]) * scaling
            if attention_mask is not None:
                approximate_visual_logits += attention_mask[0, 0, 0, positions].float()
            proxy_mass[proxy_heads] = torch.sigmoid(torch.logsumexp(approximate_visual_logits, dim=-1) - log_z_text)
            hold_local = proxy_mass[proxy_heads] < threshold[proxy_heads]
            hold_indices = torch.nonzero(hold_local, as_tuple=False).flatten()
            if hold_indices.numel():
                held_heads = proxy_heads[hold_indices]
                text_values = kv_values[0, held_heads // self.groups][:, text_positions, :].float()
                text_probs = F.softmax(text_logits[hold_indices], dim=-1, dtype=torch.float32)
                text_readout[held_heads] = torch.einsum('hs,hsd->hd', text_probs, text_values)
        stay_hold_planned = proxy_checked & (proxy_mass < threshold)
        dense_heads = torch.nonzero(~stay_hold_planned, as_tuple=False).flatten()
        exact_mass = torch.full_like(proxy_mass, float('nan'))
        exact_visual_readout = torch.full_like(text_readout, float('nan'))
        dense_output = torch.zeros_like(query)
        if dense_heads.numel():
            kv_index = dense_heads // self.groups
            keys = kv_keys[0, kv_index]
            values = kv_values[0, kv_index]
            dense_logits = query[0, dense_heads, 0, :].unsqueeze(1) @ keys.transpose(-1, -2) * scaling
            if attention_mask is not None:
                dense_logits += attention_mask[0, 0, :1, :n_total_keys]
            probabilities = F.softmax(dense_logits, dim=-1, dtype=torch.float32).to(query.dtype)
            dense_output[0, dense_heads, 0, :] = (probabilities @ values).squeeze(1)
            visual_weights = probabilities[:, 0, positions].float()
            exact_mass[dense_heads] = visual_weights.sum(-1)
            visual_values = values[:, positions, :].float()
            exact_visual = torch.einsum('hv,hvd->hd', visual_weights, visual_values)
            mass = exact_mass[dense_heads]
            exact_visual_readout[dense_heads] = torch.where(mass[:, None] > 0, exact_visual / mass.clamp_min(EPS)[:, None], torch.zeros_like(exact_visual))
        if token == 1:
            self.first_token_exact_mass[offset] = exact_mass.detach().double()
        self.execution_dense_visual_rows += dense_heads.numel()
        self.execution_skipped_visual_rows += self.num_heads - dense_heads.numel()
        self.execution_text_proxy_rows += proxy_heads.numel()
        confirmation_threshold = torch.full_like(exact_mass, current_confirmation_m1)
        decision = self._transition(holding_before, proxy_checked, proxy_mass, threshold, exact_mass, current_m1, state.anchor_ready, distance, self.max_anchor_distance, confirmation_m1=confirmation_threshold, checkpoint_due=checkpoint_due)
        stay_hold = decision['stay_hold']
        confirmation = decision['confirmation']
        confirmation_passed = decision['confirmation_passed']
        confirmation_failed = decision['confirmation_failed']
        relaxed_confirmation_pass = confirmation_passed & (exact_mass < current_m1)
        horizon_confirmation = decision['horizon_confirmation']
        checkpoint_reanchor = decision['checkpoint_reanchor']
        proxy_confirmation = confirmation & proxy_checked
        proxy_confirmation_low = proxy_confirmation & confirmation_failed
        proxy_confirmation_high = proxy_confirmation & confirmation_passed
        horizon_confirmation_low = horizon_confirmation & confirmation_failed
        horizon_confirmation_high = horizon_confirmation & confirmation_passed
        enter = decision['enter']
        reopen = decision['reopen']
        dense_now = decision['dense_now']
        holding_after = decision['holding_after']
        held_output = (1.0 - proxy_mass)[:, None] * text_readout + proxy_mass[:, None] * state.anchor_readout
        output = torch.where(stay_hold[None, :, None, None], held_output[None, :, None, :].to(dense_output.dtype), dense_output)
        update_anchor = dense_now
        episode_m3_update = torch.zeros_like(enter)
        episode_m3_value = torch.full_like(exact_mass, float('nan'))
        episode_m3_floor = torch.zeros_like(episode_m3_update)
        episode_m3_ceiling = torch.zeros_like(episode_m3_update)
        if self.entry_mass_m3_gate:
            episode_m3_update = enter | checkpoint_reanchor
            state.episode_m3_base, episode_m3_value, episode_m3_floor, episode_m3_ceiling = self._update_episode_m3_base(state.episode_m3_base, exact_mass.detach(), episode_m3_update, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max)
        state.anchor_readout = torch.where(dense_now[:, None], exact_visual_readout.detach(), state.anchor_readout)
        state.anchor_token = torch.where(update_anchor, torch.full_like(state.anchor_token, token), state.anchor_token)
        state.anchor_nonvisual_tokens = torch.where(update_anchor, torch.full_like(state.anchor_nonvisual_tokens, n_text_keys), state.anchor_nonvisual_tokens)
        state.anchor_ready |= update_anchor
        state.holding = holding_after
        self.dense_rows[offset] += dense_now.to(torch.int64)
        self.hold_rows[offset] += stay_hold.to(torch.int64)
        self.enter_rows[offset] += enter.to(torch.int64)
        self.reopen_rows[offset] += reopen.to(torch.int64)
        self.proxy_check_rows[offset] += proxy_checked.to(torch.int64)
        self.confirmation_rows[offset] += confirmation.to(torch.int64)
        self.confirmation_low_rows[offset] += confirmation_failed.to(torch.int64)
        self.confirmation_high_rows[offset] += confirmation_passed.to(torch.int64)
        self.relaxed_confirmation_pass_rows[offset] += relaxed_confirmation_pass.to(torch.int64)
        self.proxy_confirmation_rows[offset] += proxy_confirmation.to(torch.int64)
        self.proxy_confirmation_low_rows[offset] += proxy_confirmation_low.to(torch.int64)
        self.proxy_confirmation_high_rows[offset] += proxy_confirmation_high.to(torch.int64)
        self.horizon_confirmation_rows[offset] += horizon_confirmation.to(torch.int64)
        self.horizon_confirmation_low_rows[offset] += horizon_confirmation_low.to(torch.int64)
        self.horizon_confirmation_high_rows[offset] += horizon_confirmation_high.to(torch.int64)
        self.checkpoint_reanchor_rows[offset] += checkpoint_reanchor.to(torch.int64)
        self.entry_mass_m3_update_rows[offset] += episode_m3_update.to(torch.int64)
        self.entry_mass_m3_value_sum[offset] += torch.where(episode_m3_update, episode_m3_value.double(), torch.zeros_like(episode_m3_value, dtype=torch.float64))
        self.entry_mass_m3_floor_rows[offset] += episode_m3_floor.to(torch.int64)
        self.entry_mass_m3_ceiling_rows[offset] += episode_m3_ceiling.to(torch.int64)
        self.holding_candidate_rows[offset] += holding_before.to(torch.int64)
        forced = horizon_confirmation
        self.distance_forced_rows[offset] += forced.to(torch.int64)
        error_heads = torch.nonzero(proxy_checked & dense_now, as_tuple=False).flatten()
        if error_heads.numel():
            error = (proxy_mass[error_heads] - exact_mass[error_heads]).abs().double()
            self.proxy_abs_error_sum[offset, error_heads] += error
            self.proxy_error_count[offset, error_heads] += 1
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return output

def _real_hold_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs):
    manager = self._svd_checkpoint_hold_manager
    if manager.original(hidden_states, cache_position):
        return self._svd_checkpoint_hold_original_forward(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, cache_position=cache_position, **kwargs)
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

def install_real_hold_gate(model, manager):
    for layer in manager.layers:
        attention = model.model.language_model.layers[layer].self_attn
        if getattr(attention, '_svd_checkpoint_hold_wrapped', False):
            raise RuntimeError('Attention already wrapped')
        attention._svd_checkpoint_hold_manager = manager
        attention._svd_checkpoint_hold_original_forward = attention.forward
        attention.forward = MethodType(_real_hold_forward, attention)
        attention._svd_checkpoint_hold_wrapped = True
