# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""On-policy DENSE/HOLD gate with periodic exact checkpoint re-anchoring.

The experiment intentionally computes the native dense attention as an audit
side path.  Generation, however, uses only quantities available to the proposed
online algorithm on HOLD rows: exact text attention, rank-r visual mass, and a
visual readout saved from the anchor.  This separates algorithmic fidelity from
the later fused-kernel implementation.
"""
from __future__ import annotations
LEAN_STATS_BASE_SHA256 = 'ab5a2743db95a3063cc8e054fecff4f343a29e566c02976ccf3735eedb9300f1'
import math
from dataclasses import dataclass
from types import MethodType
from typing import Any
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
EPS = 1e-20

@dataclass
class VisualDecomposition:
    basis: torch.Tensor
    low_keys: torch.Tensor
    explained_energy: torch.Tensor
    effective_rank: int

@dataclass
class LayerState:
    holding: torch.Tensor
    anchor_ready: torch.Tensor
    anchor_readout: torch.Tensor
    anchor_visual_distribution: torch.Tensor
    anchor_token: torch.Tensor
    anchor_nonvisual_tokens: torch.Tensor
    root_anchor_ready: torch.Tensor
    root_anchor_readout: torch.Tensor
    root_anchor_nonvisual_tokens: torch.Tensor
    failed_checkpoint_streak: torch.Tensor
    history_reset_used: torch.Tensor
    episode_m3_base: torch.Tensor
    episode_confirmation_threshold: torch.Tensor

class SVDCheckpointHoldGate:
    """Per-layer/per-head proxy-confirmation state machine.

    A DENSE token whose exact visual mass is below ``m1`` schedules HOLD for
    the next token and leaves the anchor at the preceding DENSE token.  During
    HOLD, a rank-r proxy below ``m3_base*cos(pi/2*age/d)`` uses the anchored
    HOLD output.  A proxy above the threshold uses DENSE attention for the
    current token, but only exact mass >= ``confirmation_m1`` exits HOLD and
    refreshes the anchor.  By default ``confirmation_m1 == m1``.  The optional
    entry-recovery rule instead exits when exact visual mass recovers to the
    exact mass that began the current HOLD segment.  At distance
    ``d``, one DENSE checkpoint runs without paying for
    the proxy.  A low-mass checkpoint keeps the state in HOLD but moves the
    trust-region center to the exact current visual readout, so the following
    token starts a fresh distance window instead of remaining permanently
    DENSE.
    """
    protocol_version = 'svd_checkpoint_hold_gate_v3_reanchor'
    orbit_error_threshold = 0.8
    orbit_ratio_threshold = 0.65
    orbit_root_gap_threshold = 0.1
    orbit_window = 384
    orbit_scan_stride = 8
    orbit_min_period = 24
    orbit_max_period = 160

    def __init__(self, model: Any, layer_start: int=20, layer_end: int=28) -> None:
        self.model = model
        total = len(model.model.language_model.layers)
        if not 0 <= layer_start <= layer_end < total:
            raise ValueError('invalid targeted layer band')
        self.layers = tuple(range(layer_start, layer_end + 1))
        self.layer_offset = {layer: i for i, layer in enumerate(self.layers)}
        text = model.config.text_config
        self.num_heads = int(text.num_attention_heads)
        self.num_kv_heads = int(text.num_key_value_heads)
        if self.num_heads % self.num_kv_heads:
            raise ValueError('query heads must be divisible by KV heads')
        self.groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(text.hidden_size // self.num_heads)
        self.reset()

    def reset(self) -> None:
        self.protocol_version = type(self).protocol_version
        self.active = False
        self.prompt_length = 0
        self.prompt_nonvisual_tokens = 0
        self.last_nonvisual_tokens = 0
        self.text_key_rows = 0
        self.total_key_rows = 0
        self.max_new_tokens = 0
        self.m1 = 0.3
        self.confirmation_m1 = 0.3
        self.m3_base = 0.15
        self.max_anchor_distance = 64
        self.checkpoint_schedule = 'age'
        self.rank = 16
        self.length_adjusted_gate = False
        self.sample_mass_switch_gate = False
        self.sample_mass_continuous_gate = False
        self.entry_mass_m3_gate = False
        self.entry_mass_confirmation_gate = False
        self.entry_mass_m3_ratio = 0.5
        self.entry_mass_m3_min = 0.1
        self.entry_mass_m3_max = 0.15
        self.sample_mass_first_layer_reference: torch.Tensor | None = None
        self.sample_mass_layer_head_reference: torch.Tensor | None = None
        self.sample_mass_observed: float | None = None
        self.sample_mass_log_odds_offset: float | None = None
        self.sample_mass_continuous_log_odds_offset: torch.Tensor | None = None
        self.sample_mass_selected_branch = 'none'
        self.sample_mass_low_m1 = self.m1
        self.sample_mass_low_confirmation_m1 = self.confirmation_m1
        self.sample_mass_low_m3_base = self.m3_base
        self.sample_effective_m1 = self.m1
        self.sample_effective_confirmation_m1 = self.confirmation_m1
        self.sample_effective_m3_base = self.m3_base
        self.spectral_fallback_tau: float | None = None
        self.checkpoint_pulse_mode = 'none'
        self.orbit_all_layer_cycle_clean = False
        self.orbit_text_intervention = 'none'
        self.orbit_intervention_duration = 0
        self.orbit_suffix_cycles = 3
        self.orbit_soft_penalty = 2.0
        self.orbit_topk = 8
        self.orbit_phase_radius = 4
        self.orbit_keep_recent = 8
        self.orbit_visual_drop_fraction = 0.1
        self.orbit_prompt_boost = 1.0
        self.orbit_prompt_tail_tokens = 64
        self.orbit_prompt_tail_positions: torch.Tensor | None = None
        self.orbit_detected_period = -1
        self.orbit_clean_start = -1
        self.orbit_clean_end = -1
        self.orbit_mask_generated_start = -1
        self.orbit_mask_generated_end = -1
        self.orbit_triggered = False
        self.orbit_trigger_tokens: list[int] = []
        self.orbit_trigger_periods: list[int] = []
        self.orbit_reset_tokens: set[int] = set()
        self.orbit_gate_signature_trace: torch.Tensor | None = None
        self.orbit_confirmation_root_gap_trace: torch.Tensor | None = None
        self.orbit_driver_mask: torch.Tensor | None = None
        self.orbit_current_lag_profile: torch.Tensor | None = None
        self.orbit_current_top_lags: torch.Tensor | None = None
        self.orbit_trigger_lag_profile: torch.Tensor | None = None
        self.orbit_trigger_top_lags: torch.Tensor | None = None
        self.history_reset_after = 0
        self.history_reset_layer_start = 20
        self.history_reset_layer_end = 28
        self.decode_tokens = 0
        self.visual_positions: torch.Tensor | None = None
        self.decompositions: dict[int, VisualDecomposition] = {}
        self.states: dict[int, LayerState] = {}
        self.record_detailed_trace = False
        self.record_spatial_trace = False
        self.spatial_trace_stride = 8
        self.trace: dict[str, torch.Tensor] = {}
        self.spatial_trace: dict[str, torch.Tensor] = {}
        shape = (len(self.layers), self.num_heads)
        self.dense_rows = torch.zeros(shape, dtype=torch.int64)
        self.hold_rows = torch.zeros(shape, dtype=torch.int64)
        self.enter_rows = torch.zeros(shape, dtype=torch.int64)
        self.reopen_rows = torch.zeros(shape, dtype=torch.int64)
        self.proxy_check_rows = torch.zeros(shape, dtype=torch.int64)
        self.confirmation_rows = torch.zeros(shape, dtype=torch.int64)
        self.confirmation_low_rows = torch.zeros(shape, dtype=torch.int64)
        self.confirmation_high_rows = torch.zeros(shape, dtype=torch.int64)
        self.relaxed_confirmation_pass_rows = torch.zeros(shape, dtype=torch.int64)
        self.proxy_confirmation_rows = torch.zeros(shape, dtype=torch.int64)
        self.proxy_confirmation_low_rows = torch.zeros(shape, dtype=torch.int64)
        self.proxy_confirmation_high_rows = torch.zeros(shape, dtype=torch.int64)
        self.horizon_confirmation_rows = torch.zeros(shape, dtype=torch.int64)
        self.horizon_confirmation_low_rows = torch.zeros(shape, dtype=torch.int64)
        self.horizon_confirmation_high_rows = torch.zeros(shape, dtype=torch.int64)
        self.checkpoint_reanchor_rows = torch.zeros(shape, dtype=torch.int64)
        self.staggered_checkpoint_rows = torch.zeros(shape, dtype=torch.int64)
        self.staggered_age_fallback_rows = torch.zeros(shape, dtype=torch.int64)
        self.checkpoint_pulse_rows = torch.zeros(shape, dtype=torch.int64)
        self.checkpoint_pulse_mass_sum = torch.zeros(shape, dtype=torch.float64)
        self.checkpoint_pulse_boost_sum = torch.zeros(shape, dtype=torch.float64)
        self.orbit_all_layer_cycle_clean_rows = torch.zeros(shape, dtype=torch.int64)
        self.orbit_all_layer_cycle_clean_mass_sum = torch.zeros(shape, dtype=torch.float64)
        self.history_reset_rows = torch.zeros(shape, dtype=torch.int64)
        self.history_reset_mass_sum = torch.zeros(shape, dtype=torch.float64)
        self.max_failed_checkpoint_streak = torch.zeros(shape, dtype=torch.int64)
        self.history_reset_tokens: set[int] = set()
        self.holding_candidate_rows = torch.zeros(shape, dtype=torch.int64)
        self.distance_forced_rows = torch.zeros(shape, dtype=torch.int64)
        self.spectral_fallback_rows = torch.zeros(shape, dtype=torch.int64)
        self.spectral_fallback_risk_sum = torch.zeros(shape, dtype=torch.float64)
        self.proxy_abs_error_sum = torch.zeros(shape, dtype=torch.float64)
        self.proxy_error_count = torch.zeros(shape, dtype=torch.int64)
        self.entry_mass_m3_update_rows = torch.zeros(shape, dtype=torch.int64)
        self.entry_mass_m3_value_sum = torch.zeros(shape, dtype=torch.float64)
        self.entry_mass_m3_floor_rows = torch.zeros(shape, dtype=torch.int64)
        self.entry_mass_m3_ceiling_rows = torch.zeros(shape, dtype=torch.int64)
        self.entry_mass_confirmation_update_rows = torch.zeros(shape, dtype=torch.int64)
        self.entry_mass_confirmation_value_sum = torch.zeros(shape, dtype=torch.float64)
        self.entry_mass_confirmation_used_sum = torch.zeros(shape, dtype=torch.float64)
        self.first_token_exact_mass = torch.full(shape, float('nan'), dtype=torch.float64)
        self.energy_sum = torch.zeros(len(self.layers), dtype=torch.float64)
        self.energy_count = torch.zeros(len(self.layers), dtype=torch.int64)

    def begin_sample(self, input_ids: torch.Tensor, m1: float, m3_base: float, max_anchor_distance: int, rank: int, max_new_tokens: int, record_detailed_trace: bool=False, record_spatial_trace: bool=False, spatial_trace_stride: int=8, length_adjusted_gate: bool=False, spectral_fallback_tau: float | None=None, checkpoint_pulse_mode: str='none', orbit_all_layer_cycle_clean: bool=False, orbit_text_intervention: str='none', orbit_intervention_duration: int=16, orbit_suffix_cycles: int=3, orbit_soft_penalty: float=2.0, orbit_topk: int=8, orbit_phase_radius: int=4, orbit_keep_recent: int=8, orbit_visual_drop_fraction: float=0.1, orbit_prompt_boost: float=1.0, orbit_prompt_tail_tokens: int=64, history_reset_after: int=0, history_reset_layer_start: int=20, history_reset_layer_end: int=28, confirmation_m1: float | None=None, sample_mass_switch_gate: bool=False, sample_mass_first_layer_reference: list[float] | None=None, sample_mass_low_m1: float | None=None, sample_mass_low_confirmation_m1: float | None=None, sample_mass_low_m3_base: float | None=None, sample_mass_continuous_gate: bool=False, sample_mass_layer_head_reference: list[list[float]] | None=None, entry_mass_m3_gate: bool=False, entry_mass_m3_ratio: float=0.5, entry_mass_m3_min: float=0.1, entry_mass_m3_max: float=0.15, entry_mass_confirmation_gate: bool=False, checkpoint_schedule: str='age') -> None:
        if record_detailed_trace or record_spatial_trace or length_adjusted_gate or (spectral_fallback_tau is not None) or (checkpoint_pulse_mode != 'none') or orbit_all_layer_cycle_clean or (orbit_text_intervention != 'none') or (history_reset_after != 0) or sample_mass_switch_gate or sample_mass_continuous_gate or entry_mass_confirmation_gate or (checkpoint_schedule != 'age'):
            raise ValueError('Lightweight statistics only support plain age checkpoints with optional entry-mass m3; removed diagnostics cannot be enabled')
        if not 0.0 <= m1 <= 1.0:
            raise ValueError('m1 must lie in [0, 1]')
        if not 0.0 <= m3_base <= 1.0:
            raise ValueError('m3_base must lie in [0, 1]')
        if confirmation_m1 is not None and (not 0.0 <= confirmation_m1 <= 1.0):
            raise ValueError('confirmation_m1 must lie in [0, 1]')
        low_values = (sample_mass_low_m1, sample_mass_low_confirmation_m1, sample_mass_low_m3_base)
        if sample_mass_switch_gate and sample_mass_continuous_gate:
            raise ValueError('discrete and continuous sample-mass gates are mutually exclusive')
        if entry_mass_m3_gate and (sample_mass_switch_gate or sample_mass_continuous_gate or length_adjusted_gate):
            raise ValueError('entry-mass m3 and other adaptive gates are mutually exclusive')
        if entry_mass_confirmation_gate and (sample_mass_switch_gate or sample_mass_continuous_gate or length_adjusted_gate):
            raise ValueError('entry-mass confirmation and other adaptive gates are mutually exclusive')
        if entry_mass_m3_ratio <= 0.0:
            raise ValueError('entry_mass_m3_ratio must be positive')
        if not 0.0 <= entry_mass_m3_min <= entry_mass_m3_max <= 1.0:
            raise ValueError('invalid entry-mass m3 clipping range')
        if sample_mass_switch_gate:
            if sample_mass_first_layer_reference is None:
                raise ValueError('sample-mass switch requires a first-layer head reference')
            if any((value is None for value in low_values)):
                raise ValueError('sample-mass switch requires all low-branch thresholds')
            if any((not 0.0 <= float(value) <= 1.0 for value in low_values)):
                raise ValueError('low-branch thresholds must lie in [0, 1]')
            if length_adjusted_gate:
                raise ValueError('sample-mass switch and token-length calibration are mutually exclusive')
        elif sample_mass_first_layer_reference is not None or any((value is not None for value in low_values)):
            raise ValueError('sample-mass switch parameters require the switch gate')
        if sample_mass_continuous_gate:
            if sample_mass_layer_head_reference is None:
                raise ValueError('continuous sample-mass gate requires a layer/head reference')
            if length_adjusted_gate:
                raise ValueError('continuous sample-mass gate and token-length calibration are mutually exclusive')
            if not all((0.0 < value < 1.0 for value in (m1, m1 if confirmation_m1 is None else confirmation_m1, m3_base))):
                raise ValueError('continuous sample-mass base thresholds must lie in (0, 1)')
        elif sample_mass_layer_head_reference is not None:
            raise ValueError('sample-mass layer/head reference requires the continuous gate')
        first_layer_reference_tensor: torch.Tensor | None = None
        if sample_mass_first_layer_reference is not None:
            first_layer_reference_tensor = torch.as_tensor(sample_mass_first_layer_reference, dtype=torch.float64)
            expected_shape = (self.num_heads,)
            if tuple(first_layer_reference_tensor.shape) != expected_shape:
                raise ValueError(f'sample-mass first-layer reference must have shape {expected_shape}, got {tuple(first_layer_reference_tensor.shape)}')
            if not bool(((first_layer_reference_tensor > 0.0) & (first_layer_reference_tensor < 1.0)).all()):
                raise ValueError('sample-mass first-layer reference values must lie in (0, 1)')
        layer_head_reference_tensor: torch.Tensor | None = None
        if sample_mass_layer_head_reference is not None:
            layer_head_reference_tensor = torch.as_tensor(sample_mass_layer_head_reference, dtype=torch.float64)
            expected_shape = (len(self.layers), self.num_heads)
            if tuple(layer_head_reference_tensor.shape) != expected_shape:
                raise ValueError(f'sample-mass layer/head reference must have shape {expected_shape}, got {tuple(layer_head_reference_tensor.shape)}')
            if not bool(((layer_head_reference_tensor > 0.0) & (layer_head_reference_tensor < 1.0)).all()):
                raise ValueError('sample-mass layer/head reference values must lie in (0, 1)')
        if max_anchor_distance < 3:
            raise ValueError('max_anchor_distance must be at least 3')
        if checkpoint_schedule not in {'age', 'staggered'}:
            raise ValueError('checkpoint_schedule must be age/staggered')
        if rank < 1:
            raise ValueError('rank must be positive')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        if spatial_trace_stride < 1:
            raise ValueError('spatial_trace_stride must be positive')
        if spectral_fallback_tau is not None and spectral_fallback_tau <= 0:
            raise ValueError('spectral_fallback_tau must be positive')
        if checkpoint_pulse_mode not in {'none', 'current', 'root'}:
            raise ValueError('checkpoint_pulse_mode must be none/current/root')
        allowed_orbit_interventions = {'none', 'prompt_only', 'suffix_hard', 'suffix_soft', 'suffix_topk', 'phase_notch', 'rolling_band', 'visual_divert', 'suffix_visual_divert', 'prompt_boost', 'suffix_prompt_boost', 'diagnostic'}
        if orbit_text_intervention not in allowed_orbit_interventions:
            raise ValueError(f'orbit_text_intervention must be one of {sorted(allowed_orbit_interventions)}')
        if orbit_intervention_duration < 1:
            raise ValueError('orbit_intervention_duration must be positive')
        if orbit_suffix_cycles < 1:
            raise ValueError('orbit_suffix_cycles must be positive')
        if orbit_soft_penalty <= 0:
            raise ValueError('orbit_soft_penalty must be positive')
        if orbit_topk < 1:
            raise ValueError('orbit_topk must be positive')
        if orbit_phase_radius < 0:
            raise ValueError('orbit_phase_radius must be non-negative')
        if orbit_keep_recent < 1:
            raise ValueError('orbit_keep_recent must be positive')
        if not 0.0 < orbit_visual_drop_fraction < 1.0:
            raise ValueError('orbit_visual_drop_fraction must lie in (0, 1)')
        if orbit_prompt_boost <= 0:
            raise ValueError('orbit_prompt_boost must be positive')
        if orbit_prompt_tail_tokens < 1:
            raise ValueError('orbit_prompt_tail_tokens must be positive')
        if history_reset_after < 0:
            raise ValueError('history_reset_after must be non-negative')
        if history_reset_layer_start > history_reset_layer_end:
            raise ValueError('invalid history reset layer band')
        self.reset()
        self.active = True
        self.m1 = float(m1)
        self.confirmation_m1 = float(m1 if confirmation_m1 is None else confirmation_m1)
        self.m3_base = float(m3_base)
        self.max_anchor_distance = int(max_anchor_distance)
        self.checkpoint_schedule = checkpoint_schedule
        self.rank = int(rank)
        self.length_adjusted_gate = bool(length_adjusted_gate)
        self.sample_mass_switch_gate = bool(sample_mass_switch_gate)
        self.sample_mass_continuous_gate = bool(sample_mass_continuous_gate)
        self.entry_mass_m3_gate = bool(entry_mass_m3_gate)
        self.entry_mass_confirmation_gate = bool(entry_mass_confirmation_gate)
        self.entry_mass_m3_ratio = float(entry_mass_m3_ratio)
        self.entry_mass_m3_min = float(entry_mass_m3_min)
        self.entry_mass_m3_max = float(entry_mass_m3_max)
        self.sample_mass_first_layer_reference = first_layer_reference_tensor.to(input_ids.device) if first_layer_reference_tensor is not None else None
        self.sample_mass_layer_head_reference = layer_head_reference_tensor.to(input_ids.device) if layer_head_reference_tensor is not None else None
        self.sample_mass_observed = None
        self.sample_mass_log_odds_offset = None
        self.sample_mass_continuous_log_odds_offset = torch.full((len(self.layers), self.num_heads), float('nan'), dtype=torch.float64, device=input_ids.device) if self.sample_mass_continuous_gate else None
        self.sample_mass_selected_branch = 'none'
        self.sample_mass_low_m1 = float(self.m1 if sample_mass_low_m1 is None else sample_mass_low_m1)
        self.sample_mass_low_confirmation_m1 = float(self.confirmation_m1 if sample_mass_low_confirmation_m1 is None else sample_mass_low_confirmation_m1)
        self.sample_mass_low_m3_base = float(self.m3_base if sample_mass_low_m3_base is None else sample_mass_low_m3_base)
        self.sample_effective_m1 = self.m1
        self.sample_effective_confirmation_m1 = self.confirmation_m1
        self.sample_effective_m3_base = self.m3_base
        if self.sample_mass_continuous_gate:
            threshold_shape = (len(self.layers), self.num_heads)
            self.sample_effective_m1 = torch.full(threshold_shape, self.m1, dtype=torch.float64, device=input_ids.device)
            self.sample_effective_confirmation_m1 = torch.full(threshold_shape, self.confirmation_m1, dtype=torch.float64, device=input_ids.device)
            self.sample_effective_m3_base = torch.full(threshold_shape, self.m3_base, dtype=torch.float64, device=input_ids.device)
        self.spectral_fallback_tau = None if spectral_fallback_tau is None else float(spectral_fallback_tau)
        self.checkpoint_pulse_mode = checkpoint_pulse_mode
        if orbit_all_layer_cycle_clean and orbit_text_intervention == 'none':
            orbit_text_intervention = 'prompt_only'
        self.orbit_text_intervention = orbit_text_intervention
        self.orbit_all_layer_cycle_clean = orbit_text_intervention != 'none'
        self.orbit_intervention_duration = int(orbit_intervention_duration)
        self.orbit_suffix_cycles = int(orbit_suffix_cycles)
        self.orbit_soft_penalty = float(orbit_soft_penalty)
        self.orbit_topk = int(orbit_topk)
        self.orbit_phase_radius = int(orbit_phase_radius)
        self.orbit_keep_recent = int(orbit_keep_recent)
        self.orbit_visual_drop_fraction = float(orbit_visual_drop_fraction)
        self.orbit_prompt_boost = float(orbit_prompt_boost)
        self.orbit_prompt_tail_tokens = int(orbit_prompt_tail_tokens)
        self.history_reset_after = int(history_reset_after)
        self.history_reset_layer_start = int(history_reset_layer_start)
        self.history_reset_layer_end = int(history_reset_layer_end)
        if self.length_adjusted_gate or self.spectral_fallback_tau is not None:
            self.protocol_version = 'svd_checkpoint_hold_gate_v4_length_spectral'
        if self.checkpoint_pulse_mode != 'none':
            self.protocol_version = 'svd_checkpoint_hold_gate_v5_anchor_pulse'
        if self.history_reset_after:
            self.protocol_version = 'svd_checkpoint_hold_gate_v6_history_reset'
        if self.orbit_all_layer_cycle_clean:
            self.protocol_version = 'svd_checkpoint_hold_gate_v8_orbit_text_intervention'
        elif self.confirmation_m1 != self.m1:
            self.protocol_version = 'svd_checkpoint_hold_gate_v9_relaxed_exit'
        if self.sample_mass_switch_gate:
            self.protocol_version = 'svd_checkpoint_hold_gate_v10_sample_mass_switch'
        if self.sample_mass_continuous_gate:
            self.protocol_version = 'svd_checkpoint_hold_gate_v11_continuous_sample_mass'
        if self.entry_mass_m3_gate:
            self.protocol_version = 'svd_checkpoint_hold_gate_v12_entry_mass_m3'
        if self.entry_mass_confirmation_gate:
            self.protocol_version = 'svd_checkpoint_hold_gate_v14_entry_mass_confirmation'
        if self.checkpoint_schedule == 'staggered':
            self.protocol_version = 'svd_checkpoint_hold_gate_v13_staggered_checkpoint'
        self.max_new_tokens = int(max_new_tokens)
        self.record_detailed_trace = bool(record_detailed_trace)
        self.record_spatial_trace = bool(record_spatial_trace)
        self.spatial_trace_stride = int(spatial_trace_stride)
        self.prompt_length = int(input_ids.shape[1])
        ids = input_ids[0]
        cfg = self.model.config
        self.visual_positions = torch.nonzero((ids == int(cfg.image_token_id)) | (ids == int(cfg.video_token_id)), as_tuple=False).flatten()
        if self.visual_positions.numel() == 0:
            raise ValueError('sample contains no visual tokens')
        prompt_positions = torch.arange(self.prompt_length, device=input_ids.device)
        nonvisual = torch.ones(self.prompt_length, dtype=torch.bool, device=input_ids.device)
        nonvisual[self.visual_positions] = False
        self.orbit_prompt_tail_positions = prompt_positions[nonvisual][-self.orbit_prompt_tail_tokens:]
        self.prompt_nonvisual_tokens = max(self.prompt_length - int(self.visual_positions.numel()), 1)
        self.last_nonvisual_tokens = self.prompt_nonvisual_tokens
        for name in ('dense_rows', 'hold_rows', 'enter_rows', 'reopen_rows', 'proxy_check_rows', 'confirmation_rows', 'confirmation_low_rows', 'confirmation_high_rows', 'relaxed_confirmation_pass_rows', 'proxy_confirmation_rows', 'proxy_confirmation_low_rows', 'proxy_confirmation_high_rows', 'horizon_confirmation_rows', 'horizon_confirmation_low_rows', 'horizon_confirmation_high_rows', 'holding_candidate_rows', 'checkpoint_reanchor_rows', 'staggered_checkpoint_rows', 'staggered_age_fallback_rows', 'distance_forced_rows', 'checkpoint_pulse_rows', 'checkpoint_pulse_mass_sum', 'checkpoint_pulse_boost_sum', 'orbit_all_layer_cycle_clean_rows', 'orbit_all_layer_cycle_clean_mass_sum', 'history_reset_rows', 'history_reset_mass_sum', 'max_failed_checkpoint_streak', 'spectral_fallback_rows', 'spectral_fallback_risk_sum', 'proxy_abs_error_sum', 'proxy_error_count', 'entry_mass_m3_update_rows', 'entry_mass_m3_value_sum', 'entry_mass_m3_floor_rows', 'entry_mass_m3_ceiling_rows', 'first_token_exact_mass', 'entry_mass_confirmation_update_rows', 'entry_mass_confirmation_value_sum', 'entry_mass_confirmation_used_sum', 'energy_sum', 'energy_count'):
            setattr(self, name, getattr(self, name).to(input_ids.device))
        if self.orbit_all_layer_cycle_clean:
            orbit_shape = (self.max_new_tokens, len(self.layers), self.num_heads)
            self.orbit_gate_signature_trace = torch.full(orbit_shape, float('nan'), device=input_ids.device)
            self.orbit_confirmation_root_gap_trace = torch.full(orbit_shape, float('nan'), device=input_ids.device)
            self.orbit_driver_mask = torch.zeros((len(self.layers), self.num_heads), dtype=torch.bool, device=input_ids.device)
            self.orbit_current_lag_profile = torch.zeros((len(self.layers), self.num_heads, 8), dtype=torch.float32, device=input_ids.device)
            self.orbit_current_top_lags = torch.zeros((len(self.layers), self.num_heads, 8), dtype=torch.int16, device=input_ids.device)
        if self.record_detailed_trace:
            shape = (self.max_new_tokens, len(self.layers), self.num_heads)
            device = input_ids.device
            self.trace = {'exact_mass': torch.full(shape, float('nan'), device=device), 'proxy_mass': torch.full(shape, float('nan'), device=device), 'decayed_threshold': torch.full(shape, float('nan'), device=device), 'decision_dense': torch.full(shape, 255, dtype=torch.uint8, device=device), 'holding_after': torch.zeros(shape, dtype=torch.uint8, device=device), 'transition_enter': torch.zeros(shape, dtype=torch.uint8, device=device), 'transition_reopen': torch.zeros(shape, dtype=torch.uint8, device=device), 'proxy_checked': torch.zeros(shape, dtype=torch.uint8, device=device), 'confirmation_dense': torch.zeros(shape, dtype=torch.uint8, device=device), 'confirmation_passed': torch.zeros(shape, dtype=torch.uint8, device=device), 'confirmation_failed': torch.zeros(shape, dtype=torch.uint8, device=device), 'horizon_confirmation': torch.zeros(shape, dtype=torch.uint8, device=device), 'checkpoint_reanchor': torch.zeros(shape, dtype=torch.uint8, device=device), 'checkpoint_pulse': torch.zeros(shape, dtype=torch.uint8, device=device), 'checkpoint_pulse_mass': torch.full(shape, float('nan'), device=device), 'orbit_all_layer_cycle_clean': torch.zeros(shape, dtype=torch.uint8, device=device), 'orbit_all_layer_cycle_clean_mass': torch.full(shape, float('nan'), device=device), 'orbit_cycle_trigger': torch.zeros(self.max_new_tokens, dtype=torch.uint8, device=device), 'orbit_cycle_error': torch.full((self.max_new_tokens,), float('nan'), device=device), 'orbit_cycle_ratio': torch.full((self.max_new_tokens,), float('nan'), device=device), 'orbit_cycle_period': torch.full((self.max_new_tokens,), -1, dtype=torch.int16, device=device), 'orbit_root_gap_p50': torch.full((self.max_new_tokens,), float('nan'), device=device), 'orbit_driver_head': torch.zeros(shape, dtype=torch.uint8, device=device), 'history_reset': torch.zeros(shape, dtype=torch.uint8, device=device), 'spectral_fallback': torch.zeros(shape, dtype=torch.uint8, device=device), 'spectral_risk': torch.full(shape, float('nan'), device=device), 'm1_threshold': torch.full(shape, float('nan'), device=device), 'confirmation_threshold': torch.full(shape, float('nan'), device=device), 'anchor_visual_delta_norm': torch.full(shape, float('nan'), device=device), 'anchor_visual_relative_delta': torch.full(shape, float('nan'), device=device), 'anchor_visual_cosine': torch.full(shape, float('nan'), device=device), 'mass_weighted_anchor_delta': torch.full(shape, float('nan'), device=device), 'root_mass_weighted_delta': torch.full(shape, float('nan'), device=device), 'visual_text_cosine': torch.full(shape, float('nan'), device=device), 'anchor_token': torch.full(shape, -1, dtype=torch.int16, device=device), 'anchor_distance': torch.full(shape, -1, dtype=torch.int16, device=device)}
        if self.record_spatial_trace:
            scans = self.max_new_tokens // self.spatial_trace_stride
            visual = int(self.visual_positions.numel())
            spatial_shape = (scans, len(self.layers), self.num_heads, visual)
            row_shape = (scans, len(self.layers), self.num_heads)
            device = input_ids.device
            self.spatial_trace = {'spatial_dense_distribution': torch.zeros(spatial_shape, dtype=torch.bfloat16, device=device), 'spatial_used_distribution': torch.zeros(spatial_shape, dtype=torch.bfloat16, device=device), 'spatial_exact_mass': torch.zeros(row_shape, dtype=torch.bfloat16, device=device), 'spatial_stay_hold': torch.zeros(row_shape, dtype=torch.uint8, device=device), 'spatial_anchor_token': torch.full(row_shape, -1, dtype=torch.int16, device=device)}

    def original(self, hidden_states: torch.Tensor, cache_position: torch.Tensor | None) -> bool:
        if not self.active or hidden_states.shape[-2] != 1:
            return True
        if cache_position is None or cache_position.numel() != 1:
            return True
        return int(cache_position.item()) < self.prompt_length

    def all_layer_clean_due(self, cache_position: torch.Tensor | None) -> bool:
        if not self.active or not self.orbit_all_layer_cycle_clean or self.orbit_text_intervention == 'diagnostic' or (cache_position is None) or (cache_position.numel() != 1):
            return False
        token = int(cache_position.item()) - self.prompt_length + 1
        return self.orbit_clean_start <= token <= self.orbit_clean_end

    def apply_orbit_text_intervention(self, logits: torch.Tensor, current_position: int) -> torch.Tensor:
        """Modify logits only on the detector-selected generated-text span.

        The blocked span ends at the trigger token.  Tokens generated after
        the intervention starts remain visible, allowing a new coherent branch
        to grow instead of repeatedly decoding from prompt-only context.
        """
        mode = self.orbit_text_intervention
        if mode == 'none':
            return logits
        if mode == 'diagnostic':
            return logits
        changed = logits.clone()
        if mode == 'prompt_only':
            if current_position > self.prompt_length:
                changed[..., self.prompt_length:current_position] = -float('inf')
            return changed
        if mode == 'phase_notch':
            for multiple in range(1, self.orbit_suffix_cycles + 1):
                center = current_position - multiple * self.orbit_detected_period
                start = max(self.prompt_length, center - self.orbit_phase_radius)
                end = min(current_position, center + self.orbit_phase_radius + 1)
                if end > start:
                    changed[..., start:end] = -float('inf')
            return changed
        if mode == 'rolling_band':
            start = max(self.prompt_length, current_position - self.orbit_suffix_cycles * self.orbit_detected_period)
            end = max(start, current_position - self.orbit_keep_recent)
            if end > start:
                changed[..., start:end] = -float('inf')
            return changed
        visual_divert = mode in {'visual_divert', 'suffix_visual_divert'}
        if mode == 'visual_divert':
            return self._drop_top_visual_logits(changed)
        if mode == 'prompt_boost':
            return self._boost_prompt_tail_logits(changed)
        start = self.prompt_length + self.orbit_mask_generated_start - 1
        end = self.prompt_length + self.orbit_mask_generated_end
        start = max(self.prompt_length, min(start, current_position))
        end = max(start, min(end, current_position))
        if end <= start:
            return changed
        if mode in {'suffix_hard', 'suffix_visual_divert', 'suffix_prompt_boost'}:
            changed[..., start:end] = -float('inf')
        elif mode == 'suffix_soft':
            changed[..., start:end] -= self.orbit_soft_penalty
        elif mode == 'suffix_topk':
            span = changed[..., start:end]
            k = min(self.orbit_topk, span.shape[-1])
            indices = torch.topk(span, k=k, dim=-1).indices
            span.scatter_(-1, indices, -float('inf'))
        else:
            raise RuntimeError(f'unknown orbit intervention: {mode}')
        if visual_divert:
            changed = self._drop_top_visual_logits(changed)
        if mode == 'suffix_prompt_boost':
            changed = self._boost_prompt_tail_logits(changed)
        return changed

    def _drop_top_visual_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Force each head to use visual locations outside its current top set."""
        assert self.visual_positions is not None
        positions = self.visual_positions.to(logits.device)
        visual = logits.index_select(-1, positions)
        k = max(1, int(math.ceil(visual.shape[-1] * self.orbit_visual_drop_fraction)))
        relative = torch.topk(visual, k=k, dim=-1).indices
        absolute = positions[relative]
        logits.scatter_(-1, absolute, -float('inf'))
        return logits

    def _boost_prompt_tail_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Raise attention odds for the tail of the original user prompt."""
        if self.orbit_prompt_tail_positions is None:
            raise RuntimeError('prompt-tail positions were not initialized')
        positions = self.orbit_prompt_tail_positions.to(logits.device)
        logits[..., positions] += self.orbit_prompt_boost
        return logits

    def _decompose(self, layer: int, kv_keys: torch.Tensor) -> VisualDecomposition:
        assert self.visual_positions is not None
        positions = self.visual_positions.to(kv_keys.device)
        visual = kv_keys[0, :, positions, :].float()
        gram = visual.transpose(-1, -2) @ visual
        eigenvalues, basis = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
        basis = basis.flip(-1)
        effective_rank = min(self.rank, visual.shape[-2], visual.shape[-1])
        basis = basis[..., :effective_rank].contiguous()
        low_keys = (visual @ basis).contiguous()
        total = eigenvalues.sum(-1).clamp_min(EPS)
        energy = eigenvalues[..., :effective_rank].sum(-1) / total
        result = VisualDecomposition(basis=basis, low_keys=low_keys, explained_energy=energy, effective_rank=effective_rank)
        self.decompositions[layer] = result
        offset = self.layer_offset[layer]
        self.energy_sum[offset] += float(energy.mean().item())
        self.energy_count[offset] += 1
        return result

    def _state(self, layer: int, dim: int, device: torch.device) -> LayerState:
        value = self.states.get(layer)
        if value is None:
            value = LayerState(holding=torch.zeros(self.num_heads, dtype=torch.bool, device=device), anchor_ready=torch.zeros(self.num_heads, dtype=torch.bool, device=device), anchor_readout=torch.zeros((self.num_heads, dim), dtype=torch.float32, device=device), anchor_visual_distribution=torch.zeros((self.num_heads, int(self.visual_positions.numel())), dtype=torch.float32, device=device), anchor_token=torch.full((self.num_heads,), -1, dtype=torch.int16, device=device), anchor_nonvisual_tokens=torch.zeros(self.num_heads, dtype=torch.int32, device=device), root_anchor_ready=torch.zeros(self.num_heads, dtype=torch.bool, device=device), root_anchor_readout=torch.zeros((self.num_heads, dim), dtype=torch.float32, device=device), root_anchor_nonvisual_tokens=torch.zeros(self.num_heads, dtype=torch.int32, device=device), failed_checkpoint_streak=torch.zeros(self.num_heads, dtype=torch.int16, device=device), history_reset_used=torch.zeros(self.num_heads, dtype=torch.bool, device=device), episode_m3_base=torch.full((self.num_heads,), self.m3_base, dtype=torch.float32, device=device), episode_confirmation_threshold=torch.full((self.num_heads,), self.confirmation_m1, dtype=torch.float32, device=device))
            self.states[layer] = value
        return value

    @staticmethod
    def _length_adjusted_threshold(value: float, initial_text: int, current_text: int) -> float:
        initial = max(int(initial_text), 1)
        current = max(int(current_text), 1)
        numerator = float(value) * initial
        return numerator / (numerator + (1.0 - float(value)) * current)

    def _initialize_sample_mass_switch(self) -> None:
        if self.sample_mass_log_odds_offset is not None:
            return
        first_layer = self.first_token_exact_mass[0]
        finite = first_layer[torch.isfinite(first_layer)]
        if finite.numel() != self.num_heads:
            raise RuntimeError('first target layer visual mass is incomplete before switching')
        observed = float(torch.quantile(finite, 0.5).item())
        self.sample_mass_observed = observed
        assert self.sample_mass_first_layer_reference is not None
        epsilon = 1e-06
        observed_rows = first_layer.clamp(epsilon, 1.0 - epsilon)
        reference_rows = self.sample_mass_first_layer_reference.clamp(epsilon, 1.0 - epsilon)
        offset_rows = torch.logit(observed_rows) - torch.logit(reference_rows)
        offset = float(torch.quantile(offset_rows.flatten(), 0.5).item())
        self.sample_mass_log_odds_offset = offset
        if offset >= 0.0:
            self.sample_mass_selected_branch = 'high'
            self.sample_effective_m1 = self.m1
            self.sample_effective_confirmation_m1 = self.confirmation_m1
            self.sample_effective_m3_base = self.m3_base
        else:
            self.sample_mass_selected_branch = 'low'
            self.sample_effective_m1 = self.sample_mass_low_m1
            self.sample_effective_confirmation_m1 = self.sample_mass_low_confirmation_m1
            self.sample_effective_m3_base = self.sample_mass_low_m3_base

    def _initialize_continuous_sample_mass(self, layer: int) -> None:
        """Freeze per-head thresholds from token-one visual attention odds."""
        offset = self.layer_offset[layer]
        assert self.sample_mass_continuous_log_odds_offset is not None
        if bool(torch.isfinite(self.sample_mass_continuous_log_odds_offset[offset]).all()):
            return
        observed = self.first_token_exact_mass[offset]
        if not bool(torch.isfinite(observed).all()):
            raise RuntimeError('first-token visual mass is incomplete before continuous sample calibration')
        assert self.sample_mass_layer_head_reference is not None
        epsilon = 1e-06
        observed = observed.clamp(epsilon, 1.0 - epsilon)
        reference = self.sample_mass_layer_head_reference[offset].clamp(epsilon, 1.0 - epsilon)
        log_odds_offset = torch.logit(observed) - torch.logit(reference)
        self.sample_mass_continuous_log_odds_offset[offset] = log_odds_offset

        def shifted_threshold(base: float) -> torch.Tensor:
            base_tensor = torch.full_like(log_odds_offset, float(base))
            return torch.sigmoid(torch.logit(base_tensor) + log_odds_offset)
        assert isinstance(self.sample_effective_m1, torch.Tensor)
        assert isinstance(self.sample_effective_confirmation_m1, torch.Tensor)
        assert isinstance(self.sample_effective_m3_base, torch.Tensor)
        self.sample_effective_m1[offset] = shifted_threshold(self.m1)
        self.sample_effective_confirmation_m1[offset] = shifted_threshold(self.confirmation_m1)
        self.sample_effective_m3_base[offset] = shifted_threshold(self.m3_base)

    @staticmethod
    def _restored_visual_mass(exact_mass: torch.Tensor, current_nonvisual_tokens: int, anchor_nonvisual_tokens: torch.Tensor, cap: float) -> torch.Tensor:
        """Restore visual odds for text-KV growth, without reducing mass."""
        reference = anchor_nonvisual_tokens.clamp_min(1).to(exact_mass.dtype)
        scale = float(max(current_nonvisual_tokens, 1)) / reference
        odds = exact_mass / (1.0 - exact_mass).clamp_min(EPS)
        restored = odds * scale / (1.0 + odds * scale)
        return torch.maximum(exact_mass, restored).clamp_max(float(cap))

    @staticmethod
    def _apply_spectral_fallback(decision: dict[str, torch.Tensor], spectral_fallback: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exit HOLD and refresh the anchor on risk-triggered head rows."""
        if not bool(spectral_fallback.any()):
            return decision
        updated = dict(decision)
        updated['stay_hold'] = decision['stay_hold'] & ~spectral_fallback
        updated['dense_now'] = decision['dense_now'] | spectral_fallback
        updated['holding_after'] = decision['holding_after'] & ~spectral_fallback
        updated['update_anchor'] = decision['update_anchor'] | spectral_fallback
        return updated

    @staticmethod
    def _history_reset_transition(streak: torch.Tensor, used: torch.Tensor, horizon_confirmation_low: torch.Tensor, holding_after: torch.Tensor, enter: torch.Tensor, reset_after: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the per-head failed-checkpoint streak and fire once/episode."""
        advanced = torch.where(horizon_confirmation_low, streak + torch.ones_like(streak), streak)
        trigger = horizon_confirmation_low & ~used & (advanced >= reset_after)
        new_episode_or_exit = enter | ~holding_after
        next_streak = torch.where(new_episode_or_exit, torch.zeros_like(advanced), advanced)
        next_used = torch.where(new_episode_or_exit, torch.zeros_like(used), used | trigger)
        return (trigger, next_streak, next_used)

    @classmethod
    def _orbit_cycle_metrics(cls, signature: torch.Tensor, end: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor] | None:
        """Return the frozen three-block recurrence score at one token."""
        errors: list[torch.Tensor] = []
        head_errors: list[torch.Tensor] = []
        periods: list[int] = []
        for period in range(cls.orbit_min_period, cls.orbit_max_period + 1):
            block = min(64, period)
            if end < 2 * period + block:
                continue
            current = signature[end - block:end]
            previous = signature[end - period - block:end - period]
            older = signature[end - 2 * period - block:end - 2 * period]
            first = (current - previous).abs()
            second = (previous - older).abs()
            error = torch.maximum(first.mean(), second.mean())
            errors.append(error)
            head_errors.append(torch.maximum(first.mean(0), second.mean(0)))
            periods.append(period)
        if not errors:
            return None
        stacked = torch.stack(errors)
        best_index = int(stacked.argmin().item())
        best_error = stacked[best_index]
        ratio = best_error / torch.quantile(stacked, 0.5).clamp_min(1e-12)
        head_stack = torch.stack(head_errors)
        selected_head_error = head_stack[best_index]
        selected_head_ratio = selected_head_error / torch.quantile(head_stack, 0.5, dim=0).clamp_min(1e-12)
        return (best_error, ratio, periods[best_index], selected_head_error, selected_head_ratio)

    def _finish_orbit_token(self, token: int) -> None:
        """Evaluate the frozen online rule after all targeted layers finish."""
        if not self.orbit_all_layer_cycle_clean:
            return
        if self.orbit_gate_signature_trace is None:
            raise RuntimeError('orbit signature storage was not initialized')
        if self.orbit_confirmation_root_gap_trace is None:
            raise RuntimeError('orbit root-gap storage was not initialized')
        if token < 192 or (token - 192) % self.orbit_scan_stride:
            return
        metrics = self._orbit_cycle_metrics(self.orbit_gate_signature_trace[:token], token)
        if metrics is None:
            return
        error, ratio, period, head_error, head_ratio = metrics
        start = max(0, token - self.orbit_window)
        gaps = self.orbit_confirmation_root_gap_trace[start:token]
        finite_gaps = gaps[torch.isfinite(gaps)]
        root_gap = torch.quantile(finite_gaps, 0.5) if finite_gaps.numel() else torch.tensor(float('nan'), device=error.device)
        head_root_gap = torch.nanquantile(gaps, 0.5, dim=0)
        driver_mask = (head_error <= self.orbit_error_threshold) & (head_ratio <= self.orbit_ratio_threshold) & (head_root_gap >= self.orbit_root_gap_threshold)
        fired = bool(not self.orbit_triggered and error <= self.orbit_error_threshold and (ratio <= self.orbit_ratio_threshold) and torch.isfinite(root_gap) and (root_gap >= self.orbit_root_gap_threshold))
        if fired:
            self.orbit_clean_start = token + 1
            duration = period if self.orbit_text_intervention == 'prompt_only' else self.orbit_intervention_duration
            self.orbit_clean_end = token + duration
            self.orbit_mask_generated_start = max(1, token - self.orbit_suffix_cycles * period + 1)
            self.orbit_mask_generated_end = token
            self.orbit_detected_period = period
            self.orbit_triggered = True
            if self.orbit_driver_mask is None:
                raise RuntimeError('orbit driver storage was not initialized')
            self.orbit_driver_mask.copy_(driver_mask)
            if self.orbit_current_lag_profile is None:
                raise RuntimeError('orbit lag-profile storage was not initialized')
            if self.orbit_current_top_lags is None:
                raise RuntimeError('orbit top-lag storage was not initialized')
            self.orbit_trigger_lag_profile = self.orbit_current_lag_profile.detach().clone()
            self.orbit_trigger_top_lags = self.orbit_current_top_lags.detach().clone()
            self.orbit_trigger_tokens.append(token)
            self.orbit_trigger_periods.append(period)
        if self.record_detailed_trace:
            row = token - 1
            self.trace['orbit_cycle_trigger'][row] = int(fired)
            self.trace['orbit_cycle_error'][row] = error.detach()
            self.trace['orbit_cycle_ratio'][row] = ratio.detach()
            self.trace['orbit_cycle_period'][row] = period
            self.trace['orbit_root_gap_p50'][row] = root_gap.detach()
            if fired:
                self.trace['orbit_driver_head'][row] = driver_mask.to(torch.uint8)

    @staticmethod
    def _proxy_mask(holding_before: torch.Tensor, anchor_ready: torch.Tensor, distance: torch.Tensor, max_anchor_distance: int, checkpoint_due: torch.Tensor | None=None) -> torch.Tensor:
        if checkpoint_due is not None:
            return holding_before & anchor_ready & ~checkpoint_due
        return holding_before & anchor_ready & (distance < max_anchor_distance)

    @staticmethod
    def _checkpoint_due(holding_before: torch.Tensor, anchor_ready: torch.Tensor, distance: torch.Tensor, token: int, layer_offset: int, num_heads: int, max_anchor_distance: int, schedule: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return checkpoint, staggered-slot, and age-fallback masks."""
        eligible = holding_before & anchor_ready
        age_due = eligible & (distance >= max_anchor_distance)
        if schedule == 'age':
            zeros = torch.zeros_like(age_due)
            return (age_due, zeros, zeros)
        heads = torch.arange(num_heads, device=distance.device)
        phases = (layer_offset * num_heads + heads) % max_anchor_distance
        slot_due = eligible & (phases == token % max_anchor_distance)
        fallback_due = age_due & ~slot_due
        return (slot_due | fallback_due, slot_due, fallback_due)

    @staticmethod
    def _update_episode_m3_base(previous: torch.Tensor, exact_mass: torch.Tensor, update: torch.Tensor, ratio: float, minimum: float, maximum: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calibrate m3 once per HOLD segment from an already-exact row."""
        raw = exact_mass * ratio
        calibrated = raw.clamp(min=minimum, max=maximum)
        value = torch.where(update, calibrated, previous)
        floor = update & (raw <= minimum)
        ceiling = update & (raw >= maximum)
        return (value, calibrated, floor, ceiling)

    @staticmethod
    def _update_episode_confirmation_threshold(previous: torch.Tensor, exact_mass: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Use the exact mass at a new HOLD trust-region center."""
        return torch.where(update, exact_mass, previous)

    @staticmethod
    def _transition(holding_before: torch.Tensor, proxy_checked: torch.Tensor, proxy_mass: torch.Tensor, threshold: torch.Tensor, exact_mass: torch.Tensor, m1: float, state_anchor_ready: torch.Tensor, distance: torch.Tensor, max_anchor_distance: int, confirmation_m1: float | torch.Tensor | None=None, checkpoint_due: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        """Return one-token decisions without mutating the layer state."""
        exit_threshold = m1 if confirmation_m1 is None else confirmation_m1
        stay_hold = proxy_checked & (proxy_mass < threshold)
        confirmation = holding_before & ~stay_hold
        confirmation_passed = confirmation & (exact_mass >= exit_threshold)
        confirmation_failed = confirmation & ~confirmation_passed
        horizon_confirmation = confirmation & (distance >= max_anchor_distance if checkpoint_due is None else checkpoint_due)
        checkpoint_reanchor = horizon_confirmation & confirmation_failed
        enter = ~holding_before & state_anchor_ready & (exact_mass < m1)
        reopen = confirmation_passed
        return {'stay_hold': stay_hold, 'confirmation': confirmation, 'confirmation_passed': confirmation_passed, 'confirmation_failed': confirmation_failed, 'horizon_confirmation': horizon_confirmation, 'checkpoint_reanchor': checkpoint_reanchor, 'enter': enter, 'reopen': reopen, 'dense_now': ~holding_before | confirmation, 'holding_after': stay_hold | confirmation_failed | enter, 'update_anchor': reopen | checkpoint_reanchor | ~holding_before & ~enter}

    def apply(self, layer: int, query: torch.Tensor, kv_keys: torch.Tensor, logits: torch.Tensor, probabilities: torch.Tensor, values: torch.Tensor, dense_output: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        assert self.visual_positions is not None
        token = self.decode_tokens + 1
        offset = self.layer_offset[layer]
        if token > self.max_new_tokens:
            raise RuntimeError('generation escaped allocated trace')
        positions = self.visual_positions.to(query.device)
        full_rows = logits[0, :, 0].float()
        n_total_keys = int(full_rows.shape[-1])
        n_text_keys = n_total_keys - int(positions.numel())
        self.last_nonvisual_tokens = n_text_keys
        self.text_key_rows += self.num_heads * n_text_keys
        self.total_key_rows += self.num_heads * n_total_keys
        text_logits = full_rows.clone()
        text_logits[:, positions] = -float('inf')
        log_z_text = torch.logsumexp(text_logits, dim=-1)
        visual_weights = probabilities[0, :, 0, positions].float()
        exact_mass = visual_weights.sum(-1)
        if token == 1:
            self.first_token_exact_mass[offset] = exact_mass.detach().double()
        visual_values = values[0, :, positions, :].float()
        exact_visual = torch.einsum('hv,hvd->hd', visual_weights, visual_values)
        exact_visual_readout = exact_visual / exact_mass.clamp_min(EPS)[:, None]
        exact_visual_readout = torch.where(exact_mass[:, None] > 0, exact_visual_readout, torch.zeros_like(exact_visual_readout))
        text_probabilities = F.softmax(text_logits, dim=-1, dtype=torch.float32)
        text_readout = torch.einsum('hs,hsd->hd', text_probabilities, values[0].float())
        state = self._state(layer, values.shape[-1], query.device)
        holding_before = state.holding.clone()
        anchor_before = state.anchor_token.clone()
        distance = torch.where(anchor_before >= 0, token - anchor_before, -1)
        decay_distance = (distance - 2).clamp_min(0).float()
        decay_horizon = max(self.max_anchor_distance - 2, 1)
        phase = (decay_distance / decay_horizon).clamp(0.0, 1.0)
        current_m1 = self.m1
        current_confirmation_m1 = self.confirmation_m1
        current_m3_base = self.m3_base
        if self.entry_mass_m3_gate:
            current_m3_base = state.episode_m3_base.to(exact_mass.dtype)
        confirmation_threshold = torch.as_tensor(current_confirmation_m1, dtype=exact_mass.dtype, device=exact_mass.device).expand_as(exact_mass)
        threshold = current_m3_base * torch.cos(0.5 * math.pi * phase)
        threshold = torch.where(distance >= 0, threshold, torch.zeros_like(threshold))
        checkpoint_due, staggered_slot_due, staggered_fallback_due = self._checkpoint_due(holding_before, state.anchor_ready, distance, token, offset, self.num_heads, self.max_anchor_distance, self.checkpoint_schedule)
        proxy_checked = self._proxy_mask(holding_before, state.anchor_ready, distance, self.max_anchor_distance, checkpoint_due=checkpoint_due)
        proxy_mass = torch.full_like(exact_mass, float('nan'))
        proxy_heads = torch.nonzero(proxy_checked, as_tuple=False).flatten()
        if proxy_heads.numel():
            decomposition = self.decompositions.get(layer)
            if decomposition is None:
                decomposition = self._decompose(layer, kv_keys)
            query_rows = query[0, proxy_heads, 0, :].float()
            kv_index = proxy_heads // self.groups
            basis = decomposition.basis[kv_index]
            low_keys = decomposition.low_keys[kv_index]
            low_query = torch.einsum('hd,hdr->hr', query_rows, basis)
            approximate_visual_logits = torch.einsum('hr,hvr->hv', low_query, low_keys) * float(self.model.model.language_model.layers[layer].self_attn.scaling)
            if attention_mask is not None:
                approximate_visual_logits += attention_mask[0, 0, 0, positions].float()
            log_z_proxy_visual = torch.logsumexp(approximate_visual_logits, dim=-1)
            proxy_mass[proxy_heads] = torch.sigmoid(log_z_proxy_visual - log_z_text[proxy_heads])
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
        update_anchor = decision['update_anchor']
        episode_m3_update = torch.zeros_like(enter)
        episode_m3_value = torch.full_like(exact_mass, float('nan'))
        episode_m3_floor = torch.zeros_like(episode_m3_update)
        episode_m3_ceiling = torch.zeros_like(episode_m3_update)
        if self.entry_mass_m3_gate:
            episode_m3_update = enter | checkpoint_reanchor
            state.episode_m3_base, episode_m3_value, episode_m3_floor, episode_m3_ceiling = self._update_episode_m3_base(state.episode_m3_base, exact_mass.detach(), episode_m3_update, self.entry_mass_m3_ratio, self.entry_mass_m3_min, self.entry_mass_m3_max)
        state.anchor_readout = torch.where(update_anchor[:, None], exact_visual_readout.detach(), state.anchor_readout)
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
        if proxy_heads.numel():
            error = (proxy_mass[proxy_heads] - exact_mass[proxy_heads]).abs().double()
            self.proxy_abs_error_sum[offset, proxy_heads] += error
            self.proxy_error_count[offset, proxy_heads] += 1
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return output

    def get_trace_cpu(self) -> dict[str, torch.Tensor]:
        if not self.record_detailed_trace:
            raise RuntimeError('detailed trace recording is disabled')
        return {name: value[:self.decode_tokens].detach().cpu().contiguous() for name, value in self.trace.items()}

    def get_spatial_trace_cpu(self) -> dict[str, torch.Tensor]:
        if not self.record_spatial_trace:
            raise RuntimeError('spatial trace recording is disabled')
        scans = self.decode_tokens // self.spatial_trace_stride
        payload = {name: value[:scans].detach().cpu().contiguous() for name, value in self.spatial_trace.items()}
        payload['spatial_scan_tokens'] = torch.arange(self.spatial_trace_stride, scans * self.spatial_trace_stride + 1, self.spatial_trace_stride, dtype=torch.int32)
        payload['visual_token_positions'] = self.visual_positions.detach().cpu().to(torch.int32).contiguous()
        return payload

    def statistics(self) -> dict[str, Any]:
        dense = int(self.dense_rows.sum().item())
        hold = int(self.hold_rows.sum().item())
        eligible = dense + hold
        errors = int(self.proxy_error_count.sum().item())
        proxy_checks = int(self.proxy_check_rows.sum().item())
        confirmations = int(self.confirmation_rows.sum().item())
        confirmation_low = int(self.confirmation_low_rows.sum().item())
        confirmation_high = int(self.confirmation_high_rows.sum().item())
        relaxed_confirmation_pass = int(self.relaxed_confirmation_pass_rows.sum().item())
        proxy_confirmations = int(self.proxy_confirmation_rows.sum().item())
        proxy_confirmation_low = int(self.proxy_confirmation_low_rows.sum().item())
        proxy_confirmation_high = int(self.proxy_confirmation_high_rows.sum().item())
        horizon_confirmations = int(self.horizon_confirmation_rows.sum().item())
        horizon_confirmation_low = int(self.horizon_confirmation_low_rows.sum().item())
        horizon_confirmation_high = int(self.horizon_confirmation_high_rows.sum().item())
        checkpoint_reanchors = int(self.checkpoint_reanchor_rows.sum().item())
        staggered_checkpoints = int(self.staggered_checkpoint_rows.sum().item())
        staggered_age_fallbacks = int(self.staggered_age_fallback_rows.sum().item())
        entry_m3_updates = int(self.entry_mass_m3_update_rows.sum().item())
        entry_m3_floors = int(self.entry_mass_m3_floor_rows.sum().item())
        entry_m3_ceilings = int(self.entry_mass_m3_ceiling_rows.sum().item())
        entry_confirmation_updates = int(self.entry_mass_confirmation_update_rows.sum().item())
        checkpoint_pulses = int(self.checkpoint_pulse_rows.sum().item())
        orbit_all_layer_cycle_cleans = int(self.orbit_all_layer_cycle_clean_rows.sum().item())
        history_resets = int(self.history_reset_rows.sum().item())
        spectral_fallbacks = int(self.spectral_fallback_rows.sum().item())
        holding_candidates = int(self.holding_candidate_rows.sum().item())
        energy_count = int(self.energy_count.sum().item())
        n_visual = int(self.visual_positions.numel()) if self.visual_positions is not None else 0
        effective_rank = min(self.rank, self.head_dim, n_visual)
        recurring_vs_full_visual_qk = effective_rank / self.head_dim + effective_rank / n_visual if n_visual else float('nan')
        text_qk_macs = self.text_key_rows * self.head_dim
        baseline_attention_qk_macs = self.total_key_rows * self.head_dim
        proxy_qk_macs = proxy_checks * (self.head_dim * effective_rank + n_visual * effective_rank)
        intended_attention_qk_macs = text_qk_macs + dense * n_visual * self.head_dim + proxy_qk_macs
        decomposed_layers = len(self.decompositions)
        svd_gram_and_kb_macs = decomposed_layers * self.num_kv_heads * (n_visual * self.head_dim * self.head_dim + n_visual * self.head_dim * effective_rank)
        lag_profile = self.orbit_trigger_lag_profile.mean((0, 1)).tolist() if self.orbit_trigger_lag_profile is not None else []
        phase_match_rates: list[float] = []
        if self.orbit_trigger_top_lags is not None and self.orbit_trigger_periods:
            top1 = self.orbit_trigger_top_lags[..., 0].float()
            period = float(self.orbit_trigger_periods[0])
            phase_match_rates = [float(((top1 - multiple * period).abs() <= 4).float().mean().item()) for multiple in (1, 2, 3)]
        first_mass = self.first_token_exact_mass.flatten()
        first_mass = first_mass[torch.isfinite(first_mass)]
        first_mass_quantiles = torch.quantile(first_mass, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=first_mass.dtype, device=first_mass.device)).tolist() if first_mass.numel() else [float('nan')] * 5

        def effective_summary(value: float | torch.Tensor) -> tuple[float, list[float]]:
            if isinstance(value, torch.Tensor):
                flat = value.detach().flatten().double()
                quantiles = torch.quantile(flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=flat.dtype, device=flat.device)).tolist()
                return (float(flat.mean().item()), quantiles)
            scalar = float(value)
            return (scalar, [scalar] * 5)
        effective_m1_mean, effective_m1_quantiles = effective_summary(self.sample_effective_m1)
        effective_confirmation_mean, effective_confirmation_quantiles = effective_summary(self.sample_effective_confirmation_m1)
        effective_m3_mean, effective_m3_quantiles = effective_summary(self.sample_effective_m3_base)
        return {'version': self.protocol_version, 'm1': self.m1, 'confirmation_m1': self.confirmation_m1, 'm3_base': self.m3_base, 'sample_mass_switch_gate_active': self.sample_mass_switch_gate, 'sample_mass_continuous_gate_active': self.sample_mass_continuous_gate, 'entry_mass_m3_gate_active': self.entry_mass_m3_gate, 'entry_mass_m3_ratio': self.entry_mass_m3_ratio, 'entry_mass_m3_min': self.entry_mass_m3_min, 'entry_mass_m3_max': self.entry_mass_m3_max, 'entry_mass_m3_update_rows': entry_m3_updates, 'entry_mass_m3_floor_rows': entry_m3_floors, 'entry_mass_m3_ceiling_rows': entry_m3_ceilings, 'entry_mass_m3_mean': float(self.entry_mass_m3_value_sum.sum().item()) / max(entry_m3_updates, 1), 'entry_mass_m3_floor_fraction': entry_m3_floors / max(entry_m3_updates, 1), 'entry_mass_m3_ceiling_fraction': entry_m3_ceilings / max(entry_m3_updates, 1), 'entry_mass_confirmation_gate_active': self.entry_mass_confirmation_gate, 'entry_mass_confirmation_update_rows': entry_confirmation_updates, 'entry_mass_confirmation_mean_entry_mass': float(self.entry_mass_confirmation_value_sum.sum().item()) / max(entry_confirmation_updates, 1), 'entry_mass_confirmation_mean_used_threshold': float(self.entry_mass_confirmation_used_sum.sum().item()) / max(confirmations, 1), 'sample_mass_observed': self.sample_mass_observed, 'sample_mass_log_odds_offset': self.sample_mass_log_odds_offset, 'sample_mass_continuous_log_odds_offset_mean': float(torch.nanmean(self.sample_mass_continuous_log_odds_offset).item()) if self.sample_mass_continuous_log_odds_offset is not None else None, 'sample_mass_selected_branch': self.sample_mass_selected_branch, 'sample_mass_low_m1': self.sample_mass_low_m1, 'sample_mass_low_confirmation_m1': self.sample_mass_low_confirmation_m1, 'sample_mass_low_m3_base': self.sample_mass_low_m3_base, 'sample_effective_m1': effective_m1_mean, 'sample_effective_m1_quantiles_p10_p25_p50_p75_p90': effective_m1_quantiles, 'sample_effective_confirmation_m1': effective_confirmation_mean, 'sample_effective_confirmation_quantiles_p10_p25_p50_p75_p90': effective_confirmation_quantiles, 'sample_effective_m3_base': effective_m3_mean, 'sample_effective_m3_quantiles_p10_p25_p50_p75_p90': effective_m3_quantiles, 'rank': self.rank, 'max_anchor_distance': self.max_anchor_distance, 'checkpoint_schedule': self.checkpoint_schedule, 'selected_layers': list(self.layers), 'processed_generated_tokens': self.decode_tokens, 'num_visual_tokens': n_visual, 'first_token_visual_mass_mean': float(first_mass.mean().item()) if first_mass.numel() else float('nan'), 'first_token_visual_mass_quantiles_p10_p25_p50_p75_p90': first_mass_quantiles, 'first_token_visual_mass_layer_means': torch.nanmean(self.first_token_exact_mass, dim=-1).tolist(), 'eligible_head_rows': eligible, 'dense_head_rows': dense, 'hold_head_rows': hold, 'dense_rate': dense / max(eligible, 1), 'hold_rate': hold / max(eligible, 1), 'transition_enter_rows': int(self.enter_rows.sum().item()), 'transition_reopen_rows': int(self.reopen_rows.sum().item()), 'holding_candidate_rows': holding_candidates, 'proxy_check_rows': proxy_checks, 'confirmation_rows': confirmations, 'confirmation_low_rows': confirmation_low, 'confirmation_high_rows': confirmation_high, 'relaxed_confirmation_pass_rows': relaxed_confirmation_pass, 'relaxed_confirmation_pass_rate': relaxed_confirmation_pass / max(eligible, 1), 'relaxed_share_of_confirmation_passes': relaxed_confirmation_pass / max(confirmation_high, 1), 'proxy_confirmation_rows': proxy_confirmations, 'proxy_confirmation_low_rows': proxy_confirmation_low, 'proxy_confirmation_high_rows': proxy_confirmation_high, 'horizon_confirmation_rows': horizon_confirmations, 'horizon_confirmation_low_rows': horizon_confirmation_low, 'horizon_confirmation_high_rows': horizon_confirmation_high, 'checkpoint_reanchor_rows': checkpoint_reanchors, 'checkpoint_reanchor_rate': checkpoint_reanchors / max(eligible, 1), 'staggered_checkpoint_rows': staggered_checkpoints, 'staggered_checkpoint_rate': staggered_checkpoints / max(eligible, 1), 'staggered_age_fallback_rows': staggered_age_fallbacks, 'staggered_age_fallback_rate': staggered_age_fallbacks / max(eligible, 1), 'checkpoint_pulse_mode': self.checkpoint_pulse_mode, 'checkpoint_pulse_rows': checkpoint_pulses, 'checkpoint_pulse_rate': checkpoint_pulses / max(eligible, 1), 'checkpoint_pulse_mean_mass': float(self.checkpoint_pulse_mass_sum.sum().item()) / max(checkpoint_pulses, 1), 'checkpoint_pulse_mean_boost': float(self.checkpoint_pulse_boost_sum.sum().item()) / max(checkpoint_pulses, 1), 'orbit_all_layer_cycle_clean_active': self.orbit_all_layer_cycle_clean, 'orbit_text_intervention': self.orbit_text_intervention, 'orbit_intervention_duration': self.orbit_intervention_duration, 'orbit_suffix_cycles': self.orbit_suffix_cycles, 'orbit_soft_penalty': self.orbit_soft_penalty, 'orbit_topk': self.orbit_topk, 'orbit_phase_radius': self.orbit_phase_radius, 'orbit_keep_recent': self.orbit_keep_recent, 'orbit_visual_drop_fraction': self.orbit_visual_drop_fraction, 'orbit_prompt_boost': self.orbit_prompt_boost, 'orbit_prompt_tail_tokens': self.orbit_prompt_tail_tokens, 'orbit_detected_period': self.orbit_detected_period, 'orbit_mask_generated_span': [self.orbit_mask_generated_start, self.orbit_mask_generated_end], 'orbit_trigger_generated_lag_bins': ['1-8', '9-16', '17-32', '33-64', '65-128', '129-256', '257-512', '513+'], 'orbit_trigger_generated_lag_mass_mean': lag_profile, 'orbit_trigger_top1_phase_match_rate_p_2p_3p': phase_match_rates, 'orbit_rule': {'cycle_error_max': self.orbit_error_threshold, 'cycle_ratio_max': self.orbit_ratio_threshold, 'confirmation_root_gap_p50_min': self.orbit_root_gap_threshold, 'rolling_window_tokens': self.orbit_window, 'period_range': [self.orbit_min_period, self.orbit_max_period], 'scan_stride': self.orbit_scan_stride}, 'orbit_trigger_count': len(self.orbit_trigger_tokens), 'orbit_trigger_tokens': self.orbit_trigger_tokens, 'orbit_trigger_periods': self.orbit_trigger_periods, 'orbit_driver_head_count': int(self.orbit_driver_mask.sum().item()) if self.orbit_driver_mask is not None else 0, 'orbit_all_layer_cycle_clean_rows': orbit_all_layer_cycle_cleans, 'orbit_all_layer_cycle_clean_tokens': sorted(self.orbit_reset_tokens), 'orbit_all_layer_cycle_clean_mean_mass': float(self.orbit_all_layer_cycle_clean_mass_sum.sum().item()) / max(orbit_all_layer_cycle_cleans, 1), 'history_reset_after': self.history_reset_after, 'history_reset_layers': [self.history_reset_layer_start, self.history_reset_layer_end], 'history_reset_rows': history_resets, 'history_reset_rate': history_resets / max(eligible, 1), 'history_reset_tokens': len(self.history_reset_tokens), 'history_reset_first_token': min(self.history_reset_tokens) if self.history_reset_tokens else None, 'history_reset_mean_unmasked_mass': float(self.history_reset_mass_sum.sum().item()) / max(history_resets, 1), 'max_failed_checkpoint_streak': int(self.max_failed_checkpoint_streak.max().item()), 'spectral_fallback_tau': self.spectral_fallback_tau, 'spectral_fallback_rows': spectral_fallbacks, 'spectral_fallback_rate': spectral_fallbacks / max(eligible, 1), 'spectral_fallback_mean_risk': float(self.spectral_fallback_risk_sum.sum().item()) / max(spectral_fallbacks, 1), 'distance_forced_rows': int(self.distance_forced_rows.sum().item()), 'confirmation_rate': confirmations / max(eligible, 1), 'confirmation_given_holding_rate': confirmations / max(holding_candidates, 1), 'confirmation_pass_rate': confirmation_high / max(confirmations, 1), 'confirmation_fail_rate': confirmation_low / max(confirmations, 1), 'confirmation_passed_overall_rate': confirmation_high / max(eligible, 1), 'confirmation_failed_overall_rate': confirmation_low / max(eligible, 1), 'proxy_check_rate': proxy_checks / max(eligible, 1), 'proxy_trigger_rate': proxy_confirmations / max(proxy_checks, 1), 'proxy_confirmation_pass_rate': proxy_confirmation_high / max(proxy_confirmations, 1), 'horizon_confirmation_rate': horizon_confirmations / max(eligible, 1), 'prompt_nonvisual_tokens': self.prompt_nonvisual_tokens, 'final_nonvisual_tokens': self.last_nonvisual_tokens, 'nonvisual_token_growth': self.last_nonvisual_tokens / max(self.prompt_nonvisual_tokens, 1), 'm1_length_adjusted_at_final_token': self._length_adjusted_threshold(self.m1, self.prompt_nonvisual_tokens, self.last_nonvisual_tokens), 'm3_base_length_adjusted_at_final_token': self._length_adjusted_threshold(self.m3_base, self.prompt_nonvisual_tokens, self.last_nonvisual_tokens), 'visual_key_fraction_of_baseline_qk': eligible * n_visual / max(self.total_key_rows, 1), 'baseline_attention_qk_macs': baseline_attention_qk_macs, 'intended_attention_qk_macs_excluding_svd_preprocess': intended_attention_qk_macs, 'intended_attention_qk_ratio_excluding_svd_preprocess': intended_attention_qk_macs / max(baseline_attention_qk_macs, 1), 'svd_gram_and_kb_macs': svd_gram_and_kb_macs, 'intended_attention_qk_ratio_including_gram_and_kb': (intended_attention_qk_macs + svd_gram_and_kb_macs) / max(baseline_attention_qk_macs, 1), 'qk_cost_scope': 'QK MACs only; SVD ratio includes Gram and K@B but excludes eigh, softmax, AV, projections, and kernel overhead', 'length_adjusted_gate_active': self.length_adjusted_gate, 'proxy_error_count': errors, 'proxy_mass_mae': float(self.proxy_abs_error_sum.sum().item() / max(errors, 1)), 'mean_explained_energy': float(self.energy_sum.sum().item() / max(energy_count, 1)), 'rank_proxy_vs_full_visual_qk': recurring_vs_full_visual_qk, 'rank_proxy_vs_dense_visual_qk_av': recurring_vs_full_visual_qk / 2.0, 'layer_dense_rows': self.dense_rows.sum(-1).tolist(), 'layer_hold_rows': self.hold_rows.sum(-1).tolist(), 'layer_enter_rows': self.enter_rows.sum(-1).tolist(), 'layer_reopen_rows': self.reopen_rows.sum(-1).tolist(), 'actual_visual_qk_saving': False, 'generation_uses_only_online_hold_signals': True}

def _svd_checkpoint_forward(self: Any, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor | None, past_key_values: Any=None, cache_position: torch.Tensor | None=None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    manager: SVDCheckpointHoldGate = self._svd_checkpoint_hold_manager
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
        key, value = past_key_values.update(key, value, self.layer_idx, {'sin': sin, 'cos': cos, 'cache_position': cache_position})
    kv_keys = key
    repeated_key = repeat_kv(key, self.num_key_value_groups)
    repeated_value = repeat_kv(value, self.num_key_value_groups)
    logits = torch.matmul(query, repeated_key.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        logits += attention_mask[:, :, :, :repeated_key.shape[-2]]
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = F.dropout(probabilities, p=0.0 if not self.training else self.attention_dropout, training=self.training)
    dense_output = torch.matmul(probabilities, repeated_value)
    output = manager.apply(self.layer_idx, query, kv_keys, logits, probabilities, repeated_value, dense_output, attention_mask)
    output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return (self.o_proj(output), None)

def _orbit_all_layer_cycle_clean_forward(self: Any, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor | None, past_key_values: Any=None, cache_position: torch.Tensor | None=None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Apply the detector-selected text intervention in untargeted layers."""
    manager: SVDCheckpointHoldGate = self._svd_checkpoint_hold_manager
    if not manager.all_layer_clean_due(cache_position):
        return self._svd_checkpoint_hold_original_forward(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, cache_position=cache_position, **kwargs)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx, {'sin': sin, 'cos': cos, 'cache_position': cache_position})
    repeated_key = repeat_kv(key, self.num_key_value_groups)
    repeated_value = repeat_kv(value, self.num_key_value_groups)
    logits = torch.matmul(query, repeated_key.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        logits += attention_mask[:, :, :, :repeated_key.shape[-2]]
    current_position = repeated_key.shape[-2] - 1
    logits = manager.apply_orbit_text_intervention(logits, current_position)
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = F.dropout(probabilities, p=0.0 if not self.training else self.attention_dropout, training=self.training)
    output = torch.matmul(probabilities, repeated_value)
    output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return (self.o_proj(output), None)

def install_svd_checkpoint_hold_gate(model: Any, manager: SVDCheckpointHoldGate) -> None:
    for layer in manager.layers:
        attention = model.model.language_model.layers[layer].self_attn
        conflict = any((getattr(attention, flag, False) for flag in ('_oracle_replay_wrapped', '_refresh_causal_wrapped', '_phase_attention_wrapped', '_evidence_support_wrapped', '_visual_evidence_cache_wrapped', '_oracle_last_dense_gate_wrapped', '_lowrank_visual_k_wrapped', '_svd_checkpoint_hold_wrapped')))
        if conflict:
            raise RuntimeError('SVD cosine HOLD requires untouched attention')
        attention._svd_checkpoint_hold_manager = manager
        attention._svd_checkpoint_hold_original_forward = attention.forward
        attention.forward = MethodType(_svd_checkpoint_forward, attention)
        attention._svd_checkpoint_hold_wrapped = True

def install_orbit_all_layer_clean_gate(model: Any, manager: SVDCheckpointHoldGate) -> None:
    """Wrap untargeted layers only for the explicit all-layer experiment."""
    for layer, block in enumerate(model.model.language_model.layers):
        if layer in manager.layer_offset:
            continue
        attention = block.self_attn
        conflict = any((getattr(attention, flag, False) for flag in ('_oracle_replay_wrapped', '_refresh_causal_wrapped', '_phase_attention_wrapped', '_evidence_support_wrapped', '_visual_evidence_cache_wrapped', '_oracle_last_dense_gate_wrapped', '_lowrank_visual_k_wrapped', '_svd_checkpoint_hold_wrapped', '_orbit_all_layer_cycle_clean_wrapped')))
        if conflict:
            raise RuntimeError('all-layer clean token requires untouched attention')
        attention._svd_checkpoint_hold_manager = manager
        attention._svd_checkpoint_hold_original_forward = attention.forward
        attention.forward = MethodType(_orbit_all_layer_cycle_clean_forward, attention)
        attention._orbit_all_layer_cycle_clean_wrapped = True
