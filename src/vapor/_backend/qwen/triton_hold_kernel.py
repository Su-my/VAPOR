# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""GPU-resident routing and counters with no host head selection.

Every head computes exact text QK once. Ordinary HOLD computes only rank-r
visual logits and text AV; Dense/confirmation/checkpoint compute full visual
QK/AV. Scratch logits avoid recomputing text when a proxy triggers confirmation.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
COUNTER_NAMES = ('dense_rows', 'hold_rows', 'enter_rows', 'reopen_rows', 'proxy_check_rows', 'confirmation_rows', 'confirmation_low_rows', 'confirmation_high_rows', 'relaxed_confirmation_pass_rows', 'proxy_confirmation_rows', 'proxy_confirmation_low_rows', 'proxy_confirmation_high_rows', 'horizon_confirmation_rows', 'horizon_confirmation_low_rows', 'horizon_confirmation_high_rows', 'checkpoint_reanchor_rows', 'holding_candidate_rows', 'distance_forced_rows', 'entry_mass_m3_update_rows', 'entry_mass_m3_floor_rows', 'entry_mass_m3_ceiling_rows', 'proxy_error_count')

@triton.jit
def _exact_logits(K, Qrow, Mask, seq, valid, kv_head, k_stride, scale, D: tl.constexpr, HAS_MASK: tl.constexpr):
    dim = tl.arange(0, D)
    key = tl.load(K + kv_head * k_stride + seq[:, None] * D + dim[None, :], mask=valid[:, None], other=0).to(tl.float32)
    dots = tl.sum(key.to(tl.float64) * Qrow[None, :].to(tl.float64), axis=1).to(tl.float32)
    logits = (dots.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    if HAS_MASK:
        bias = tl.load(Mask + seq, mask=valid, other=0).to(tl.float32)
        logits = (logits + bias).to(tl.bfloat16).to(tl.float32)
    return tl.where(valid, logits, float('-inf'))

@triton.jit
def _merge_partition(old_max, old_sum, logits):
    current_max = tl.max(logits, axis=0)
    new_max = tl.maximum(old_max, current_max)
    alpha = tl.exp(old_max - new_max)
    total = old_sum * alpha + tl.sum(tl.exp(logits - new_max), axis=0)
    return (new_max, total)

@triton.jit(do_not_specialize=['n_keys', 'k_stride', 'scratch_stride', 'partition_stride'])
def text_logits(Q, K, Mask, VisualFlags, Scratch, RoundingFlags, PartitionMax, PartitionSum, PartitionRows, n_keys, k_stride, scratch_stride, partition_stride, scale, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr=64):
    kv_head = tl.program_id(0)
    block = tl.program_id(1)
    rows = tl.arange(0, 16)
    heads = kv_head * GROUPS + rows
    head_valid = rows < GROUPS
    dim = tl.arange(0, D)
    seq = block * BLOCK + tl.arange(0, BLOCK)
    in_range = seq < n_keys
    visual = tl.load(VisualFlags + seq, mask=in_range, other=False)
    valid = in_range & ~visual
    if tl.sum(valid.to(tl.int32), axis=0) > 0:
        query = tl.load(Q + heads[:, None] * D + dim[None, :], mask=head_valid[:, None], other=0)
        key = tl.load(K + kv_head * k_stride + seq[None, :] * D + dim[:, None], mask=valid[None, :], other=0)
        dots = tl.dot(query, key, out_dtype=tl.float32)
        absolute = tl.dot(tl.abs(query), tl.abs(key), out_dtype=tl.float32)
        bound = absolute * (2.0 * D * 5.960464477539063e-08)
        uncertain = (dots - bound).to(tl.bfloat16) != (dots + bound).to(tl.bfloat16)
        logits = (dots.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        if HAS_MASK:
            bias = tl.load(Mask + seq, mask=valid, other=0).to(tl.float32)
            logits = (logits + bias[None, :]).to(tl.bfloat16).to(tl.float32)
            uncertain = uncertain & (bias[None, :] > float('-inf'))
        pointers = heads[:, None] * scratch_stride + seq[None, :]
        tl.store(Scratch + pointers, logits, mask=head_valid[:, None] & valid[None, :])
        tl.store(RoundingFlags + pointers, uncertain, mask=head_valid[:, None] & valid[None, :])

@triton.jit(do_not_specialize=['n_keys', 'k_stride', 'scratch_stride', 'partition_stride'])
def correct_text_logits(Q, K, Mask, VisualFlags, Scratch, RoundingFlags, PartitionMax, PartitionSum, PartitionRows, n_keys, k_stride, scratch_stride, partition_stride, scale, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr=64):
    head = tl.program_id(0)
    block = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    seq = block * BLOCK + lane
    in_range = seq < n_keys
    visual = tl.load(VisualFlags + seq, mask=in_range, other=False)
    valid = in_range & ~visual
    logits = tl.load(Scratch + head * scratch_stride + seq, mask=valid, other=float('-inf')).to(tl.float32)
    needs = tl.load(RoundingFlags + head * scratch_stride + seq, mask=valid, other=False)
    number = tl.sum(needs.to(tl.int32), axis=0)
    if number > 0:
        dim = tl.arange(0, D)
        query = tl.load(Q + head * D + dim).to(tl.float32)
        original_needs = needs
        while number > 0:
            index = tl.min(tl.where(needs, lane, BLOCK), axis=0)
            selected = block * BLOCK + index
            key = tl.load(K + head // GROUPS * k_stride + selected * D + dim).to(tl.float32)
            dot = tl.sum(query.to(tl.float64) * key.to(tl.float64), axis=0).to(tl.float32)
            exact = (dot.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
            if HAS_MASK:
                bias = tl.load(Mask + selected).to(tl.float32)
                exact = (exact + bias).to(tl.bfloat16).to(tl.float32)
            logits = tl.where(lane == index, exact, logits)
            needs = needs & (lane != index)
            number -= 1
        tl.store(Scratch + head * scratch_stride + seq, logits, mask=original_needs)
    maximum = tl.max(logits, axis=0)
    safe = tl.where(maximum > float('-inf'), maximum, 0.0)
    total = tl.sum(tl.exp(logits - safe), axis=0)
    offset = head * partition_stride + block
    tl.store(PartitionMax + offset, maximum)
    tl.store(PartitionSum + offset, total)
    tl.store(PartitionRows + offset, tl.sum(valid.to(tl.int32), axis=0))

@triton.jit(do_not_specialize=['n_keys', 'n_visual', 'k_stride', 'v_stride', 'low_stride', 'scratch_stride', 'partition_stride', 'token', 'layer'])
def route_decode(Q, K, V, Mask, VisualFlags, VisualPositions, Basis, LowKeys, Scratch, Output, TextPartitionMax, TextPartitionSum, TextPartitionRows, RouteFlags, RouteFloat, RouteWork, State, Counters, FloatCounters, WorkCounters, n_keys, n_visual, k_stride, v_stride, low_stride, scratch_stride, partition_stride, token, layer, scale, m1, confirmation_threshold, m3, ratio, floor, ceiling, horizon, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, RANK: tl.constexpr, HAS_MASK: tl.constexpr, ENTRY_M3: tl.constexpr, BLOCK: tl.constexpr=32, AGE_DECAY: tl.constexpr=True, USE_HORIZON: tl.constexpr=True, SHARED_AGE_GATES: tl.constexpr=False):
    head = tl.program_id(0)
    kv_head = head // GROUPS
    row = layer * H + head
    dim = tl.arange(0, D)
    seq_lane = tl.arange(0, BLOCK)
    holding = tl.load(State[0] + head)
    ready = tl.load(State[1] + head)
    anchor_token = tl.load(State[2] + head).to(tl.int32)
    age = tl.where(anchor_token >= 0, token - anchor_token, -1)
    checkpoint = holding & ready & (age >= horizon) & USE_HORIZON
    proxy_checked = holding & ready & ~checkpoint
    phase = tl.minimum(tl.maximum((age - 2).to(tl.float32), 0.0) / (horizon - 2), 1.0)
    episode_base = tl.load(State[5] + head)
    threshold = episode_base if ENTRY_M3 else m3
    if AGE_DECAY:
        threshold = threshold * libdevice.cos(1.5707963267948966 * phase)
    threshold = tl.where(age >= 0, threshold, 0.0)
    if SHARED_AGE_GATES:
        threshold = tl.where(age >= 0, tl.maximum(threshold, 0.0), m3)
        tl.store(RouteFloat + 3 * H + head, threshold)
    query = tl.load(Q + head * D + dim).to(tl.float32)
    proxy_mass = tl.full((), float('nan'), tl.float32)
    stay_hold = tl.full((), False, tl.int1)
    text_max = tl.full((), float('-inf'), tl.float32)
    text_sum = tl.full((), 0.0, tl.float32)
    log_z_text = tl.full((), float('nan'), tl.float32)
    text_qk_rows = tl.full((), 0, tl.int64)
    visual_qk_rows = tl.full((), 0, tl.int64)
    visual_av_rows = tl.full((), 0, tl.int64)
    n_partitions = tl.cdiv(n_keys, BLOCK)
    for start in range(0, n_partitions, BLOCK):
        part = start + seq_lane
        valid = part < n_partitions
        offset = head * partition_stride + part
        maxima = tl.load(TextPartitionMax + offset, mask=valid, other=float('-inf'))
        sums = tl.load(TextPartitionSum + offset, mask=valid, other=0.0)
        counts = tl.load(TextPartitionRows + offset, mask=valid, other=0)
        text_qk_rows += tl.sum(counts.to(tl.int64), axis=0)
        current_max = tl.max(maxima, axis=0)
        if current_max > float('-inf'):
            new_max = tl.maximum(text_max, current_max)
            text_sum = text_sum * tl.exp(text_max - new_max) + tl.sum(sums * tl.exp(maxima - new_max), axis=0)
            text_max = new_max
    log_z_text = text_max + tl.log(text_sum)
    if proxy_checked:
        rank_lane = tl.arange(0, RANK)
        basis = tl.load(Basis + kv_head * D * RANK + dim[:, None] * RANK + rank_lane[None, :])
        low_query = tl.sum(query[:, None] * basis, axis=0)
        proxy_max = tl.full((), float('-inf'), tl.float32)
        proxy_sum = tl.full((), 0.0, tl.float32)
        for start in range(0, n_visual, BLOCK):
            visual_index = start + seq_lane
            valid = visual_index < n_visual
            low = tl.load(LowKeys + kv_head * low_stride + visual_index[:, None] * RANK + rank_lane[None, :], mask=valid[:, None], other=0.0)
            logits = tl.sum(low * low_query[None, :], axis=1) * scale
            if HAS_MASK:
                visual_seq = tl.load(VisualPositions + visual_index, mask=valid, other=0)
                logits += tl.load(Mask + visual_seq, mask=valid, other=0.0).to(tl.float32)
            logits = tl.where(valid, logits, float('-inf'))
            if tl.max(logits, axis=0) > float('-inf'):
                proxy_max, proxy_sum = _merge_partition(proxy_max, proxy_sum, logits)
        log_z_proxy = proxy_max + tl.log(proxy_sum)
        proxy_mass = 1.0 / (1.0 + tl.exp(log_z_text - log_z_proxy))
        stay_hold = proxy_mass < threshold
        previously_recorded = tl.atomic_cas(FloatCounters[5] + layer, 0, 1)
        if previously_recorded == 0:
            energy = tl.load(FloatCounters[6] + layer).to(tl.float64)
            tl.store(FloatCounters[3] + layer, energy)
            tl.store(FloatCounters[4] + layer, 1)
    dense_now = ~stay_hold
    log_z_dense = tl.full((), float('nan'), tl.float32)
    if dense_now:
        dense_max = tl.full((), float('-inf'), tl.float32)
        dense_sum = tl.full((), 0.0, tl.float32)
        for start in range(0, n_visual, BLOCK):
            index = start + seq_lane
            valid = index < n_visual
            seq = tl.load(VisualPositions + index, mask=valid, other=0)
            logits = _exact_logits(K, query, Mask, seq, valid, kv_head, k_stride, scale, D, HAS_MASK)
            tl.store(Scratch + head * scratch_stride + seq, logits, mask=valid)
            if tl.max(logits, axis=0) > float('-inf'):
                dense_max, dense_sum = _merge_partition(dense_max, dense_sum, logits)
            visual_qk_rows += tl.sum(valid.to(tl.int64), axis=0)
        log_z_visual = dense_max + tl.log(dense_sum)
        zmax = tl.maximum(log_z_text, log_z_visual)
        log_z_dense = zmax + tl.log(tl.exp(log_z_text - zmax) + tl.exp(log_z_visual - zmax))
    tl.store(RouteFlags + head, dense_now.to(tl.int32))
    tl.store(RouteFlags + H + head, proxy_checked.to(tl.int32))
    tl.store(RouteFlags + 2 * H + head, checkpoint.to(tl.int32))
    tl.store(RouteFloat + head, proxy_mass)
    tl.store(RouteFloat + H + head, log_z_text)
    tl.store(RouteFloat + 2 * H + head, log_z_dense)
    tl.store(RouteWork + head, visual_qk_rows)
    tl.store(RouteWork + H + head, text_qk_rows)

@triton.jit(do_not_specialize=['n_keys', 'v_stride', 'scratch_stride', 'partition_stride'])
def attention_values(V, VisualFlags, Scratch, RouteFlags, RouteFloat, PartialReadout, PartialVisualReadout, PartialMass, PartialVisualRows, n_keys, v_stride, scratch_stride, partition_stride, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr=64):
    head = tl.program_id(0)
    block = tl.program_id(1)
    dim = tl.arange(0, D)
    seq = block * BLOCK + tl.arange(0, BLOCK)
    in_range = seq < n_keys
    visual = tl.load(VisualFlags + seq, mask=in_range, other=False)
    dense_now = tl.load(RouteFlags + head) != 0
    valid = in_range & (dense_now | ~visual)
    log_z_text = tl.load(RouteFloat + H + head)
    log_z_dense = tl.load(RouteFloat + 2 * H + head)
    z = tl.where(dense_now, log_z_dense, log_z_text)
    readout = tl.full((D,), 0.0, tl.float32)
    visual_readout = tl.full((D,), 0.0, tl.float32)
    mass = tl.full((), 0.0, tl.float32)
    visual_rows = tl.full((), 0, tl.int32)
    if tl.sum(valid.to(tl.int32), axis=0) > 0:
        logits = tl.load(Scratch + head * scratch_stride + seq, mask=valid, other=float('-inf')).to(tl.float32)
        probabilities = tl.exp(logits - z)
        if dense_now:
            probabilities = probabilities.to(tl.bfloat16).to(tl.float32)
        value = tl.load(V + head // GROUPS * v_stride + seq[:, None] * D + dim[None, :], mask=valid[:, None], other=0).to(tl.float32)
        readout = tl.sum(probabilities[:, None] * value, axis=0)
        if dense_now:
            visual_p = tl.where(visual & valid, probabilities, 0.0)
            mass = tl.sum(visual_p, axis=0)
            visual_readout = tl.sum(visual_p[:, None] * value, axis=0)
            visual_rows = tl.sum((visual & valid).to(tl.int32), axis=0)
    part = head * partition_stride + block
    tl.store(PartialReadout + part * D + dim, readout)
    tl.store(PartialVisualReadout + part * D + dim, visual_readout)
    tl.store(PartialMass + part, mass)
    tl.store(PartialVisualRows + part, visual_rows)

@triton.jit(do_not_specialize=['n_keys', 'n_visual', 'partition_stride', 'token', 'layer'])
def finalize_decode(Output, State, Counters, FloatCounters, WorkCounters, RouteFlags, RouteFloat, RouteWork, PartialReadout, PartialVisualReadout, PartialMass, PartialVisualRows, n_keys, n_visual, partition_stride, token, layer, m1, confirmation_threshold, ratio, floor, ceiling, D: tl.constexpr, H: tl.constexpr, ENTRY_M3: tl.constexpr, BLOCK: tl.constexpr=64, PART_BLOCK: tl.constexpr=8, SHARED_AGE_GATES: tl.constexpr=False, SHARED_HORIZON: tl.constexpr=512):
    head = tl.program_id(0)
    row = layer * H + head
    dim = tl.arange(0, D)
    part_lane = tl.arange(0, PART_BLOCK)
    holding = tl.load(State[0] + head)
    ready = tl.load(State[1] + head)
    dense_now = tl.load(RouteFlags + head) != 0
    proxy_checked = tl.load(RouteFlags + H + head) != 0
    checkpoint = tl.load(RouteFlags + 2 * H + head) != 0
    proxy_mass = tl.load(RouteFloat + head)
    stay_hold = ~dense_now
    visual_qk_rows = tl.load(RouteWork + head)
    text_qk_rows = tl.load(RouteWork + H + head)
    visual_av_rows = tl.full((), 0, tl.int64)
    readout = tl.full((D,), 0.0, tl.float32)
    visual_readout = tl.full((D,), 0.0, tl.float32)
    exact_mass = tl.full((), 0.0, tl.float32)
    n_partitions = tl.cdiv(n_keys, BLOCK)
    for start in range(0, n_partitions, PART_BLOCK):
        part = start + part_lane
        valid = part < n_partitions
        offset = head * partition_stride + part
        partial = tl.load(PartialReadout + offset[:, None] * D + dim[None, :], mask=valid[:, None], other=0.0)
        readout += tl.sum(partial, axis=0)
        if dense_now:
            visual_partial = tl.load(PartialVisualReadout + offset[:, None] * D + dim[None, :], mask=valid[:, None], other=0.0)
            visual_readout += tl.sum(visual_partial, axis=0)
            mass = tl.load(PartialMass + offset, mask=valid, other=0.0)
            exact_mass += tl.sum(mass, axis=0)
            rows = tl.load(PartialVisualRows + offset, mask=valid, other=0)
            visual_av_rows += tl.sum(rows.to(tl.int64), axis=0)
    exact_mass = tl.where(dense_now, exact_mass, float('nan'))
    if stay_hold:
        anchor = tl.load(State[3] + head * D + dim)
        readout = (1.0 - proxy_mass) * readout + proxy_mass * anchor
    tl.store(Output + head * D + dim, readout)
    confirmation = holding & ~stay_hold
    passed = confirmation & (exact_mass >= m1)
    failed = confirmation & ~passed
    horizon_confirmation = confirmation & checkpoint
    checkpoint_reanchor = horizon_confirmation & failed
    enter = ~holding & ready & (exact_mass < m1)
    holding_after = stay_hold | dense_now & ready & (exact_mass < m1)
    update_anchor = dense_now
    if dense_now:
        anchor_readout = tl.where(exact_mass > 0, visual_readout / tl.maximum(exact_mass, 1e-20), 0.0)
        tl.store(State[3] + head * D + dim, anchor_readout)
    if update_anchor:
        tl.store(State[2] + head, token)
        tl.store(State[4] + head, n_keys - n_visual)
        tl.store(State[1] + head, True)
    tl.store(State[0] + head, holding_after)
    episode_update = tl.full((), False, tl.int1)
    episode_floor = tl.full((), False, tl.int1)
    episode_ceiling = tl.full((), False, tl.int1)
    if ENTRY_M3:
        episode_update = enter | checkpoint_reanchor
        if episode_update:
            raw = ratio * exact_mass
            calibrated = tl.minimum(tl.maximum(raw, floor), ceiling)
            tl.store(State[5] + head, calibrated)
            episode_floor = raw <= floor
            episode_ceiling = raw >= ceiling
            previous = tl.load(FloatCounters[1] + row)
            tl.store(FloatCounters[1] + row, previous + calibrated.to(tl.float64))
    proxy_confirmation = confirmation & proxy_checked
    if proxy_confirmation:
        error = tl.abs(proxy_mass - exact_mass).to(tl.float64)
        tl.store(FloatCounters[0] + row, tl.load(FloatCounters[0] + row) + error)
    if token == 1:
        tl.store(FloatCounters[2] + row, exact_mass.to(tl.float64))
    deltas = (dense_now, stay_hold, enter, passed, proxy_checked, confirmation, failed, passed, passed & (exact_mass < m1), proxy_confirmation, proxy_confirmation & failed, proxy_confirmation & passed, horizon_confirmation, horizon_confirmation & failed, horizon_confirmation & passed, checkpoint_reanchor, holding, horizon_confirmation, episode_update, episode_floor, episode_ceiling, proxy_confirmation)
    for i in tl.static_range(22):
        tl.store(Counters[i] + row, tl.load(Counters[i] + row) + deltas[i].to(tl.int64))
    tl.store(WorkCounters[0] + row, tl.load(WorkCounters[0] + row) + visual_qk_rows)
    tl.store(WorkCounters[1] + row, tl.load(WorkCounters[1] + row) + visual_av_rows)
    tl.store(WorkCounters[2] + row, tl.load(WorkCounters[2] + row) + text_qk_rows)
