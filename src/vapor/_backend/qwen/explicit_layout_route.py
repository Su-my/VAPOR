# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Historical reduction layouts made explicit; exact visual QK stays FP64.

Layouts are from the saved original compiled kernel. Proxy logits are produced
in parallel; their ordered FP32 normalizer uses the same 64x16 slice as before.
The original TL/libdevice cosine is computed by the preceding proxy kernel.
"""
import triton.experimental.gluon as g
import triton.experimental.gluon.language as gl

@g.jit(do_not_specialize=['n_keys', 'n_visual', 'k_stride', 'v_stride', 'low_stride', 'scratch_stride', 'partition_stride', 'token', 'layer'])
def route_decode(Q, K, V, Mask, VisualFlags, VisualPositions, Basis, LowKeys, Scratch, Output, TextPartitionMax, TextPartitionSum, TextPartitionRows, RouteFlags, RouteFloat, RouteWork, State, Counters, FloatCounters, WorkCounters, n_keys, n_visual, k_stride, v_stride, low_stride, scratch_stride, partition_stride, token, layer, scale, m1, confirmation_threshold, m3, ratio, floor, ceiling, horizon, D: gl.constexpr, H: gl.constexpr, GROUPS: gl.constexpr, RANK: gl.constexpr, HAS_MASK: gl.constexpr, ENTRY_M3: gl.constexpr, BLOCK: gl.constexpr=64, AGE_DECAY: gl.constexpr=True, USE_HORIZON: gl.constexpr=True, SHARED_AGE_GATES: gl.constexpr=False):
    standard: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    dense_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [2, 4], [1, 0])
    proxy_layout: gl.constexpr = gl.BlockedLayout([1, 1], [2, 16], [8, 1], [1, 0])
    head = gl.program_id(0)
    kv_head = head // GROUPS
    holding = gl.load(State[0] + head)
    ready = gl.load(State[1] + head)
    anchor_token = gl.load(State[2] + head).to(gl.int32)
    age = gl.where(anchor_token >= 0, token - anchor_token, -1)
    checkpoint = holding & ready & (age >= horizon) & USE_HORIZON
    proxy_checked = holding & ready & ~checkpoint
    threshold = gl.load(RouteFloat + 3 * H + head)
    dim = gl.arange(0, D, layout=standard)
    query = gl.load(Q + head * D + dim).to(gl.float32)
    seq_lane = gl.arange(0, BLOCK, layout=standard)
    text_max = float('-inf')
    text_sum = 0.0
    text_qk_rows = gl.sum(gl.full((BLOCK,), 0, gl.int64, layout=standard), 0)
    visual_qk_rows = gl.sum(gl.full((BLOCK,), 0, gl.int64, layout=standard), 0)
    n_partitions = (n_keys + BLOCK - 1) // BLOCK
    for start in range(0, n_partitions, BLOCK):
        part = start + seq_lane
        valid = part < n_partitions
        offset = head * partition_stride + part
        maxima = gl.load(TextPartitionMax + offset, mask=valid, other=gl.full((BLOCK,), float('-inf'), gl.float32, layout=standard))
        sums = gl.load(TextPartitionSum + offset, mask=valid, other=gl.full((BLOCK,), 0.0, gl.float32, layout=standard))
        counts = gl.load(TextPartitionRows + offset, mask=valid, other=gl.full((BLOCK,), 0, gl.int32, layout=standard))
        text_qk_rows += gl.sum(counts.to(gl.int64), 0)
        current_max = gl.max(maxima, 0)
        if current_max > float('-inf'):
            new_max = gl.maximum(text_max, current_max)
            text_sum = text_sum * gl.exp(text_max - new_max) + gl.sum(sums * gl.exp(maxima - new_max), 0)
            text_max = new_max
    log_z_text = text_max + gl.log(text_sum)
    proxy_mass = float('nan')
    stay_hold = token < 0
    if proxy_checked:
        proxy_max = float('-inf')
        proxy_sum = 0.0
        p_lane = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, proxy_layout))
        for start in range(0, n_visual, BLOCK):
            index = start + p_lane
            valid = index < n_visual
            logits = gl.load(Output + head * n_visual + index, mask=valid, other=gl.full((BLOCK,), float('-inf'), gl.float32, layout=gl.SliceLayout(1, proxy_layout)))
            if gl.max(logits, 0) > float('-inf'):
                current_max = gl.max(logits, 0)
                new_max = gl.maximum(proxy_max, current_max)
                proxy_sum = proxy_sum * gl.exp(proxy_max - new_max) + gl.sum(gl.exp(logits - new_max), 0)
                proxy_max = new_max
        log_z_proxy = proxy_max + gl.log(proxy_sum)
        proxy_mass = 1.0 / (1.0 + gl.exp(log_z_text - log_z_proxy))
        stay_hold = proxy_mass < threshold
        atomic_lane = gl.arange(0, 256, layout=standard)
        recorded_lanes = gl.inline_asm_elementwise('{.reg .pred p; setp.eq.s32 p, $2, 0; mov.u32 $0, 1; @p atom.global.cas.b32 $0, [$1], 0, 1;}', constraints='=r,l,r', args=[FloatCounters[5] + layer + atomic_lane, atomic_lane], dtype=gl.int32, is_pure=False, pack=1)
        previously_recorded = gl.min(recorded_lanes, 0)
        if previously_recorded == 0:
            energy = gl.load(FloatCounters[6] + layer).to(gl.float64)
            gl.store(FloatCounters[3] + layer, energy)
            gl.store(FloatCounters[4] + layer, 1)
    dense_now = ~stay_hold
    log_z_dense = float('nan')
    if dense_now:
        dense_max = float('-inf')
        dense_sum = 0.0
        d_lane = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, dense_layout))
        d_dim = gl.arange(0, D, layout=gl.SliceLayout(0, dense_layout))
        q_dense = gl.convert_layout(query, gl.SliceLayout(0, dense_layout))
        for start in range(0, n_visual, BLOCK):
            index = start + d_lane
            valid = index < n_visual
            seq = gl.load(VisualPositions + index, mask=valid, other=gl.full((BLOCK,), 0, gl.int64, layout=gl.SliceLayout(1, dense_layout)))
            key = gl.load(K + kv_head * k_stride + seq[:, None] * D + d_dim[None, :], mask=valid[:, None], other=gl.full((BLOCK, D), 0, K.dtype.element_ty, layout=dense_layout)).to(gl.float32)
            dots = gl.sum(key.to(gl.float64) * q_dense[None, :].to(gl.float64), 1).to(gl.float32)
            logits = (dots.to(gl.bfloat16).to(gl.float32) * scale).to(gl.bfloat16).to(gl.float32)
            if HAS_MASK:
                bias = gl.load(Mask + seq, mask=valid, other=gl.full((BLOCK,), 0, Mask.dtype.element_ty, layout=gl.SliceLayout(1, dense_layout))).to(gl.float32)
                logits = (logits + bias).to(gl.bfloat16).to(gl.float32)
            logits = gl.where(valid, logits, float('-inf'))
            gl.store(Scratch + head * scratch_stride + gl.convert_layout(seq, standard), gl.convert_layout(logits, standard), mask=gl.convert_layout(valid, standard))
            if gl.max(logits, 0) > float('-inf'):
                current_max = gl.max(logits, 0)
                new_max = gl.maximum(dense_max, current_max)
                dense_sum = dense_sum * gl.exp(dense_max - new_max) + gl.sum(gl.exp(logits - new_max), 0)
                dense_max = new_max
            visual_qk_rows += gl.sum(gl.convert_layout(valid, standard).to(gl.int64), 0)
        log_z_visual = dense_max + gl.log(dense_sum)
        zmax = gl.maximum(log_z_text, log_z_visual)
        log_z_dense = zmax + gl.log(gl.exp(log_z_text - zmax) + gl.exp(log_z_visual - zmax))
    gl.store(RouteFlags + head, dense_now.to(gl.int32))
    gl.store(RouteFlags + H + head, proxy_checked.to(gl.int32))
    gl.store(RouteFlags + 2 * H + head, checkpoint.to(gl.int32))
    gl.store(RouteFloat + head, proxy_mass)
    gl.store(RouteFloat + H + head, log_z_text)
    gl.store(RouteFloat + 2 * H + head, log_z_dense)
    gl.store(RouteWork + head, visual_qk_rows)
    gl.store(RouteWork + H + head, text_qk_rows)
