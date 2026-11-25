# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Parallelize only logits; the route retains the original ordered partitions."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit(do_not_specialize=['n_visual', 'low_stride', 'token'])
def proxy_logits(Q, Mask, VisualPositions, Basis, LowKeys, Output, State, RouteFloat, n_visual, low_stride, token, scale, horizon, m3, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, RANK: tl.constexpr, HAS_MASK: tl.constexpr, USE_HORIZON: tl.constexpr=True, BLOCK: tl.constexpr=64):
    count = tl.cdiv(n_visual, BLOCK)
    head = tl.program_id(0) // count
    block = tl.program_id(0) % count
    holding = tl.load(State[0] + head)
    ready = tl.load(State[1] + head)
    anchor_token = tl.load(State[2] + head).to(tl.int32)
    age = token - anchor_token
    if block == 0:
        effective_age = tl.where(anchor_token >= 0, age, -1)
        phase = tl.minimum(tl.maximum((effective_age - 2).to(tl.float32), 0.0) / (horizon - 2), 1.0)
        threshold = m3 * libdevice.cos(1.5707963267948966 * phase)
        threshold = tl.where(effective_age >= 0, threshold, 0.0)
        tl.store(RouteFloat + 3 * H + head, threshold)
    checked = holding & ready & ~((age >= horizon) & USE_HORIZON)
    if checked:
        dim = tl.arange(0, D)
        rank_lane = tl.arange(0, RANK)
        query = tl.load(Q + head * D + dim).to(tl.float32)
        kv_head = head // GROUPS
        basis = tl.load(Basis + kv_head * D * RANK + dim[:, None] * RANK + rank_lane[None, :])
        low_query = tl.sum(query[:, None] * basis, axis=0)
        index = block * BLOCK + tl.arange(0, BLOCK)
        valid = index < n_visual
        low = tl.load(LowKeys + kv_head * low_stride + index[:, None] * RANK + rank_lane[None, :], mask=valid[:, None], other=0.0)
        logits = tl.sum(low * low_query[None, :], axis=1) * scale
        if HAS_MASK:
            seq = tl.load(VisualPositions + index, mask=valid, other=0)
            logits += tl.load(Mask + seq, mask=valid, other=0.0).to(tl.float32)
        tl.store(Output + head * n_visual + index, tl.where(valid, logits, float('-inf')), mask=valid)
