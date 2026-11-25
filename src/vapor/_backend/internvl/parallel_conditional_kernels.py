# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
import triton
import triton.language as tl
from .cached_triton_dispatch import CachedDispatch
from .triton_hold_kernel import attention_values as _attention_values, finalize_decode as _finalize
attention_values_all = CachedDispatch(_attention_values)
finalize_all = CachedDispatch(_finalize)

@triton.jit(do_not_specialize=['n_visual', 'k_stride', 'scratch_stride', 'partition_stride'])
def dense_visual_logits(Q, K, Mask, Positions, RouteFlags, Scratch, PartMax, PartSum, n_visual, k_stride, scratch_stride, partition_stride, scale, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    block = tl.program_id(1)
    dense = tl.load(RouteFlags + head) != 0
    maximum = tl.full((), float('-inf'), tl.float32)
    total = tl.full((), 0.0, tl.float32)
    if dense:
        vi = block * BLOCK + tl.arange(0, BLOCK)
        valid = vi < n_visual
        d = tl.arange(0, D)
        seq = tl.load(Positions + vi, valid, 0)
        query = tl.load(Q + head * D + d).to(tl.float64)
        key = tl.load(K + head // GROUPS * k_stride + seq[:, None] * D + d[None, :], valid[:, None], 0).to(tl.float64)
        dot = tl.sum(key * query[None, :], axis=1).to(tl.float32)
        logits = (dot.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        if HAS_MASK:
            bias = tl.load(Mask + seq, valid, 0.0).to(tl.float32)
            logits = (logits + bias).to(tl.bfloat16).to(tl.float32)
        logits = tl.where(valid, logits, float('-inf'))
        tl.store(Scratch + head * scratch_stride + seq, logits, valid)
        maximum = tl.max(logits, axis=0)
        safe = tl.where(maximum > float('-inf'), maximum, 0.0)
        total = tl.sum(tl.exp(logits - safe), axis=0)
    offset = head * partition_stride + block
    tl.store(PartMax + offset, maximum)
    tl.store(PartSum + offset, total)

@triton.jit(do_not_specialize=['n_visual', 'partition_stride'])
def dense_normalizer(RouteFlags, RouteFloat, PartMax, PartSum, n_visual, partition_stride, H: tl.constexpr, PART_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    dense = tl.load(RouteFlags + head) != 0
    if dense:
        p = tl.arange(0, PART_BLOCK)
        valid = p < tl.cdiv(n_visual, BLOCK)
        maxima = tl.load(PartMax + head * partition_stride + p, valid, float('-inf'))
        sums = tl.load(PartSum + head * partition_stride + p, valid, 0.0)
        visual_max = tl.max(maxima, axis=0)
        visual_sum = tl.sum(sums * tl.exp(maxima - visual_max), axis=0)
        text_max = tl.load(RouteFloat + 4 * H + head)
        text_sum = tl.load(RouteFloat + 5 * H + head)
        maximum = tl.maximum(text_max, visual_max)
        total = text_sum * tl.exp(text_max - maximum) + visual_sum * tl.exp(visual_max - maximum)
        tl.store(RouteFloat + 2 * H + head, maximum + tl.log(total))
dense_visual_logits = CachedDispatch(dense_visual_logits)
dense_normalizer = CachedDispatch(dense_normalizer)
