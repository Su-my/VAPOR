# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['n_keys', 'v_stride', 'scratch_stride', 'partition_stride'])
def stable_attention_values(V, VisualFlags, Scratch, RouteFlags, RouteFloat, PartialReadout, PartialVisualReadout, PartialMass, PartialVisualRows, n_keys, v_stride, scratch_stride, partition_stride, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr=64):
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
    maximum = tl.load(RouteFloat + 4 * H + head)
    normalizer = tl.load(RouteFloat + 5 * H + head)
    readout = tl.full((D,), 0.0, tl.float32)
    visual_readout = tl.full((D,), 0.0, tl.float32)
    mass = tl.full((), 0.0, tl.float32)
    visual_rows = tl.full((), 0, tl.int32)
    if tl.sum(valid.to(tl.int32), axis=0) > 0:
        logits = tl.load(Scratch + head * scratch_stride + seq, mask=valid, other=float('-inf')).to(tl.float32)
        probabilities = tl.exp(logits - maximum) / normalizer
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

@triton.jit(do_not_specialize=['n_visual', 'partition_stride'])
def stable_dense_normalizer(RouteFlags, RouteFloat, PartMax, PartSum, n_visual, partition_stride, H: tl.constexpr, PART_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
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
        tl.store(RouteFloat + 4 * H + head, maximum)
        tl.store(RouteFloat + 5 * H + head, total)
