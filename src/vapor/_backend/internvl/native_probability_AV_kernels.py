# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['n_keys', 'v_stride', 'scratch_stride', 'partition_stride'])
def native_probability_attention_values(V, VisualFlags, Scratch, RouteFlags, RouteFloat, PartialReadout, PartialVisualReadout, PartialMass, PartialVisualRows, n_keys, v_stride, scratch_stride, partition_stride, NativeP, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr=64):
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
        if dense_now:
            probabilities = tl.load(NativeP + head * scratch_stride + seq, mask=valid, other=0.0).to(tl.float32)
        else:
            probabilities = tl.exp(logits - maximum) / normalizer
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
