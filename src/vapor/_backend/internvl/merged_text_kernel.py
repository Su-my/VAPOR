# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['n_keys', 'k_stride', 'scratch_stride', 'partition_stride'])
def merged_text_logits(Q, K, Mask, VisualFlags, Scratch, RoundingFlags, PartitionMax, PartitionSum, PartitionRows, n_keys, k_stride, scratch_stride, partition_stride, scale, D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr=64):
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
        group_query = tl.load(Q + heads[:, None] * D + dim[None, :], mask=head_valid[:, None], other=0)
        group_key = tl.load(K + kv_head * k_stride + seq[None, :] * D + dim[:, None], mask=valid[None, :], other=0)
        group_dots = tl.dot(group_query, group_key, out_dtype=tl.float32)
        group_absolute = tl.dot(tl.abs(group_query), tl.abs(group_key), out_dtype=tl.float32)
        group_bound = group_absolute * (2.0 * D * 5.960464477539063e-08)
        group_uncertain = (group_dots - group_bound).to(tl.bfloat16) != (group_dots + group_bound).to(tl.bfloat16)
        group_logits = (group_dots.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        if HAS_MASK:
            group_bias = tl.load(Mask + seq, mask=valid, other=0).to(tl.float32)
            group_logits = (group_logits + group_bias[None, :]).to(tl.bfloat16).to(tl.float32)
            group_uncertain = group_uncertain & (group_bias[None, :] > float('-inf'))
        group_pointers = heads[:, None] * scratch_stride + seq[None, :]
        tl.store(Scratch + group_pointers, group_logits, mask=head_valid[:, None] & valid[None, :])
        tl.store(RoundingFlags + group_pointers, group_uncertain, mask=head_valid[:, None] & valid[None, :])
    tl.debug_barrier()
    for local_head in tl.static_range(GROUPS):
        head = kv_head * GROUPS + local_head
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
