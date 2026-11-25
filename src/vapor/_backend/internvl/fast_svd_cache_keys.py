# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""One integer-only GPU comparison of all visual K used by the SVD cache."""
import triton
import triton.language as tl
from .svd_cache_keys import exact_visual_keys_match as reference_match

@triton.jit(do_not_specialize=['n_visual'])
def compare_visual_words(Table, OldPositions, NewPositions, Mismatch, n_visual, D: tl.constexpr, BLOCK: tl.constexpr):
    layer = tl.program_id(0)
    head = tl.program_id(1)
    index = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    valid = index < n_visual * D
    visual = index // D
    dim = index % D
    old_position = tl.load(OldPositions + visual, mask=valid, other=0)
    new_position = tl.load(NewPositions + visual, mask=valid, other=0)
    old_pointer = tl.load(Table + layer * 6).to(tl.pointer_type(tl.uint16))
    new_pointer = tl.load(Table + layer * 6 + 3).to(tl.pointer_type(tl.uint16))
    old_head_stride = tl.load(Table + layer * 6 + 1)
    old_row_stride = tl.load(Table + layer * 6 + 2)
    new_head_stride = tl.load(Table + layer * 6 + 4)
    new_row_stride = tl.load(Table + layer * 6 + 5)
    a = tl.load(old_pointer + head * old_head_stride + old_position * old_row_stride + dim, mask=valid, other=0)
    b = tl.load(new_pointer + head * new_head_stride + new_position * new_row_stride + dim, mask=valid, other=0)
    differs = tl.sum(((a != b) & valid).to(tl.int32), axis=0) > 0
    if differs:
        tl.atomic_or(Mismatch, 1)

def exact_visual_keys_match(torch, entry, current_keys, positions):
    if not positions.numel() or positions.numel() != entry['positions'].numel() or set(current_keys) != set(entry['keys']):
        return False
    layers = sorted(current_keys)
    if not layers:
        return False
    first = current_keys[layers[0]]
    for layer in layers:
        old, new = (entry['keys'][layer], current_keys[layer])
        if old.dtype != new.dtype or old.device != new.device or old.shape[1] != new.shape[1] or (old.shape[-1] != new.shape[-1]):
            return False
        if old.device.type != 'cuda' or old.dtype != torch.bfloat16 or old.ndim != 4 or (new.ndim != 4) or (old.shape[0] != 1) or (new.shape[0] != 1) or (old.stride(-1) != 1) or (new.stride(-1) != 1) or (new.device != first.device) or (new.shape[1] != first.shape[1]) or (new.shape[-1] != first.shape[-1]):
            return reference_match(torch, entry, current_keys, positions)
    if positions.device != first.device or entry['positions'].device != first.device:
        return reference_match(torch, entry, current_keys, positions)
    rows = []
    for layer in layers:
        old, new = (entry['keys'][layer], current_keys[layer])
        rows.append([old.data_ptr(), old.stride(1), old.stride(2), new.data_ptr(), new.stride(1), new.stride(2)])
    table = torch.tensor(rows, dtype=torch.int64, device=first.device)
    mismatch = torch.zeros((), dtype=torch.int32, device=first.device)
    count = positions.numel()
    width = first.shape[-1]
    compare_visual_words[len(layers), first.shape[1], triton.cdiv(count * width, 256)](table, entry['positions'], positions, mismatch, count, width, 256, num_warps=4)
    return mismatch.item() == 0
