# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Exact byte comparison of the visual keys used by a HOLD decomposition."""

def exact_visual_keys_match(torch, entry, current_keys, positions):
    if not positions.numel() or positions.numel() != entry['positions'].numel():
        return False
    if set(current_keys) != set(entry['keys']):
        return False
    comparisons = []
    for layer, current in current_keys.items():
        old = entry['keys'][layer]
        if old.dtype != current.dtype or old.device != current.device or old.shape[1] != current.shape[1] or (old.shape[-1] != current.shape[-1]):
            return False
        a = old[0, :, entry['positions'], :].contiguous().view(torch.uint8)
        b = current[0, :, positions, :].contiguous().view(torch.uint8)
        comparisons.append((a == b).all())
    return bool(torch.stack(comparisons).all().item())

def unique_storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[str(tensor.device), storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())
