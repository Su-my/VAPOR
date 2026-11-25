# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Choose a method-only SVD cache by its conservative retained byte budget."""

def visual_cache_upper_bytes(prompt_tokens, visual_tokens, layers, kv_heads, head_dim, rank):
    if not (0 < visual_tokens <= prompt_tokens and min(layers, kv_heads, head_dim, rank) > 0):
        raise ValueError('Invalid admitted prompt/cache dimensions')
    effective = min(rank, visual_tokens, head_dim)
    padded = 1 << (effective - 1).bit_length()
    keys = layers * kv_heads * (prompt_tokens + 1) * head_dim * 2
    basis = layers * kv_heads * head_dim * padded * 4
    low_keys = layers * kv_heads * visual_tokens * padded * 4
    energy = layers * kv_heads * 4
    positions = visual_tokens * 8
    return keys + basis + low_keys + energy + positions

def cache_family(prompt_tokens, visual_tokens, layers=36, kv_heads=8, head_dim=128, rank=16, max_bytes=128 * 1024 * 1024):
    if max_bytes <= 0:
        raise ValueError('Cache budget must be positive')
    return 'visual_keys' if visual_cache_upper_bytes(prompt_tokens, visual_tokens, layers, kv_heads, head_dim, rank) <= max_bytes else 'gram_basis'
