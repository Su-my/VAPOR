# SPDX-License-Identifier: Apache-2.0
"""Small tensor references for inspecting the math, not a generation backend."""
import math

def visual_factors(keys, rank):
    """keys: [..., visual_tokens, head_dim]; return U_r Sigma_r and V_r."""
    import torch
    if rank < 1: raise ValueError("rank must be positive")
    keys = keys.float()
    _, basis = torch.linalg.eigh(keys.transpose(-1, -2) @ keys)
    r = min(rank, keys.shape[-2], keys.shape[-1])
    basis = basis.flip(-1)[..., :r].contiguous()
    return keys @ basis, basis

def visual_mass(visual_logits, text_logits):
    import torch
    return torch.sigmoid(torch.logsumexp(visual_logits.float(), -1)
                         - torch.logsumexp(text_logits.float(), -1))

def proxy_mass(query, low_keys, basis, text_logits):
    return visual_mass((low_keys @ (basis.transpose(-1, -2) @ query.unsqueeze(-1))).squeeze(-1)
                       / math.sqrt(query.shape[-1]), text_logits)

def age_threshold(base, age, horizon):
    """Existing kernel convention: age starts at token - anchor_token."""
    if horizon < 3: raise ValueError("horizon must be >= 3")
    phase = min(1.0, max(0.0, (age - 2) / (horizon - 2)))
    return base * math.cos(.5 * math.pi * phase)
