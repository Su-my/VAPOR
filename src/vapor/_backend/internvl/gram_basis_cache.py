# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Reuse checked eigenvectors only when the FP32 Gram bytes match exactly."""

def gram_bytes_match(torch, left, right):
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return False
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))

def decompose_visual(torch, visual, rank, cached=None):
    gram = visual.transpose(-1, -2) @ visual
    effective = min(rank, visual.shape[-2], visual.shape[-1])
    hit = cached is not None and cached['effective_rank'] == effective and gram_bytes_match(torch, cached['gram'], gram)
    if hit:
        core_basis = cached['core_basis']
        basis = cached['basis']
        energy = cached['energy']
    else:
        eigenvalues, vectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
        core_basis = vectors.flip(-1)[..., :effective].contiguous()
        basis = core_basis
    low_keys = (visual @ core_basis).contiguous()
    padded = 1 << (effective - 1).bit_length()
    if padded != effective:
        if not hit:
            basis = torch.nn.functional.pad(core_basis, (0, padded - effective)).contiguous()
        low_keys = torch.nn.functional.pad(low_keys, (0, padded - effective)).contiguous()
    if not hit:
        energy = eigenvalues[..., :effective].sum(-1) / eigenvalues.sum(-1).clamp_min(1e-20)
    entry = cached if hit else dict(gram=gram, core_basis=core_basis, basis=basis, energy=energy, effective_rank=effective)
    return (basis, low_keys, energy, effective, entry, bool(hit))
