# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Allocate only HOLD episode state in one tensor per field, with disjoint layers."""
from types import SimpleNamespace
import torch

class PackedHoldState:

    def __init__(self, layers, heads, dim, m3, confirmation, device):
        shape = (layers, heads)
        self.buffers = dict(holding=torch.zeros(shape, dtype=torch.bool, device=device), anchor_ready=torch.zeros(shape, dtype=torch.bool, device=device), anchor_token=torch.full(shape, -1, dtype=torch.int16, device=device), anchor_readout=torch.zeros((*shape, dim), dtype=torch.float32, device=device), anchor_nonvisual_tokens=torch.zeros(shape, dtype=torch.int32, device=device), episode_m3_base=torch.full(shape, m3, dtype=torch.float32, device=device), episode_confirmation_threshold=torch.full(shape, confirmation, dtype=torch.float32, device=device))

    def row(self, offset):
        return SimpleNamespace(**{k: v[offset] for k, v in self.buffers.items()})
