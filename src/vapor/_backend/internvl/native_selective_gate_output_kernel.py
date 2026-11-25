# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['token', 'horizon'])
def select_gate(Holding, Ready, Anchor, Lut, Threshold, Checked, token, horizon, H: tl.constexpr, B: tl.constexpr):
    h = tl.arange(0, B)
    valid = h < H
    holding = tl.load(Holding + h, valid, False)
    ready = tl.load(Ready + h, valid, False)
    anchor = tl.load(Anchor + h, valid, -1).to(tl.int32)
    distance = tl.where(anchor >= 0, token - anchor, -1)
    age = tl.minimum(tl.maximum(distance, 0), horizon)
    threshold = tl.load(Lut + age, valid, 0.0)
    threshold = tl.where(distance >= 0, threshold, 0.0)
    checked = holding & ready & (distance < horizon)
    tl.store(Threshold + h, threshold, valid)
    tl.store(Checked + h, checked, valid)

@triton.jit
def combine_output(Dense, Proxy, Text, Anchor, Stay, Output, D: tl.constexpr):
    h = tl.program_id(0)
    d = tl.arange(0, D)
    stay = tl.load(Stay + h)
    proxy = tl.load(Proxy + h)
    text = tl.load(Text + h * D + d)
    anchor = tl.load(Anchor + h * D + d)
    complement = 1.0 - proxy
    left = complement * text
    right = proxy * anchor
    held = left + right
    dense = tl.load(Dense + h * D + d).to(tl.float32)
    tl.store(Output + h * D + d, tl.where(stay, held, dense))
from .scoped_method_launch_cache import MethodLaunchCache
select_gate = MethodLaunchCache(select_gate)
combine_output = MethodLaunchCache(combine_output)
