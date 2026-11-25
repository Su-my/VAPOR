# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Update state and aggregate counters; attention/output remain native ATen."""
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['token', 'layer', 'n_keys', 'n_visual', 'horizon'])
def finish_state(State, Counters, FloatCounters, Work, Proxy, Mass, Visual, Stay, Checked, token, layer, n_keys, n_visual, m1, horizon, H: tl.constexpr, D: tl.constexpr):
    head = tl.program_id(0)
    row = layer * H + head
    dim = tl.arange(0, D)
    holding = tl.load(State[0] + head)
    ready = tl.load(State[1] + head)
    anchor = tl.load(State[2] + head).to(tl.int32)
    age = tl.where(anchor >= 0, token - anchor, -1)
    stay = tl.load(Stay + head)
    dense = ~stay
    checked = tl.load(Checked + head)
    proxy = tl.load(Proxy + head).to(tl.float32)
    mass = tl.load(Mass + head).to(tl.float32)
    confirmation = holding & dense
    passed = confirmation & (mass >= m1)
    failed = confirmation & ~passed
    enter = ~holding & ready & (mass < m1)
    checkpoint = holding & ready & (age >= horizon)
    hc = confirmation & checkpoint
    pc = confirmation & checked
    after = stay | dense & ready & (mass < m1)
    if dense:
        visual = tl.load(Visual + head * D + dim)
        tl.store(State[3] + head * D + dim, visual)
        tl.store(State[2] + head, token)
        tl.store(State[4] + head, n_keys - n_visual)
    tl.store(State[1] + head, ready | dense)
    tl.store(State[0] + head, after)
    zero = tl.full((), False, tl.int1)
    changes = (dense, stay, enter, passed, checked, confirmation, failed, passed, passed & (mass < m1), pc, pc & failed, pc & passed, hc, hc & failed, hc & passed, hc & failed, holding, hc, zero, zero, zero, pc)
    for i in tl.static_range(22):
        before = tl.load(Counters[i] + row)
        tl.store(Counters[i] + row, before + changes[i].to(tl.int64))
    if pc:
        error = tl.abs(proxy - mass).to(tl.float64)
        tl.store(FloatCounters[0] + row, tl.load(FloatCounters[0] + row) + error)
    if token == 1:
        tl.store(FloatCounters[2] + row, mass.to(tl.float64))
    visual_rows = dense.to(tl.int64) * n_visual
    tl.store(Work[0] + row, tl.load(Work[0] + row) + visual_rows)
    tl.store(Work[1] + row, tl.load(Work[1] + row) + visual_rows)
    text_rows = (dense.to(tl.int64) + checked.to(tl.int64)) * (n_keys - n_visual)
    tl.store(Work[2] + row, tl.load(Work[2] + row) + text_rows)
from .scoped_method_launch_cache import MethodLaunchCache
finish_state = MethodLaunchCache(finish_state)
