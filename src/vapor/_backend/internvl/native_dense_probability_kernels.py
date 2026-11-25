# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def block_sum_native(x):
    warps = tl.reshape(x, (32, 32))
    return tl.sum(tl.sum(warps, axis=1), axis=0)

@triton.jit(do_not_specialize=['n_keys', 'scratch_stride', 'layer'])
def persistent(X, Routes, P, HeadWork, KeyWork, n_keys, scratch_stride, layer, H: tl.constexpr, ITER: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(Routes + row) != 0:
        lane = tl.arange(0, 32)
        maximum = tl.full((32,), float('-inf'), tl.float32)
        for j in tl.static_range(ITER):
            i = lane + j * 32
            value = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
            maximum = tl.maximum(maximum, value)
        maximum = tl.max(maximum, axis=0)
        total = tl.full((32,), 0.0, tl.float32)
        for j in tl.static_range(ITER):
            i = lane + j * 32
            value = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
            total = total + libdevice.exp(value - maximum)
        total = tl.sum(total, axis=0)
        for j in tl.static_range(ITER):
            i = lane + j * 32
            value = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
            p = tl.div_rn(libdevice.exp(value - maximum), total)
            tl.store(P + row * scratch_stride + i, p, i < n_keys)
        index = layer * H + row
        tl.store(HeadWork + index, tl.load(HeadWork + index) + 1)
        tl.store(KeyWork + index, tl.load(KeyWork + index) + n_keys)

@triton.jit(do_not_specialize=['n_keys', 'scratch_stride', 'layer'])
def register_block(X, Routes, P, HeadWork, KeyWork, n_keys, scratch_stride, layer, H: tl.constexpr, REG: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(Routes + row) != 0:
        lane = tl.arange(0, 1024)
        maximum = tl.full((1024,), -3.4028234663852886e+38, tl.float32)
        for j in tl.static_range(REG):
            i = lane + j * 1024
            value = tl.load(X + row * scratch_stride + i, i < n_keys, -3.4028234663852886e+38).to(tl.float32)
            maximum = tl.maximum(maximum, value)
        maximum = tl.max(maximum, axis=0)
        total = tl.full((1024,), 0.0, tl.float32)
        for j in tl.static_range(REG):
            i = lane + j * 1024
            value = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
            total = total + libdevice.exp(value - maximum)
        total = block_sum_native(total)
        for j in tl.static_range(REG):
            i = lane + j * 1024
            value = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
            p = tl.div_rn(libdevice.exp(value - maximum), total)
            tl.store(P + row * scratch_stride + i, p, i < n_keys)
        index = layer * H + row
        tl.store(HeadWork + index, tl.load(HeadWork + index) + 1)
        tl.store(KeyWork + index, tl.load(KeyWork + index) + n_keys)

@triton.jit(do_not_specialize=['n_keys', 'scratch_stride', 'layer'])
def ilp4_block(X, Routes, P, HeadWork, KeyWork, n_keys, scratch_stride, layer, H: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(Routes + row) != 0:
        lane = tl.arange(0, 1024)
        heads = tl.arange(0, H)
        dense_ordinal = tl.sum(((heads < row) & (tl.load(Routes + heads) != 0)).to(tl.int32), axis=0)
        shift = dense_ordinal * n_keys % 4
        base = tl.full((), 0, tl.int32)
        size = n_keys
        maximum = tl.full((1024,), -3.4028234663852886e+38, tl.float32)
        if shift > 0:
            i = lane - shift
            v = tl.load(X + row * scratch_stride + i, (lane >= shift) & (i < n_keys), -3.4028234663852886e+38).to(tl.float32)
            maximum = tl.maximum(maximum, v)
            base = 1024 - shift
            size = n_keys + shift - 1024
        last = tl.where((n_keys % 4 == 0) & (n_keys < 12256), 0, size % 4096)
        for start in range(0, size - last, 4096):
            for j in tl.static_range(4):
                i = base + start + lane * 4 + j
                v = tl.load(X + row * scratch_stride + i, i < n_keys, -3.4028234663852886e+38).to(tl.float32)
                maximum = tl.maximum(maximum, v)
        for j in tl.static_range(4):
            i = base + size - last + lane + j * 1024
            v = tl.load(X + row * scratch_stride + i, i < n_keys, -3.4028234663852886e+38).to(tl.float32)
            maximum = tl.maximum(maximum, v)
        maximum = tl.max(maximum, axis=0)
        total = tl.full((1024,), 0.0, tl.float32)
        if shift > 0:
            i = lane - shift
            v = tl.load(X + row * scratch_stride + i, (lane >= shift) & (i < n_keys), float('-inf')).to(tl.float32)
            total = total + libdevice.exp(v - maximum)
        for start in range(0, size - last, 4096):
            for j in tl.static_range(4):
                i = base + start + lane * 4 + j
                v = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
                total = total + libdevice.exp(v - maximum)
        for j in tl.static_range(4):
            i = base + size - last + lane + j * 1024
            v = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
            total = total + libdevice.exp(v - maximum)
        total = block_sum_native(total)
        for start in range(0, n_keys, 1024):
            i = start + lane
            v = tl.load(X + row * scratch_stride + i, i < n_keys, float('-inf')).to(tl.float32)
            p = tl.div_rn(libdevice.exp(v - maximum), total)
            tl.store(P + row * scratch_stride + i, p, i < n_keys)
        index = layer * H + row
        tl.store(HeadWork + index, tl.load(HeadWork + index) + 1)
        tl.store(KeyWork + index, tl.load(KeyWork + index) + n_keys)
