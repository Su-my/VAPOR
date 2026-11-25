# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""One host call for the five unchanged HOLD kernels; CUDA math is untouched."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
DYNAMIC = ('Q', 'K', 'V', 'Mask', 'Basis', 'LowKeys', 'n_keys', 'k_stride', 'v_stride', 'token')
KINDS = {'i1': 'i', 'i8': 'b', 'i16': 's', 'i32': 'i', 'i64': 'L', 'u1': 'u', 'u8': 'B', 'u16': 'S', 'u32': 'u', 'u64': 'U', 'fp32': 'f', 'f32': 'f', 'fp64': 'd'}
_EXTENSION = None

def extension():
    global _EXTENSION
    if _EXTENSION is None:
        from triton.backends.nvidia.driver import include_dirs
        source = Path(__file__).with_name('hold_batch_launcher.cpp')
        key = hashlib.sha256(source.read_bytes() + sys.implementation.cache_tag.encode()).hexdigest()[:20]
        directory = Path('/tmp') / ('phase_hold_batch_' + key)
        directory.mkdir(exist_ok=True)
        binary = directory / ('_phase_hold_batch' + sysconfig.get_config_var('EXT_SUFFIX'))
        with (directory / 'build.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not binary.exists():
                compiler = shutil.which('gcc')
                assert compiler, 'A C++-capable compiler is required'
                temporary = binary.with_name(binary.name + '.tmp')
                include = Path(sysconfig.get_path('include'))
                if not (include / 'Python.h').exists():
                    include = Path(__file__).parent / 'python312_headers/python3.12'
                if not (include / 'Python.h').exists():
                    raise RuntimeError('Python 3.12 development headers are required')
                subprocess.run([compiler, '-x', 'c++', '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-I' + str(include), *['-I' + p for p in include_dirs], '-ldl', '-lstdc++', '-o', str(temporary)], check=True)
                temporary.replace(binary)
        spec = importlib.util.spec_from_file_location('_phase_hold_batch', binary)
        _EXTENSION = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXTENSION)
    return _EXTENSION

def flatten(signature, value, dynamic, types, values, changes):
    if isinstance(signature, tuple):
        if not isinstance(value, tuple) or len(value) != len(signature):
            raise ValueError('Kernel tuple layout differs')
        if dynamic is not None:
            raise ValueError('Dynamic tuple arguments are not supported')
        for s, v in zip(signature, value):
            flatten(s, v, None, types, values, changes)
    elif signature != 'constexpr':
        kind = 'p' if signature.startswith('*') else KINDS.get(signature)
        if kind is None:
            raise ValueError('Unsupported HOLD kernel argument: ' + signature)
        if dynamic is not None:
            changes.append((len(values), DYNAMIC.index(dynamic)))
        types.append(kind)
        values.append(value)

def descriptions(launches, groups, n_visual):
    if len(launches) != 5:
        raise ValueError('Four HOLD kernels required')
    result = []
    for index, launch in enumerate(launches):
        metadata = launch.kernel.metadata
        if metadata.num_ctas != 1 or tuple(metadata.cluster_dims) != (1, 1, 1) or metadata.launch_cooperative_grid or metadata.launch_pdl or metadata.global_scratch_size:
            raise ValueError('Unsupported CUDA launch attributes')
        names = launch.kernel.src.fn.arg_names
        signature = launch.kernel.src.signature
        args = launch.arguments + list(launch.constants)
        if len(args) != len(names):
            raise ValueError('HOLD kernel bound argument count differs')
        dynamic = dict(launch.dynamic)
        types = []
        values = []
        changes = []
        patches = []
        for i, (name, value) in enumerate(zip(names, args)):
            start = len(values)
            flatten(signature[name], value, dynamic.get(i), types, values, changes)
            if name == 'State':
                if len(values) - start != 6:
                    raise ValueError('Unexpected HOLD state tuple')
                patches.extend(((start + j, j) for j in range(6)))
            elif name == 'layer':
                if len(values) - start != 1:
                    raise ValueError('Layer must remain a runtime argument')
                patches.append((start, 6))
        if len(patches) != (6 if index == 1 else 7 if index in (2, 4) else 0):
            raise ValueError('HOLD layer/state patch layout differs')
        result.append((launch.kernel.function, metadata.num_warps, metadata.shared, 32 // groups if index == 0 else 32 * ((n_visual + 63) // 64) if index == 1 else 32, 64 if index == 0 else 32 if index == 3 else 0, ''.join(types), values, changes, patches))
    return result

class HoldBatch:

    def __init__(self, launches, groups, n_visual):
        self.launches = launches
        self.ext = extension()
        self.handle = self.ext.create(descriptions(launches, groups, n_visual))

    def launch_values(self, values):
        import torch
        self.ext.run(self.handle, values, torch._C._cuda_getCurrentRawStream(0))

    def clone(self, state_pointers, layer):
        result = object.__new__(HoldBatch)
        result.launches = self.launches
        result.ext = self.ext
        result.handle = self.ext.clone(self.handle, tuple((p.data_ptr() for p in state_pointers)) + (layer,))
        return result
