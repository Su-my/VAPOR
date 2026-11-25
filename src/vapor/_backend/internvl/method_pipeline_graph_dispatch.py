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
        source = Path(__file__).with_name('method_pipeline_graph_launcher.cpp')
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

def descriptions(launches, grids):
    import re
    assert len(launches) == len(grids) and len(launches) in (8, 9)
    result = []
    for index, (launch, grid) in enumerate(zip(launches, grids)):
        metadata = launch.kernel.metadata
        assert metadata.num_ctas == 1 and tuple(metadata.cluster_dims) == (1, 1, 1)
        assert not (metadata.launch_cooperative_grid or metadata.launch_pdl or metadata.global_scratch_size)
        names = launch.kernel.src.fn.arg_names
        signature = launch.kernel.src.signature
        args = launch.arguments + list(launch.constants)
        assert len(args) == len(names)
        dynamic = dict(launch.dynamic)
        types = []
        values = []
        changes = []
        patches = []
        for i, (name, value) in enumerate(zip(names, args)):
            start = len(values)
            flatten(signature[name], value, dynamic.get(i), types, values, changes)
            if name == 'State':
                assert len(values) - start == 6
                patches.extend(((start + j, j) for j in range(6)))
            elif name == 'layer':
                assert len(values) - start == 1
                patches.append((start, 6))
        ptx = launch.kernel.asm['ptx']
        entry = re.search('\\.entry\\s+\\w+\\s*\\((.*?)\\)', ptx, re.S)
        assert entry
        assert len(re.findall('\\.param\\b', entry.group(1))) == len(values) + 1
        divisor = 64 if index == 0 else 32 if index == 6 else -grid[1] if len(grid) > 1 else 0
        result.append((launch.kernel.function, metadata.num_warps, metadata.shared, grid[0], divisor, ''.join(types), values, changes, patches))
    return result

class MethodBatch:

    def __init__(self, launches, grids):
        self.launches = launches
        self.ext = extension()
        self.handle = self.ext.create(descriptions(launches, grids))

    def launch_values(self, values):
        import torch
        self.ext.run(self.handle, values, torch._C._cuda_getCurrentRawStream(0))

    def clone(self, state, layer):
        result = object.__new__(MethodBatch)
        result.launches = self.launches
        result.ext = self.ext
        result.handle = self.ext.clone(self.handle, tuple((v.data_ptr() for v in state)) + (layer,))
        return result
