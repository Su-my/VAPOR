# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Cache only the three method kernels, with explicit tensor/scalar ABI guards."""
import torch

class MethodLaunchCache:

    def __init__(self, function):
        self.function = function
        self.cache = {}

    def _abi(self, x):
        if isinstance(x, torch.Tensor):
            return ('tensor', str(x.dtype), x.device.index, x.data_ptr() % 16)
        if isinstance(x, (list, tuple)):
            return tuple((self._abi(v) for v in x))
        if isinstance(x, bool):
            return ('bool', x)
        if isinstance(x, int):
            assert -2 ** 31 <= x < 2 ** 31
            return 'i32'
        if isinstance(x, float):
            return 'fp32'
        raise TypeError(type(x))

    def __getitem__(self, grid):

        def launch(*args, **options):
            key = (tuple(sorted(options.items())), tuple((self._abi(x) for x in args)))
            entry = self.cache.get(key)
            if entry is None:
                kernel = self.function.warmup(*args, grid=grid, **options)
                kernel._init_handles()
                kwargs = {n: v for n, v in options.items() if n in self.function.arg_names}
                bound = self.function.signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = tuple(bound.arguments.values())
                assert values[:len(args)] == args
                entry = (kernel, values[len(args):])
                self.cache[key] = entry
            kernel, constants = entry
            stream = torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
            g = tuple(grid) + (1,) * (3 - len(grid))
            kernel.run(*g, stream, kernel.function, kernel.packed_metadata, None, None, None, *args, *constants)
            return kernel
        return launch
