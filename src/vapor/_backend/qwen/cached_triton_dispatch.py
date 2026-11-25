# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Reuse a compiled Triton launcher under the frozen attention ABI."""
import torch

class CachedDispatch:

    def __init__(self, function):
        self.function = function
        self.cache = {}
        self.mask_position = function.arg_names.index('Mask') if 'Mask' in function.arg_names else None

    def __getitem__(self, grid):

        def launch(*args, **options):
            mask_type = args[self.mask_position].dtype if self.mask_position is not None else None
            key = (tuple(sorted(options.items())), mask_type)
            entry = self.cache.get(key)
            if entry is None:
                kernel = self.function.warmup(*args, grid=grid, **options)
                kernel._init_handles()
                kwargs = {name: value for name, value in options.items() if name in self.function.arg_names}
                bound = self.function.signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = tuple(bound.arguments.values())
                assert values[:len(args)] == args
                entry = (kernel, values[len(args):])
                self.cache[key] = entry
            kernel, constants = entry
            stream = torch._C._cuda_getCurrentRawStream(0)
            grid3 = tuple(grid) + (1,) * (3 - len(grid))
            kernel.run(grid3[0], grid3[1], grid3[2], stream, kernel.function, kernel.packed_metadata, None, None, None, *args, *constants)
            return kernel
        return launch
