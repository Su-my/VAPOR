# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Prepare HOLD launch arguments once per layer/sample; no math changes."""
import torch

class PreparedLaunch:
    DYNAMIC = {'Q', 'K', 'V', 'Mask', 'Basis', 'LowKeys', 'n_keys', 'k_stride', 'v_stride', 'token'}

    def __init__(self, function, kernel, constants, args):
        self.kernel = kernel
        self.constants = constants
        self.references = []

        def convert(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cuda':
                    raise ValueError('Prepared HOLD pointers require CUDA tensors')
                self.references.append(value)
                return value.data_ptr()
            if isinstance(value, tuple):
                return tuple((convert(x) for x in value))
            return value
        self.dynamic = []
        self.arguments = []
        for index, value in enumerate(args):
            name = function.arg_names[index]
            if name in self.DYNAMIC:
                self.dynamic.append((index, name))
                self.arguments.append(0)
            else:
                self.arguments.append(convert(value))

    def launch(self, grid, updates):
        for index, name in self.dynamic:
            self.arguments[index] = updates[name]
        stream = torch._C._cuda_getCurrentRawStream(0)
        grid3 = tuple(grid) + (1,) * (3 - len(grid))
        self.kernel.run(grid3[0], grid3[1], grid3[2], stream, self.kernel.function, self.kernel.packed_metadata, None, None, None, *self.arguments, *self.constants)

class PreparedDispatch:

    def __init__(self, function):
        self.function = function
        self.cache = {}
        self.mask_position = function.arg_names.index('Mask') if 'Mask' in function.arg_names else None

    def prepare(self, args, grid, options):
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
        return PreparedLaunch(self.function, entry[0], entry[1], args)
