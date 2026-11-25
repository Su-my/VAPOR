# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Same rank16 Gram eigendecomposition on CPU; all IO and completion are timed."""
import torch
EIGEN_DEVICE = 'GPU'
from .lean_stats_gate import VisualDecomposition
from .lazy_batched_hold_gate import LazyPackedBatchedHoldGate
from .pinned_stream_prepared_hold_gate import resources

def prepare_one_CPU(keys, stream, ready, positions, rank):
    with torch.cuda.device(keys.device), torch.cuda.stream(stream), torch.inference_mode():
        stream.wait_event(ready)
        keys.record_stream(stream)
        visual = keys[0, :, positions, :].float()
        gram = visual.transpose(-1, -2) @ visual
        assert EIGEN_DEVICE in ('CPU', 'GPU')
        eigen, basis = torch.linalg.eigh(gram if EIGEN_DEVICE == 'GPU' else gram.cpu())
        eigen = eigen.to(keys.device)
        basis = basis.to(keys.device)
        eigen = eigen.flip(-1).clamp_min(0.0)
        effective = min(rank, visual.shape[-2], visual.shape[-1])
        basis = basis.flip(-1)[..., :effective].contiguous()
        low = (visual @ basis).contiguous()
        padded = 1 << (effective - 1).bit_length()
        if padded != effective:
            basis = torch.nn.functional.pad(basis, (0, padded - effective)).contiguous()
            low = torch.nn.functional.pad(low, (0, padded - effective)).contiguous()
        energy = eigen[..., :effective].sum(-1) / eigen.sum(-1).clamp_min(1e-20)
        end = torch.cuda.Event()
        end.record(stream)
        return (VisualDecomposition(basis=basis, low_keys=low, explained_energy=energy, effective_rank=effective), end)

class CPUEigenMixin:

    def _start_decompositions(self, layer, keys):
        if self._prefill_pending.futures:
            return
        snapshot = {i: self.decode_cache.layers[i].keys for i in self.layers} if self.decode_cache is not None else {layer: keys}
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        bank = resources(keys.device)
        for i, value in snapshot.items():
            executor, stream = bank[self.layer_offset[i] % len(bank)]
            self._prefill_pending.enqueue(i, value, lambda k, _e=executor, _s=stream: _e.submit(prepare_one_CPU, k, _s, ready, self.visual_positions, self.rank))

    def _prepare_decomposition(self, layer, keys):
        if self.decode_tokens + 1 < 3:
            if self.max_new_tokens > 3:
                self._start_decompositions(layer, keys)
            return LazyPackedBatchedHoldGate._prepare_decomposition(self, layer, keys)
        if layer in self.decompositions:
            return self.decompositions[layer]
        self._start_decompositions(layer, keys)
        decomp, event = self._prefill_pending.futures[layer].result()
        torch.cuda.current_stream().wait_event(event)
        self.raw_energy[self.layer_offset[layer]] = decomp.explained_energy.mean()
        self.decompositions[layer] = decomp
        return decomp

    def finish_async_preparation(self):
        self._prefill_pending.join(torch.cuda.current_stream().wait_event)
        for layer, future in self._prefill_pending.futures.items():
            self.decompositions.setdefault(layer, future.result()[0])

    def statistics(self):
        r = super().statistics()
        r.update(method_eigensolver=EIGEN_DEVICE + ' float32 eigh on identical GPU Gram; transfers and all preparation in measured request', method_eigensolver_device=EIGEN_DEVICE, no_inter_sample_SVD_reuse=True, independent_accuracy_required=True)
        return r
_CLASSES = {}

def cpu_eigen_gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('CPUEigen' + base.__name__, (CPUEigenMixin, base), {})
    return _CLASSES[base]
