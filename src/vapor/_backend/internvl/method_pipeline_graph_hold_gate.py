# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""One CPU dispatch for eight unchanged method kernels; no generic model work."""
from contextlib import contextmanager
import torch
from .prepared_triton_dispatch import PreparedDispatch
from .method_pipeline_graph_dispatch import MethodBatch
from .parallel_conditional_hold_gate import ParallelConditionalMixin
from . import parallel_conditional_hold_gate as reference_module
from . import parallel_conditional_kernels as auxiliary_module
SHADOW_ACTIVE = False
SHADOW_COUNTS = {'passed': 0}

def set_shadow(enabled):
    global SHADOW_ACTIVE
    SHADOW_ACTIVE = bool(enabled)
_DISPATCH = {}

class Recorder:

    def __init__(self, name, original, entries):
        self.name = name
        self.original = original
        self.entries = entries

    def __getitem__(self, grid):

        def record(*args, **options):
            raw = self.original
            while hasattr(raw, 'function'):
                raw = raw.function
            if raw not in _DISPATCH:
                _DISPATCH[raw] = PreparedDispatch(raw)
            launch = _DISPATCH[raw].prepare(args, grid, options)
            self.entries.append((self.name, launch, tuple(grid)))
        return record

@contextmanager
def record_eight(entries):
    originals = []
    try:
        for module, names in [(reference_module, ['merged_text_logits', 'parallel_low_query', 'parallel_visual_proxy', 'route_decode']), (auxiliary_module, ['dense_visual_logits', 'dense_normalizer', 'attention_values_all', 'finalize_all'])]:
            for name in names:
                old = getattr(module, name)
                originals.append((module, name, old))
                setattr(module, name, Recorder(name, old, entries))
        yield
    finally:
        for module, name, old in originals:
            setattr(module, name, old)

class BatchedPipelineMixin:

    def statistics(self):
        result = super().statistics()
        result.update(execution_backend=self.backend, numerical_implementation='Unchanged fast conditional attention kernels; eigensolver device explicitly recorded in method_eigensolver_device; independent full evaluation', hot_loop_host_head_selection=False, all_compaction_cost_in_timing=True, ordinary_HOLD_visual_Dense_computed=False, common_QKV_O_RoPE_MLP_norm_lm_head_unchanged=True, canonical_reference_accuracy_reused=False)
        return result

    def _prepare_buffers(self, query):
        super()._prepare_buffers(query)
        self._pipeline_batches = {}
        self._pipeline_prototypes = {}
        self.backend = 'method_parallel_conditional_HOLD_CPU_eigen_method_graph_v1'

    def _batch_attention(self, layer, query, keys, values, mask=None):
        if self.work_counts is None:
            self._prepare_buffers(query)
        token = self.decode_tokens + 1
        n = keys.shape[-2]
        off = self.layer_offset[layer]
        if token > self.max_new_tokens or n > self.scratch_logits.shape[-1]:
            raise RuntimeError('Frozen allocation exceeded')
        state = self._state(layer, self.head_dim, query.device)
        decomp = self.decompositions.get(layer)
        if decomp is None:
            decomp = self._prepare_decomposition(layer, keys)
        if not hasattr(state, 'kernel_pointer_args'):
            state.kernel_pointer_args = (state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base)
        sp = state.kernel_pointer_args
        actual_mask = query if mask is None else mask
        signature = (mask is not None, actual_mask.dtype, decomp.basis.shape[-1], decomp.low_keys.stride(0), float(self.model.model.language_model.layers[layer].self_attn.scaling))
        key = (layer, signature)
        batch = self._pipeline_batches.get(key)
        if batch is None:
            proto = self._pipeline_prototypes.get(signature)
            if proto is None:
                cpu = tuple((getattr(self, k) for k in ['decode_tokens', 'last_nonvisual_tokens', 'text_key_rows', 'total_key_rows']))
                entries = []
                with record_eight(entries):
                    ParallelConditionalMixin.attention(self, layer, query, keys, values, mask)
                for k, v in zip(['decode_tokens', 'last_nonvisual_tokens', 'text_key_rows', 'total_key_rows'], cpu):
                    setattr(self, k, v)
                assert [v[0] for v in entries] == ['merged_text_logits', 'parallel_low_query', 'parallel_visual_proxy', 'route_decode', 'dense_visual_logits', 'dense_normalizer', 'attention_values_all', 'finalize_all']
                batch = MethodBatch([v[1] for v in entries], [v[2] for v in entries])
                self._pipeline_prototypes[signature] = batch
            else:
                batch = proto.clone(sp, off)
            self._pipeline_batches[key] = batch
        self.last_nonvisual_tokens = n - self.visual_positions.numel()
        self.text_key_rows += self.num_heads * self.last_nonvisual_tokens
        self.total_key_rows += self.num_heads * n
        updates = (query.data_ptr(), keys.data_ptr(), values.data_ptr(), actual_mask.data_ptr(), decomp.basis.data_ptr(), decomp.low_keys.data_ptr(), n, keys.stride(1), values.stride(1), token)
        batch.launch_values(updates)
        if layer == self.layers[-1]:
            self.decode_tokens += 1
        return self.output_buffer

    def attention(self, layer, query, keys, values, mask=None):
        if not SHADOW_ACTIVE or self.decode_tokens + 1 > 4:
            return self._batch_attention(layer, query, keys, values, mask)
        if self.work_counts is None:
            self._prepare_buffers(query)
        state = self._state(layer, self.head_dim, query.device)
        if layer not in self.decompositions:
            self._prepare_decomposition(layer, keys)
        tensors = [state.holding, state.anchor_ready, state.anchor_token, state.anchor_readout, state.anchor_nonvisual_tokens, state.episode_m3_base, *self.kernel_counters, *self.kernel_floats, *self.kernel_work]
        before = [v.clone() for v in tensors]
        cpu = tuple((getattr(self, k) for k in ['decode_tokens', 'last_nonvisual_tokens', 'text_key_rows', 'total_key_rows']))
        expected = ParallelConditionalMixin.attention(self, layer, query, keys, values, mask).clone()
        after = [v.clone() for v in tensors]
        for v, b in zip(tensors, before):
            v.copy_(b)
        for k, v in zip(['decode_tokens', 'last_nonvisual_tokens', 'text_key_rows', 'total_key_rows'], cpu):
            setattr(self, k, v)
        actual = self._batch_attention(layer, query, keys, values, mask)
        checks = [torch.all(actual == expected)]
        for v, b in zip(tensors, after):
            checks.append(torch.all((v == b) | torch.isnan(v) & torch.isnan(b)) if v.dtype.is_floating_point else torch.all(v == b))
        assert bool(torch.stack(checks).all()), 'Eight-kernel batching changed output/state/counters'
        SHADOW_COUNTS['passed'] += 1
        return actual
_CLASSES = {}

def batched_gate(base):
    if base not in _CLASSES:
        _CLASSES[base] = type('BatchedPipeline' + base.__name__, (BatchedPipelineMixin, base), {})
    return _CLASSES[base]
