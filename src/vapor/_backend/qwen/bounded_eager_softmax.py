# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
"""Keep full Native eager QK/AV; bound only the temporary FP32 softmax.

Each query row has the same keys, mask and torch.softmax call as Native. No
query/key/value approximation, no attention backend or sequence chunking.
"""

def make_bounded_eager(original, row_chunk=128, threshold=8192):

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.shape[-2] <= threshold:
            return original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
        assert not torch.is_grad_enabled() and (not module.training) and (dropout == 0.0)
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :key_states.shape[-2]]
        for start in range(0, query.shape[-2], row_chunk):
            end = min(start + row_chunk, query.shape[-2])
            weights[:, :, start:end, :] = torch.nn.functional.softmax(weights[:, :, start:end, :], dim=-1, dtype=torch.float32).to(query.dtype)
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return (output, weights)
    return attention
