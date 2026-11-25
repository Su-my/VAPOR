# SPDX-License-Identifier: Apache-2.0
"""Full-key, query-chunked eager prefill."""
def make_query_chunked_eager(reference, chunk_size=512, threshold=8192):
    if chunk_size < 1 or threshold < 1:
        raise ValueError('Query chunk size and activation threshold must be positive')

    def bounded(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        queries = query.shape[-2]
        if queries <= threshold:
            return reference(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        if kwargs.get('output_attentions', False):
            raise RuntimeError('Bounded prefill cannot materialize the full attention matrix')
        if dropout or module.training:
            raise RuntimeError('Bounded prefill is restricted to evaluation without dropout')
        import torch
        output = query.new_empty((query.shape[0], queries, query.shape[1], query.shape[3]))
        for start in range(0, queries, chunk_size):
            end = min(start + chunk_size, queries)
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            part, weights = reference(module, query[..., start:end, :], key, value, mask, scaling, dropout=0.0, **kwargs)
            output[:, start:end, :, :] = part
            del part, weights
        return (output, None)
    return bounded
