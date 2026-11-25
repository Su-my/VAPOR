# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle helpers. One request at a time in each process."""
from threading import Lock

REQUEST_LOCK = Lock()

class RequestState:
    def __init__(self, manager, config):
        self.manager = manager
        self.config = config
        self.consistency = None
        self.threshold = config.threshold

    def summary(self):
        m = self.manager
        refresh = int(m.dense_rows.sum().item())
        persist = int(m.hold_rows.sum().item())
        return dict(refresh_head_steps=refresh, persist_head_steps=persist,
                    refresh_rate=refresh / (refresh + persist) if refresh + persist else None,
                    cached_decode_steps=m.decode_tokens, threshold=self.threshold,
                    consistency_median=self.consistency, rank=self.config.rank,
                    cache_age=self.config.cache_age)


def validate(model, text, layers, input_ids, config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("VAPOR's research kernels require NVIDIA CUDA")
    if model.training or torch.is_grad_enabled():
        raise ValueError("Use model.eval() inside torch.inference_mode()")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")
    if input_ids.shape[1] + config.max_new_tokens > 32768:
        raise ValueError("This release supports at most 32768 prompt + output tokens")
    if (len(layers), text.num_attention_heads, text.num_key_value_heads,
        text.hidden_size // text.num_attention_heads) != (36, 32, 8, 128):
        raise ValueError("Supported text geometry: 36 layers, 32 query heads, 8 KV heads, head_dim 128")
    for p in model.parameters():
        if p.device != torch.device('cuda:0') or (p.is_floating_point() and p.dtype != torch.bfloat16):
            raise ValueError("Load the full model in BF16 on cuda:0; select a GPU with CUDA_VISIBLE_DEVICES")
    if getattr(text, '_attn_implementation', None) != 'eager':
        raise ValueError("The language attention backend must be eager")
    if any(getattr(x.self_attn, '_svd_checkpoint_hold_wrapped', False) for x in layers):
        raise ValueError("Model already has a VAPOR adapter")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.linalg.eigh(torch.eye(128, device='cuda:0', dtype=torch.float32))


def begin(manager, input_ids, config):
    manager.begin_sample(input_ids.to('cuda:0'), m1=config.threshold,
                         confirmation_m1=config.threshold, m3_base=config.threshold,
                         max_anchor_distance=config.cache_age, rank=config.rank,
                         max_new_tokens=config.max_new_tokens)
    manager.fixed_age_decay = True
    manager.fixed_use_horizon = True
    manager.fixed_shared_age_gates = False
    manager.fixed_decay_kind = 0
