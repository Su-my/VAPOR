# SPDX-License-Identifier: Apache-2.0
"""InternViT vision-only SDPA integration used by the research runner."""
import types
def install_vision_sdpa(model):
    import torch.nn.functional as F
    count = 0

    def sdpa(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_normalization:
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        assert not self.training
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))
    for module in model.vision_model.modules():
        if module.__class__.__name__ == 'InternAttention':
            assert not module.use_flash_attn
            module._naive_attn = types.MethodType(sdpa, module)
            count += 1
    assert count == 24
    return count
