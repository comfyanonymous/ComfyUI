import torch
import torch.nn as nn
import torch.nn.functional as F
# from comfy.ldm.modules.attention import optimized_attention #TODO


class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, dtype=None, device=None, operations=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.q_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.c_proj = operations.Linear(embed_dim, output_dim or embed_dim, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(dtype=x.dtype, device=x.device)  # (L+1)NC

        q = self.q_proj(x[:1])
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size = q.shape[1]
        q = q.view(1, batch_size * self.num_heads, self.embed_dim // self.num_heads).transpose(0, 1)
        k = k.view(k.shape[0], batch_size * self.num_heads, self.embed_dim // self.num_heads).transpose(0, 1)
        v = v.view(v.shape[0], batch_size * self.num_heads, self.embed_dim // self.num_heads).transpose(0, 1)

        attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(1, batch_size, self.embed_dim)

        attn_output = self.c_proj(attn_output)
        return attn_output.squeeze(0)
