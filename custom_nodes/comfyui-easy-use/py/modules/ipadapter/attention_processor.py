import numbers
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states

class IPAFluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, timestep_range=None):
        super().__init__()

        self.hidden_size = hidden_size  # 3072
        self.cross_attention_dim = cross_attention_dim  # 4096
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)
        self.norm_added_v = RMSNorm(128, eps=1e-5, elementwise_affine=False)
        self.timestep_range = timestep_range

    def __call__(
            self,
            num_heads,
            query,
            image_emb: torch.FloatTensor,
            t: torch.FloatTensor
    ) -> torch.FloatTensor:
        # only apply IPA if timestep is within range
        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return None
        # `ip-adapter` projections
        ip_hidden_states = image_emb
        ip_hidden_states_key_proj = self.to_k_ip(ip_hidden_states)
        ip_hidden_states_value_proj = self.to_v_ip(ip_hidden_states)

        ip_hidden_states_key_proj = rearrange(ip_hidden_states_key_proj, 'B L (H D) -> B H L D', H=num_heads)
        ip_hidden_states_value_proj = rearrange(ip_hidden_states_value_proj, 'B L (H D) -> B H L D', H=num_heads)

        ip_hidden_states_key_proj = self.norm_added_k(ip_hidden_states_key_proj)
        ip_hidden_states_value_proj = self.norm_added_v(ip_hidden_states_value_proj)

        ip_hidden_states = F.scaled_dot_product_attention(query.to(image_emb.device).to(image_emb.dtype),
                                                          ip_hidden_states_key_proj,
                                                          ip_hidden_states_value_proj,
                                                          dropout_p=0.0, is_causal=False)

        ip_hidden_states = rearrange(ip_hidden_states, "B H L D -> B L (H D)", H=num_heads)
        ip_hidden_states = ip_hidden_states.to(query.dtype).to(query.device)

        return self.scale * ip_hidden_states