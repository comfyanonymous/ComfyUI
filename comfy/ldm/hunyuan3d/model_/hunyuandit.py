import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import MoEBlock
from torch.nn.attention import SDPBackend

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, downscale_freq_shift: float = 0.0,
                 scale: float = 1.0, max_period: int = 10000):
        super().__init__()

        self.num_channels = num_channels
        half_dim = num_channels // 2

        # precompute the “inv_freq” vector once
        exponent = -math.log(max_period) * torch.arange(
            half_dim, dtype=torch.float32
        ) / (half_dim - downscale_freq_shift)

        inv_freq = torch.exp(exponent) 

        # pad
        if num_channels % 2 == 1:
            # we’ll pad a zero at the end of the cos-half
            inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(1)])

        # register to buffer so it moves with the device
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        self.scale = scale

    def forward(self, timesteps: torch.Tensor):

        x = timesteps.float().unsqueeze(1) * self.inv_freq.unsqueeze(0)
        
        # scale factor
        if self.scale != 1.0:
            emb = emb * self.scale

        # fused CUDA kernels for sin and cos
        sin_emb = x.sin()
        cos_emb = x.cos()

        emb = torch.cat([sin_emb, cos_emb], dim = 1)

        # If we padded inv_freq for odd, emb is already wide enough; otherwise:
        if emb.shape[1] > self.num_channels:
            emb = emb[:, :self.num_channels]

        return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size = 256, cond_proj_dim = None):
        super().__init__()  
    
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, timesteps, condition):

        timestep_embed = self.time_embed(timesteps).type(self.mlp[0].weight.dtype)

        if condition is not None:
            cond_embed = self.cond_proj(condition)
            timestep_embed = timestep_embed + cond_embed

        time_conditioned = self.mlp(timestep_embed)

        # for broadcasting with image tokens
        return time_conditioned.unsqueeze(1) 
    
class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        use_fp16: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        
        self.num_heads = num_heads
        self.head_dim = self.qdim // num_heads
        
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        if use_fp16:
            eps = 1.0 / 65504
        else:
            eps = 1e-6

        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)
        
    def forward(self, x, y):

        b, s1, _ = x.shape  
        _, s2, _ = y.shape 

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2

        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)  
        k = k.view(b, s2, self.num_heads, self.head_dim) 
        v = v.view(b, s2, self.num_heads, self.head_dim) 

        q = self.q_norm(q)
        k = self.k_norm(k)

        # replaced with torch.nn.attention (avoid FutureWarning from backends.cuda.sdp_kerenl)
        with torch.nn.attention.sdpa_kernel(
            backends=[
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.MATH,
                    SDPBackend.EFFICIENT_ATTENTION,
                    ]
            ):
            q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]
            context = F.scaled_dot_product_attention(
                q, k, v
            ).transpose(1, 2).reshape(b, s1, -1)

        out = self.out_proj(context)

        return out
    
class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias = True,
        qk_norm = False,
        norm_layer = nn.LayerNorm,
        use_fp16: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias = qkv_bias)

        if use_fp16: 
            eps = 1.0 / 65504
        else: eps = 1e-6

        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps = eps) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, _ = x.shape

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        qkv_combined = torch.cat((query, key, value), dim=-1)
        split_size = qkv_combined.shape[-1] // self.num_heads // 3

        qkv = qkv_combined.view(1, -1, self.num_heads, split_size * 3)
        query, key, value = torch.split(qkv, split_size, dim=-1)

        query = query.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) 
        key = key.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  
        value = value.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        # replaced with torch.nn.attention (avoid FutureWarning from backends.cuda.sdp_kerenl)
        with torch.nn.attention.sdpa_kernel(
            backends=[
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            x = F.scaled_dot_product_attention(query, key, value)
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.out_proj(x)
        return x
    
class HunYuanDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        use_fp16: bool = False
    ):
        super().__init__()

        # eps can't be 1e-6 in fp16 mode because of numerical stability issues
        if use_fp16: 
            eps = 1.0 / 65504
        else: eps = 1e-6
        
        self.norm1 = norm_layer(hidden_size, elementwise_affine = True, eps = eps)

        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                               norm_layer=qk_norm_layer, use_fp16 = use_fp16)

        self.norm2 = norm_layer(hidden_size, elementwise_affine = True, eps = eps)

        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(c_emb_size, hidden_size, bias=True)
            )

        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_norm=qk_norm, norm_layer=qk_norm_layer, use_fp16 = use_fp16)
        
        self.norm3 = norm_layer(hidden_size, elementwise_affine = True, eps = eps)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine = True, eps = eps)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.use_moe = use_moe

        if self.use_moe:
            self.moe = MoEBlock(
                hidden_size,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                dropout = 0.0,
                ff_inner_dim = int(hidden_size * 4.0),
            )
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, hidden_states, conditioning=None, text_states=None, skip_tensor=None):

        if self.skip_linear is not None:
            combined = torch.cat([skip_tensor, hidden_states], dim=-1)
            hidden_states = self.skip_linear(combined)
            hidden_states = self.skip_norm(hidden_states)

        # self attention
        if self.timested_modulate:
            modulation_shift = self.default_modulation(conditioning).unsqueeze(dim=1)
            hidden_states = hidden_states + modulation_shift

        self_attn_out = self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + self_attn_out

        # cross attention
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), text_states)

        # MLP Layer
        mlp_input = self.norm3(hidden_states)

        if self.use_moe:
            hidden_states = hidden_states + self.moe(mlp_input)
        else:
            hidden_states = hidden_states + self.mlp(mlp_input)

        return hidden_states
    
class FinalLayer(nn.Module):

    def __init__(self, final_hidden_size, out_channels, use_fp16: bool = False):
        super().__init__()

        if use_fp16: 
            eps = 1.0 / 65504
        else: eps = 1e-6

        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine = True, eps = eps)
        self.linear = nn.Linear(final_hidden_size, out_channels, bias = True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x

class HunYuanDiTPlain(nn.Module):
    
    # init with the defaults values from https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/hunyuan3d-dit-v2-1/config.yaml
    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        num_moe_layers: int = 6,
        guidance_cond_proj_dim = None,
        norm_type = 'layer',
        num_experts: int = 8,
        moe_top_k: int = 2,
        use_fp16: bool = False
        ):

        super().__init__()

        self.depth = depth

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.num_heads = num_heads
        self.hidden_size = hidden_size

        norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        qk_norm = nn.RMSNorm

        self.context_dim = context_dim
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias = True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)


        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer = norm,
                            qk_norm_layer = qk_norm,
                            skip_connection=layer > depth // 2,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            use_fp16 = use_fp16)
            for layer in range(depth)
        ])

        self.depth = depth

        self.final_layer = FinalLayer(hidden_size, self.out_channels, use_fp16 = use_fp16)

    def forward(self, x, t, contexts, **kwargs):
        
        main_condition = contexts['main']

        time_embedded = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x_embedded = self.x_embedder(x)

        combined = torch.cat([time_embedded, x_embedded], dim=1)

        skip_stack = []
        for idx, block in enumerate(self.blocks):
            if idx <= self.depth // 2:
                skip_input = None
            else:
                skip_input = skip_stack.pop()

            combined = block(combined, time_embedded, main_condition, skip_tensor = skip_input)

            if idx < self.depth // 2:
                skip_stack.append(combined)

        output = self.final_layer(combined)
        return output
    
def get_diffusion_checkpoint():
    import requests

    url = "https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/hunyuan3d-dit-v2-1/model.fp16.ckpt"
    output_path = "model.fp16.ckpt"
    
    response = requests.get(url, stream=True)
    response.raise_for_status() 
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Downloaded to: {output_path}")

def load_dit(dit: HunYuanDiTPlain):

    DEBUG = False
    checkpoint = torch.load("model.fp16.ckpt")
    missing, unexpected = dit.load_state_dict(checkpoint["model"], strict = not DEBUG)
    
    if DEBUG:
        print(f"Missing {len(missing)}", missing)
        print(f"Unexpected {len(unexpected)}", unexpected)

    return dit

if __name__ == "__main__":

    torch.manual_seed(2025)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.bfloat16)

    import time

    timings = {}
    
    start = time.time()
    
    model = HunYuanDiTPlain(depth = 10)

    timings["model_initialization"] = time.time() - start

    batch_size = 2
    seq_len = 1370
    in_channels = 64
    context_dim = 1024

    # Random inputs
    x = torch.randn(batch_size, seq_len, in_channels)
    t = torch.randint(0, seq_len, (batch_size,))
    contexts = {
        'main': torch.randn(batch_size, seq_len, context_dim)
    }

    # Forward pass
    start = time.time()
    output = model(x, t, contexts)
    timings["forward_timing"] = time.time() - start

    print("\n=== Timing Summary ===")
    for key, value in timings.items():
        print(f"{key}: {value:.3f} seconds")