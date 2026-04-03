import torch
import torch.nn as nn
from dataclasses import dataclass

from comfy import sd1_clip
import comfy.utils
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import RMSNorm, BaseLlama, BaseGenerate, Llama2_


GEMMA4_VISION_CONFIG = {"num_channels": 3, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 768, "image_size": 896, "intermediate_size": 3072, "model_type": "gemma4_vision", "num_attention_heads": 12, "num_hidden_layers": 16, "patch_size": 16, "head_dim": 64, "rms_norm_eps": 1e-6, "position_embedding_size": 10240, "pooling_kernel_size": 3}
GEMMA4_AUDIO_CONFIG = {"hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 8, "intermediate_size": 4096, "conv_kernel_size": 5, "attention_chunk_size": 12, "attention_context_left": 13, "attention_context_right": 0, "attention_logit_cap": 50.0, "output_proj_dims": 1536, "rms_norm_eps": 1e-6, "residual_weight": 0.5, "gradient_clipping": 1e10, "hidden_act": "silu"}

@dataclass
class Gemma4_E4B_Config:
    vocab_size: int = 262144
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 42
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    transformer_type: str = "gemma4"
    head_dim = 256
    global_head_dim = 512
    rms_norm_add = False
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [512, 512, 512, 512, 512, False]
    rope_scale = None
    partial_rotary_factor: float = 0.25
    final_norm: bool = True
    lm_head: bool = False
    final_logit_softcapping: float = 30.0
    hidden_size_per_layer_input: int = 256
    num_kv_shared_layers: int = 18
    stop_tokens = [1, 106]
    vision_config = GEMMA4_VISION_CONFIG
    audio_config = GEMMA4_AUDIO_CONFIG
    mm_tokens_per_image = 280


def precompute_freqs_cis_proportional(head_dim, partial_rotary_factor, position_ids, theta, device=None):
    """Proportional RoPE: compute freqs for full head_dim, but only first rope_angles get non-zero frequencies."""
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles

    theta_numerator = torch.arange(0, 2 * rope_angles, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))

    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles, device=device)], dim=0)

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    sin_split = sin.shape[-1] // 2
    return (cos, sin[..., :sin_split], -sin[..., sin_split:])


class Gemma4Attention(nn.Module):
    def __init__(self, config, head_dim, device=None, dtype=None, ops=None):
        super().__init__()
        from comfy.text_encoders.llama import RMSNorm
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.inner_size = self.num_heads * head_dim

        ops = ops or nn
        self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = None
        self.k_norm = None
        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        freqs_cis=None,
        optimized_attention=None,
        past_key_value=None,
        sliding_window=None,
        shared_kv=None,
    ):
        from comfy.text_encoders.llama import apply_rope
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            xq = self.q_norm(xq)

        if shared_kv is not None:
            # KV-shared layer: borrow KV from source layer, skip own cache
            if len(shared_kv) == 3:
                xk, xv = shared_kv[0][:, :, :shared_kv[2]], shared_kv[1][:, :, :shared_kv[2]]
            else:
                xk, xv = shared_kv
            # Apply RoPE to Q only (K already has RoPE from source layer)
            xq, _ = apply_rope(xq, xq, freqs_cis=freqs_cis)  # dummy K, only Q result used
            present_key_value = None
            shareable_kv = None
        else:
            xk = self.k_proj(hidden_states)
            xv = self.v_proj(hidden_states)
            xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
            xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if self.k_norm is not None:
                xk = self.k_norm(xk)
            xv = _parameterless_rms_norm(xv)
            xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

            present_key_value = None
            if past_key_value is not None:
                index = 0
                num_tokens = xk.shape[2]
                if len(past_key_value) > 0:
                    past_key, past_value, index = past_key_value
                    if past_key.shape[2] >= (index + num_tokens):
                        past_key[:, :, index:index + xk.shape[2]] = xk
                        past_value[:, :, index:index + xv.shape[2]] = xv
                        xk = past_key[:, :, :index + xk.shape[2]]
                        xv = past_value[:, :, :index + xv.shape[2]]
                        present_key_value = (past_key, past_value, index + num_tokens)
                    else:
                        xk = torch.cat((past_key[:, :, :index], xk), dim=2)
                        xv = torch.cat((past_value[:, :, :index], xv), dim=2)
                        present_key_value = (xk, xv, index + num_tokens)
                else:
                    present_key_value = (xk, xv, index + num_tokens)

                if sliding_window is not None and xk.shape[2] > sliding_window:
                    xk = xk[:, :, -sliding_window:]
                    xv = xv[:, :, -sliding_window:]
                    attention_mask = attention_mask[..., -sliding_window:] if attention_mask is not None else None

            # KV for sharing with later layers
            shareable_kv = present_key_value if present_key_value is not None else (xk, xv)

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # scaling=1.0: pre-multiply Q to cancel optimized_attention's 1/sqrt(head_dim)
        xq = xq * (self.head_dim ** 0.5)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        return self.o_proj(output), present_key_value, shareable_kv


class TransformerBlockGemma4(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        from comfy.text_encoders.llama import MLP
        if config.sliding_attention is not None:
            self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]
        else:
            self.sliding_attention = False

        head_dim = config.head_dim if self.sliding_attention else config.global_head_dim

        self.self_attn = Gemma4Attention(config, head_dim=head_dim, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

        self.hidden_size_per_layer_input = getattr(config, 'hidden_size_per_layer_input', 0)
        if self.hidden_size_per_layer_input:
            ops_pl = ops or nn
            self.per_layer_input_gate = ops_pl.Linear(config.hidden_size, self.hidden_size_per_layer_input, bias=False, device=device, dtype=dtype)
            self.per_layer_projection = ops_pl.Linear(self.hidden_size_per_layer_input, config.hidden_size, bias=False, device=device, dtype=dtype)
            self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
            self.register_buffer("layer_scalar", torch.ones(1, device=device, dtype=dtype))
        else:
            self.layer_scalar = None

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None,
                past_key_value=None, per_layer_input=None, shared_kv=None):
        sliding_window = None
        if self.sliding_attention:
            sliding_window = self.sliding_attention
            if x.shape[1] > self.sliding_attention:
                sliding_mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)
                sliding_mask.tril_(diagonal=-self.sliding_attention)
                attention_mask = attention_mask + sliding_mask if attention_mask is not None else sliding_mask
            freqs_cis = freqs_cis[1]
        else:
            freqs_cis = freqs_cis[0]

        residual = x
        x = self.input_layernorm(x)
        x, present_key_value, shareable_kv = self.self_attn(
            hidden_states=x, attention_mask=attention_mask, freqs_cis=freqs_cis,
            optimized_attention=optimized_attention, past_key_value=past_key_value,
            sliding_window=sliding_window, shared_kv=shared_kv,
        )
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            x = self.per_layer_input_gate(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = x * per_layer_input
            x = self.per_layer_projection(x)
            x = self.post_per_layer_input_norm(x)
            x = residual + x

        if self.layer_scalar is not None:
            x = x * self.layer_scalar

        return x, present_key_value, shareable_kv


class Gemma4Transformer(Llama2_):
    """Llama2_ subclass with Gemma4-specific features: per-layer inputs, KV sharing, proportional RoPE."""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__(config, device=device, dtype=dtype, ops=ops)
        # Override transformer type
        self.normalize_in = True
        # Replace layers with Gemma4 blocks
        self.layers = nn.ModuleList([
            TransformerBlockGemma4(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])
        # Per-layer input mechanism
        self.hidden_size_per_layer_input = getattr(config, 'hidden_size_per_layer_input', 0)
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = ops.Embedding(
                config.vocab_size, config.num_hidden_layers * self.hidden_size_per_layer_input,
                device=device, dtype=dtype)
            self.per_layer_model_projection = ops.Linear(
                config.hidden_size, config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False, device=device, dtype=dtype)
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input, eps=config.rms_norm_eps,
                add=config.rms_norm_add, device=device, dtype=dtype)

    def compute_freqs_cis(self, position_ids, device):
        from comfy.text_encoders.llama import precompute_freqs_cis
        global_freqs = precompute_freqs_cis_proportional(
            self.config.global_head_dim, self.config.partial_rotary_factor,
            position_ids, self.config.rope_theta[0], device=device)
        sliding_freqs = precompute_freqs_cis(
            self.config.head_dim, position_ids, self.config.rope_theta[1], device=device)
        return [global_freqs, sliding_freqs]

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[],
                past_key_values=None, input_ids=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        if self.normalize_in:
            x *= self.config.hidden_size ** 0.5

        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = self.get_past_len(past_key_values)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        freqs_cis = self.compute_freqs_cis(position_ids, x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            mask = mask + causal_mask if mask is not None else causal_mask

        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        # Per-layer inputs
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            num_layers = self.config.num_hidden_layers
            hpl = self.hidden_size_per_layer_input
            per_layer_proj = self.per_layer_model_projection(x) * (1.0 / (self.config.hidden_size ** 0.5))
            per_layer_proj = self.per_layer_projection_norm(per_layer_proj.reshape(*x.shape[:-1], num_layers, hpl))
            if input_ids is not None and input_ids.shape[1] == x.shape[1]:
                per_layer_emb = self.embed_tokens_per_layer(input_ids).reshape(*input_ids.shape, num_layers, hpl) * (hpl ** 0.5)
                per_layer_inputs = (per_layer_proj + per_layer_emb) * (0.5 ** 0.5)
            else:
                per_layer_inputs = per_layer_proj

        # KV sharing: only last sliding (22) and last global (23) layers store KV for sharing
        num_kv_shared = getattr(self.config, 'num_kv_shared_layers', 0)
        first_kv_shared = self.config.num_hidden_layers - num_kv_shared if num_kv_shared > 0 else self.config.num_hidden_layers
        shared_sliding_kv = None  # KV from last non-shared sliding layer
        shared_global_kv = None   # KV from last non-shared global layer

        intermediate = None
        next_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None and len(past_key_values) > 0 else None

            layer_kwargs = {}
            if per_layer_inputs is not None:
                layer_kwargs['per_layer_input'] = per_layer_inputs[:, :, i, :]
            if i >= first_kv_shared and num_kv_shared > 0:
                is_sliding = hasattr(layer, 'sliding_attention') and layer.sliding_attention
                shared = shared_sliding_kv if is_sliding else shared_global_kv
                if shared is not None:
                    layer_kwargs['shared_kv'] = shared

            x, current_kv, shareable_kv = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis,
                                                optimized_attention=optimized_attention, past_key_value=past_kv, **layer_kwargs)

            next_key_values.append(current_kv if current_kv is not None else ())

            # Only track the last sliding/global before the sharing boundary
            if i < first_kv_shared and shareable_kv is not None:
                is_sliding = hasattr(layer, 'sliding_attention') and layer.sliding_attention
                if is_sliding:
                    shared_sliding_kv = shareable_kv
                else:
                    shared_global_kv = shareable_kv

            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        if len(next_key_values) > 0:
            return x, intermediate, next_key_values
        return x, intermediate


class Gemma4_E4B(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma4_E4B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Gemma4Transformer(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

        self.multi_modal_projector = Gemma4MultiModalProjector(config, dtype, device, operations)
        self.vision_model = Gemma4VisionEncoder(config.vision_config, dtype, device, operations)
        self.audio_model = Gemma4AudioEncoder(config.audio_config, dtype, device, operations)
        self.audio_projector = Gemma4AudioProjector({"audio_output_proj_dims": config.audio_config["output_proj_dims"], "text_hidden_size": config.hidden_size, "rms_norm_eps": config.rms_norm_eps}, dtype, device, operations)

    def logits(self, x):
        logits = super().logits(x)
        cap = self.model.config.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        config = self.model.config
        num_kv_shared = getattr(config, 'num_kv_shared_layers', 0)
        first_kv_shared = config.num_hidden_layers - num_kv_shared
        past_key_values = []
        for i in range(config.num_hidden_layers):
            if i >= first_kv_shared:
                past_key_values.append(())  # shared layers don't need KV cache
            else:
                sa = config.sliding_attention[i % len(config.sliding_attention)]
                hd = config.head_dim if sa else config.global_head_dim
                past_key_values.append((
                    torch.empty([batch, config.num_key_value_heads, max_cache_len, hd], device=device, dtype=execution_dtype),
                    torch.empty([batch, config.num_key_value_heads, max_cache_len, hd], device=device, dtype=execution_dtype),
                    0))
        return past_key_values

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = embed["data"].movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            vision_out = self.vision_model(image.to(device, dtype=torch.float32))
            return self.multi_modal_projector(vision_out), None
        if embed["type"] == "audio":
            audio = embed["data"].to(device, dtype=torch.float32)
            audio_out = self.audio_model(audio)
            return self.audio_projector(audio_out), None
        return None, None


# --- Vision Encoder ---
# Matches HF weight structure after conversion:
#   vision_model.patch_embedder.input_proj.weight [768, 768]
#   vision_model.patch_embedder.position_embedding_table [2, 10240, 768]
#   vision_model.encoder.layers.X.self_attn.{q,k,v,o}_proj.weight [768, 768]
#   vision_model.encoder.layers.X.self_attn.{q,k}_norm.weight [64]
#   vision_model.encoder.layers.X.mlp.{gate,up}_proj.weight [3072, 768]
#   vision_model.encoder.layers.X.mlp.down_proj.weight [768, 3072]
#   vision_model.encoder.layers.X.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight [768]

def _parameterless_rms_norm(x, eps=1e-6):
    """RMSNorm without learnable weight (used by Gemma4 v_norm and projectors)."""
    mean_squared = x.float().pow(2).mean(-1, keepdim=True) + eps
    return (x.float() * torch.pow(mean_squared, -0.5)).to(x.dtype)


def _compute_vision_2d_rope(head_dim, pixel_position_ids, theta=100.0, device=None):
    """Compute 2D RoPE for vision: separate frequencies for x and y dimensions.

    Args:
        head_dim: dimension per head (e.g. 64)
        pixel_position_ids: [batch, num_patches, 2] with (x, y) coords
        theta: RoPE base frequency
    Returns:
        (cos, sin) each of shape [batch, num_patches, head_dim]
    """
    rotary_dim_per_axis = head_dim // 2
    freq_indices = torch.arange(0, rotary_dim_per_axis, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (freq_indices / rotary_dim_per_axis))

    all_cos, all_sin = [], []
    for i in range(2):  # x and y
        dim_positions = pixel_position_ids[:, :, i].float()  # [batch, num_patches]
        freqs = torch.einsum('bi,j->bij', dim_positions, inv_freq.to(device))  # [batch, num_patches, rotary_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, num_patches, rotary_dim]
        all_cos.append(emb.cos())
        all_sin.append(emb.sin())

    cos = torch.cat(all_cos, dim=-1).to(pixel_position_ids.device)  # [batch, num_patches, head_dim]
    sin = torch.cat(all_sin, dim=-1).to(pixel_position_ids.device)
    return cos, sin


def _apply_vision_2d_rope(x, cos, sin):
    """Apply 2D RoPE (multidimensional) to vision query/key states.

    Splits x and cos/sin into ndim=2 parts, applies rotate_half RoPE to each independently.

    x: [batch, heads, seq, head_dim]
    cos, sin: [batch, seq, head_dim]
    """
    cos = cos.unsqueeze(1)  # [batch, 1, seq, head_dim]
    sin = sin.unsqueeze(1)

    def rotate_half(t):
        t1 = t[..., :t.shape[-1]//2]
        t2 = t[..., t.shape[-1]//2:]
        return torch.cat((-t2, t1), dim=-1)

    # Split into 2 parts (y and x dimensions)
    half = x.shape[-1] // 2
    x_parts = [x[..., :half], x[..., half:]]
    cos_parts = [cos[..., :half], cos[..., half:]]
    sin_parts = [sin[..., :half], sin[..., half:]]

    rotated_parts = []
    for xp, cp, sp in zip(x_parts, cos_parts, sin_parts):
        rotated_parts.append((xp * cp) + (rotate_half(xp) * sp))

    return torch.cat(rotated_parts, dim=-1)


class ClippedLinear(nn.Module):
    """Linear layer with activation clipping (from quantization-aware training).

    Stores input_max/min and output_max/min as buffers loaded from checkpoint.
    """
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations or nn
        self.linear = ops.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer('input_max', torch.tensor(float('inf'), device=device, dtype=dtype))
        self.register_buffer('input_min', torch.tensor(float('-inf'), device=device, dtype=dtype))
        self.register_buffer('output_max', torch.tensor(float('inf'), device=device, dtype=dtype))
        self.register_buffer('output_min', torch.tensor(float('-inf'), device=device, dtype=dtype))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        x = x.clamp(min=self.input_min, max=self.input_max)
        x = self.linear(x)
        x = x.clamp(min=self.output_min, max=self.output_max)
        return x


def _make_clipped_linear(in_f, out_f, bias=False, device=None, dtype=None, operations=None):
    return ClippedLinear(in_f, out_f, bias=bias, device=device, dtype=dtype, operations=operations)


class Gemma4VisionMLP(nn.Module):
    """SwiGLU MLP matching gate_proj/up_proj/down_proj structure."""
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.gate_proj = _make_clipped_linear(hidden_size, intermediate_size, device=device, dtype=dtype, operations=operations)
        self.up_proj = _make_clipped_linear(hidden_size, intermediate_size, device=device, dtype=dtype, operations=operations)
        self.down_proj = _make_clipped_linear(intermediate_size, hidden_size, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)

        self.q_proj = _make_clipped_linear(self.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype, operations=operations)
        self.k_proj = _make_clipped_linear(self.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype, operations=operations)
        self.v_proj = _make_clipped_linear(self.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype, operations=operations)
        self.o_proj = _make_clipped_linear(self.num_heads * self.head_dim, self.hidden_size, device=device, dtype=dtype, operations=operations)

        self.q_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)

    def forward(self, x, cos_sin=None, attention_mask=None, optimized_attention=None):
        batch_size, seq_length, _ = x.shape

        xq = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        xv = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xv = _parameterless_rms_norm(xv)

        # Apply 2D RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            xq = xq.transpose(1, 2)  # [B, H, S, D]
            xk = xk.transpose(1, 2)
            xq = _apply_vision_2d_rope(xq, cos, sin)
            xk = _apply_vision_2d_rope(xk, cos, sin)
        else:
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)

        xv = xv.to(xq.dtype).transpose(1, 2)

        # scaling=1.0 (Q/K already normalized), cancel optimized_attention's 1/sqrt(d)
        xq = xq * (self.head_dim ** 0.5)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        return self.o_proj(output)


class Gemma4VisionLayer(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config, device=device, dtype=dtype, operations=operations)
        self.mlp = Gemma4VisionMLP(config, device=device, dtype=dtype, operations=operations)
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)

    def forward(self, x, cos_sin=None, attention_mask=None, optimized_attention=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos_sin=cos_sin, attention_mask=attention_mask, optimized_attention=optimized_attention)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class Gemma4PatchEmbedder(nn.Module):
    """Patch embedding with learned 2D position embeddings via one-hot lookup."""
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]
        self.patch_size = patch_size
        self.position_embedding_size = config.get("position_embedding_size", 10240)

        self.input_proj = operations.Linear(3 * patch_size * patch_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.position_embedding_table = nn.Parameter(
            torch.empty(2, self.position_embedding_size, hidden_size, device=device, dtype=dtype)
        )

    def forward(self, pixel_values, pixel_position_ids):
        """
        pixel_values: [B, C, H, W] normalized as 2*(x-0.5)
        pixel_position_ids: [B, num_patches, 2] with (x,y) positions
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        # Extract and flatten patches: [B, num_patches, 3*patch_size^2]
        x = pixel_values.reshape(batch_size, channels, patches_h, self.patch_size, patches_w, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch_size, patches_h * patches_w, -1)

        hidden_states = self.input_proj(x.to(self.input_proj.weight.dtype))

        # Position embeddings via one-hot lookup
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = torch.nn.functional.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        pos_table = comfy.model_management.cast_to_device(self.position_embedding_table, hidden_states.device, hidden_states.dtype)
        one_hot = one_hot.permute(0, 2, 1, 3).to(pos_table)  # [B, 2, num_patches, pos_size]
        position_embeddings = one_hot @ pos_table  # [B, 2, num_patches, hidden]
        position_embeddings = position_embeddings.sum(dim=1)  # [B, num_patches, hidden]

        return hidden_states + position_embeddings


class Gemma4VisionEncoderLayers(nn.Module):
    """Wrapper to produce state dict keys as encoder.layers.X.*"""
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            Gemma4VisionLayer(config, device=device, dtype=dtype, operations=operations)
            for _ in range(config["num_hidden_layers"])
        ])


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.patch_size = config["patch_size"]
        self.pooling_kernel_size = config.get("pooling_kernel_size", 3)
        self.root_hidden_size = self.hidden_size ** 0.5

        self.patch_embedder = Gemma4PatchEmbedder(config, device=device, dtype=dtype, operations=operations)
        self.encoder = Gemma4VisionEncoderLayers(config, dtype=dtype, device=device, operations=operations)

    def forward(self, pixel_values):
        """
        pixel_values: [B, C, H, W] in [0, 1] range
        Returns: [B, output_tokens, hidden_size] projected vision tokens
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size
        num_patches = patches_h * patches_w

        # Generate position IDs: grid of (col, row) per patch
        # HF processor uses (x=col, y=row) convention for position_ids
        rows = torch.arange(patches_h, device=pixel_values.device)
        cols = torch.arange(patches_w, device=pixel_values.device)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        pixel_position_ids = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [num_patches, 2]
        pixel_position_ids = pixel_position_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_patches, 2]

        # Patch embedding + position embedding
        x = self.patch_embedder(pixel_values, pixel_position_ids)

        # Compute 2D RoPE cos/sin for attention
        cos_sin = _compute_vision_2d_rope(self.head_dim, pixel_position_ids, device=pixel_values.device)

        optimized_attention = optimized_attention_for_device(x.device, mask=False, small_input=True)

        for layer in self.encoder.layers:
            x = layer(x, cos_sin=cos_sin, optimized_attention=optimized_attention)

        # Pooling: position-aware average pooling matching HF's Gemma4VisionPooler
        k = self.pooling_kernel_size  # 3
        k_squared = k * k
        output_length = num_patches // k_squared
        if num_patches != output_length and output_length > 0:
            # Assign each patch to a kernel block based on its (col, row) position
            kernel_col = pixel_position_ids[:, :, 0] // k  # col // k
            kernel_row = pixel_position_ids[:, :, 1] // k  # row // k
            stride = patches_w // k  # matches HF's (max_x + 1) // k
            kernel_idxs = kernel_col + stride * kernel_row  # [B, num_patches]

            # One-hot assignment matrix and weighted average
            weights = torch.nn.functional.one_hot(kernel_idxs.long(), output_length).float() / k_squared
            x = (weights.transpose(1, 2) @ x.float()).to(x.dtype)  # [B, output_length, hidden]

        # Scale by sqrt(hidden_size) like HF pooler
        x = x * self.root_hidden_size
        return x


class Gemma4MultiModalProjector(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        vision_hidden_size = config.vision_config["hidden_size"]
        text_hidden_size = config.hidden_size
        self.embedding_projection = operations.Linear(vision_hidden_size, text_hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, vision_outputs):
        return self.embedding_projection(_parameterless_rms_norm(vision_outputs))


# --- Audio Encoder ---
# Conformer-style architecture matching HF weight structure after conversion:
#   audio_model.subsample_conv_projection.layer0.conv.weight [128, 1, 3, 3]
#   audio_model.subsample_conv_projection.layer0.norm.weight [128]
#   audio_model.subsample_conv_projection.layer1.conv.weight [32, 128, 3, 3]
#   audio_model.subsample_conv_projection.layer1.norm.weight [32]
#   audio_model.subsample_conv_projection.input_proj_linear.weight [1024, 1024]
#   audio_model.layers.X.feed_forward1.{pre,post}_layer_norm.weight [1024]
#   audio_model.layers.X.feed_forward1.ffw_layer_{1,2}.weight [4096/1024, 1024/4096]
#   audio_model.layers.X.self_attn.{q,k,v}_proj.weight [1024, 1024]
#   audio_model.layers.X.self_attn.post.weight [1024, 1024]
#   audio_model.layers.X.self_attn.per_dim_scale [128]
#   audio_model.layers.X.self_attn.relative_k_proj.weight [1024, 1024]
#   audio_model.layers.X.lconv1d.{linear_start,linear_end}.weight, depthwise_conv1d.weight
#   audio_model.layers.X.feed_forward2.* (same as feed_forward1)
#   audio_model.output_proj.{weight, bias}

class Gemma4AudioConvSubsampler(nn.Module):
    """2D convolution subsampling for audio features, matching HF Gemma4AudioSubSampleConvProjection."""
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        eps = config.get("rms_norm_eps", 1e-6)
        self.layer0 = nn.ModuleDict({
            'conv': operations.Conv2d(1, 128, kernel_size=3, stride=2, padding=1, bias=False, device=device, dtype=dtype),
            'norm': operations.LayerNorm(128, eps=eps, elementwise_affine=True, bias=False, device=device, dtype=dtype),
        })
        self.layer1 = nn.ModuleDict({
            'conv': operations.Conv2d(128, 32, kernel_size=3, stride=2, padding=1, bias=False, device=device, dtype=dtype),
            'norm': operations.LayerNorm(32, eps=eps, elementwise_affine=True, bias=False, device=device, dtype=dtype),
        })
        # proj_input_dim = (128 // 4) * 32 = 1024
        self.input_proj_linear = operations.Linear(1024, config["hidden_size"], bias=False, device=device, dtype=dtype)

    def forward(self, x):
        # x: [batch, time, features]
        x = x.unsqueeze(1)  # [batch, 1, time, features]
        x = self.layer0['conv'](x.to(self.layer0['conv'].weight.dtype))
        x = torch.relu(self.layer0['norm'](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())

        x = self.layer1['conv'](x)
        x = torch.relu(self.layer1['norm'](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())

        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        return self.input_proj_linear(x)


class Gemma4AudioFeedForward(nn.Module):
    """Conformer feed-forward with gradient clipping and residual scaling."""
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        self.pre_layer_norm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.ffw_layer_1 = _make_clipped_linear(hidden_size, intermediate_size, device=device, dtype=dtype, operations=operations)
        self.ffw_layer_2 = _make_clipped_linear(intermediate_size, hidden_size, device=device, dtype=dtype, operations=operations)
        self.post_layer_norm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.post_layer_scale = config.get("residual_weight", 0.5)
        self.gradient_clipping = config.get("gradient_clipping", 1e10)

    def forward(self, x):
        residual = x
        gc = min(self.gradient_clipping, torch.finfo(x.dtype).max)
        x = torch.clamp(x, -gc, gc)
        x = self.pre_layer_norm(x)
        x = torch.nn.functional.silu(self.ffw_layer_1(x))
        x = self.ffw_layer_2(x)
        x = torch.clamp(x, -gc, gc)
        x = self.post_layer_norm(x)
        x = x * self.post_layer_scale
        return x + residual


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding for audio attention."""
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        chunk_size = config.get("attention_chunk_size", 12)
        context_left = config.get("attention_context_left", 13)
        context_right = config.get("attention_context_right", 0)
        self.context_size = chunk_size + context_left - 1 + context_right

        import math
        num_timescales = hidden_size // 2
        log_inc = math.log(10000.0) / max(num_timescales - 1, 1)
        inv_timescales = torch.exp(torch.arange(num_timescales) * -log_inc).unsqueeze(0).unsqueeze(0)
        self.register_buffer("inv_timescales", inv_timescales, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states):
        chunk_size = 12  # matches HF hardcoded value
        positions = torch.arange(chunk_size, -1, -1, device=hidden_states.device).unsqueeze(-1)
        scaled = positions * self.inv_timescales.to(device=hidden_states.device)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).to(dtype=hidden_states.dtype)


class Gemma4AudioAttention(nn.Module):
    """Chunked block attention with relative position bias and softcap."""
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        import math
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.chunk_size = config.get("attention_chunk_size", 12)
        self.max_past_horizon = config.get("attention_context_left", 13) - 1
        self.max_future_horizon = config.get("attention_context_right", 0)
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.q_scale = (self.head_dim ** -0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.softcap = config.get("attention_logit_cap", 50.0)

        self.q_proj = _make_clipped_linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, operations=operations)
        self.k_proj = _make_clipped_linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, operations=operations)
        self.v_proj = _make_clipped_linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, operations=operations)
        self.post = _make_clipped_linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, operations=operations)
        self.per_dim_scale = nn.Parameter(torch.empty(self.head_dim, device=device, dtype=dtype))
        self.relative_k_proj = operations.Linear(self.hidden_size, self.hidden_size, bias=False, device=device, dtype=dtype)

    def _convert_to_block(self, x):
        B, S, H, D = x.shape
        num_blocks = (S + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - S
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        return x.reshape(B, num_blocks, self.chunk_size, H, D)

    def _extract_block_context(self, x):
        B, S, H, D = x.shape
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1))
        x = x.unfold(1, self.context_size, self.chunk_size)
        return torch.movedim(x, -1, 2).contiguous()

    def _rel_shift(self, x):
        B, H, NB, BS, PL = x.shape
        CS = self.context_size
        x = torch.nn.functional.pad(x, (0, CS + 1 - PL))
        x = x.view(B, H, NB, BS * (CS + 1))
        x = x[..., :BS * CS]
        return x.view(B, H, NB, BS, CS)

    def forward(self, x, position_embeddings=None):
        B, S, _ = x.shape

        q = self.q_proj(x).float().view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).float().view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).float().view(B, S, self.num_heads, self.head_dim)

        q = q * self.q_scale * torch.nn.functional.softplus(self.per_dim_scale.float())
        k = k * self.k_scale

        q_blocks = self._convert_to_block(q)
        k_context = self._extract_block_context(k)
        v_context = self._extract_block_context(v)
        num_blocks = q_blocks.shape[1]

        rel_k = self.relative_k_proj(position_embeddings).view(-1, self.num_heads, self.head_dim).to(q.dtype)

        queries = q_blocks.permute(0, 3, 1, 2, 4)  # [B, H, NB, CS, D]
        matrix_ac = queries @ k_context.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(B, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ rel_k.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(B, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
        out = attn_weights @ v_context.permute(0, 3, 1, 2, 4)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, num_blocks * self.chunk_size, -1)
        out = out[:, :S].contiguous()
        return self.post(out.to(self.post.linear.weight.dtype))


class Gemma4AudioLConv1d(nn.Module):
    """Lightweight convolution with standard GLU."""
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        conv_kernel_size = config.get("conv_kernel_size", 5)
        self.gradient_clipping = config.get("gradient_clipping", 1e10)
        self.pre_layer_norm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.linear_start = _make_clipped_linear(hidden_size, hidden_size * 2, device=device, dtype=dtype, operations=operations)
        # Causal conv: left-pad only (no right padding)
        self.depthwise_conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=conv_kernel_size, padding=0, groups=hidden_size, bias=False, device=device, dtype=dtype)
        self.conv_left_pad = conv_kernel_size - 1  # causal: pad left by kernel-1
        self.conv_norm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.linear_end = _make_clipped_linear(hidden_size, hidden_size, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        x = torch.nn.functional.glu(x, dim=-1)  # standard GLU, not gelu-gated
        x = x.transpose(1, 2)
        x = torch.nn.functional.pad(x, (self.conv_left_pad, 0))
        x = self.depthwise_conv1d(x).transpose(1, 2)
        gc = min(self.gradient_clipping, torch.finfo(x.dtype).max)
        x = torch.clamp(x, -gc, gc)
        x = self.conv_norm(x)
        x = torch.nn.functional.silu(x)
        x = self.linear_end(x)
        return x + residual


class Gemma4AudioLayer(nn.Module):
    """Conformer block: FFN1 -> Attention -> LConv -> FFN2."""
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.gradient_clipping = config.get("gradient_clipping", 1e10)
        self.feed_forward1 = Gemma4AudioFeedForward(config, device=device, dtype=dtype, operations=operations)
        self.self_attn = Gemma4AudioAttention(config, device=device, dtype=dtype, operations=operations)
        self.norm_pre_attn = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.norm_post_attn = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)
        self.lconv1d = Gemma4AudioLConv1d(config, device=device, dtype=dtype, operations=operations)
        self.feed_forward2 = Gemma4AudioFeedForward(config, device=device, dtype=dtype, operations=operations)
        self.norm_out = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), device=device, dtype=dtype)

    def forward(self, x, position_embeddings=None):
        gc = min(self.gradient_clipping, torch.finfo(x.dtype).max)
        x = self.feed_forward1(x)

        residual = x
        x = torch.clamp(x, -gc, gc)
        x = self.norm_pre_attn(x)
        x = self.self_attn(x, position_embeddings=position_embeddings)
        x = torch.clamp(x, -gc, gc)
        x = self.norm_post_attn(x)
        x = x + residual

        x = self.lconv1d(x)
        x = self.feed_forward2(x)

        x = torch.clamp(x, -gc, gc)
        x = self.norm_out(x)
        return x


class Gemma4AudioEncoder(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.output_proj_dims = config.get("output_proj_dims", 1536)

        self.subsample_conv_projection = Gemma4AudioConvSubsampler(config, device=device, dtype=dtype, operations=operations)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            Gemma4AudioLayer(config, device=device, dtype=dtype, operations=operations)
            for _ in range(config["num_hidden_layers"])
        ])

        self.output_proj = operations.Linear(self.hidden_size, self.output_proj_dims, bias=True, device=device, dtype=dtype)

    def forward(self, audio_features):
        x = self.subsample_conv_projection(audio_features)
        position_embeddings = self.rel_pos_enc(x)

        for layer in self.layers:
            x = layer(x, position_embeddings=position_embeddings)

        x = self.output_proj(x)
        return x


class Gemma4AudioProjector(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        audio_output_dim = config.get("audio_output_proj_dims", 1536)
        text_hidden_size = config.get("text_hidden_size", 2560)
        self.embedding_projection = operations.Linear(audio_output_dim, text_hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, audio_outputs):
        return self.embedding_projection(_parameterless_rms_norm(audio_outputs))


# --- Tokenizer & Wrappers ---

class Gemma4_Tokenizer():
    def state_dict(self):
        return {}

    def _extract_mel_spectrogram(self, waveform, sample_rate):
        """Extract log mel spectrogram using HF's Gemma4AudioFeatureExtractor."""
        import torchaudio
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Convert to numpy for HF feature extractor
        audio_np = waveform.squeeze(0).numpy()
        fe = Gemma4AudioFeatureExtractor()
        result = fe([audio_np], return_tensors='pt')
        return result['input_features'][0]  # [T, 128]

    def tokenize_with_weights(self, text, return_word_ids=False, image=None, audio=None, llama_template=None, skip_template=True, thinking=False, **kwargs):
        if thinking:
            self.llama_template = "<|turn>system\n<|think|><turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n"
            self.llama_template_images = "<|turn>system\n<|think|><turn|>\n<|turn>user\n\n\n<|image><|image|><image|>\n\n{}<turn|>\n<|turn>model\n"
        else:
            self.llama_template = "<|turn>user\n{}<turn|>\n<|turn>model\n"
            self.llama_template_images = "<|turn>user\n\n\n<|image><|image|><image|>\n\n{}<turn|>\n<|turn>model\n"

        # Process audio
        audio_features = []
        if audio is not None:
            waveform = audio["waveform"].squeeze(0) if isinstance(audio, dict) else audio
            sample_rate = audio.get("sample_rate", 16000) if isinstance(audio, dict) else 16000
            mel = self._extract_mel_spectrogram(waveform, sample_rate)
            audio_features = [mel.unsqueeze(0)]  # [1, T, 128]

        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)  # [B, C, H, W]
            h, w = samples.shape[2], samples.shape[3]
            # Aspect-ratio-preserving resize matching HF Gemma4ImageProcessor
            patch_size = 16
            pooling_k = 3
            max_patches = 280 * pooling_k * pooling_k  # 2520
            target_px = max_patches * patch_size * patch_size
            factor = (target_px / (h * w)) ** 0.5
            side_mult = pooling_k * patch_size  # 48
            target_h = max(int(factor * h // side_mult) * side_mult, side_mult)
            target_w = max(int(factor * w // side_mult) * side_mult, side_mult)

            # Resize via PIL to match HF processor (operates on uint8, not float tensors)
            from PIL import Image
            import numpy as np
            img_uint8 = (samples[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img_uint8).resize((target_w, target_h), Image.BICUBIC)
            s = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            s = s.permute(2, 0, 1).unsqueeze(0).to(samples.device)
            s = 2 * (s - 0.5)  # normalize [0,1] -> [-1,1]
            images = [s.movedim(1, -1)[:, :, :, :3]]

        if text.startswith('<|turn>'):
            skip_template = True

        if skip_template:
            llama_text = text
        else:
            if llama_template is None:
                if len(images) > 0:
                    llama_text = self.llama_template_images.format(text)
                elif len(audio_features) > 0:
                    llama_text = f"<|turn>user\n\n<|audio><|audio|><audio|>{text}<turn|>\n<|turn>model\n"
                else:
                    llama_text = self.llama_template.format(text)
            else:
                llama_text = llama_template.format(text)

        text_tokens = super().tokenize_with_weights(llama_text, return_word_ids)

        if len(images) > 0:
            embed_count = 0
            for r in text_tokens:
                for i, token in enumerate(r):
                    if token[0] == 258880 and embed_count < len(images):
                        r[i] = ({"type": "image", "data": images[embed_count]},) + token[1:]
                        embed_count += 1

        if len(audio_features) > 0:
            embed_count = 0
            for r in text_tokens:
                for i, token in enumerate(r):
                    if token[0] == 258881 and embed_count < len(audio_features):
                        r[i] = ({"type": "audio", "data": audio_features[embed_count]},) + token[1:]
                        embed_count += 1

        return text_tokens


class Gemma4HFTokenizer:
    """Wrapper to load GemmaTokenizer from tokenizer.json bytes embedded in safetensors."""
    def __init__(self, tokenizer_json_bytes=None, **kwargs):
        import tempfile, os, json
        from transformers import AutoTokenizer
        self.temp_dir = tempfile.mkdtemp()
        if isinstance(tokenizer_json_bytes, torch.Tensor):
            tokenizer_json_bytes = bytes(tokenizer_json_bytes.tolist())
        with open(os.path.join(self.temp_dir, "tokenizer.json"), "wb") as f:
            f.write(tokenizer_json_bytes)
        # Minimal tokenizer_config.json
        with open(os.path.join(self.temp_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"tokenizer_class": "GemmaTokenizer", "add_bos_token": True, "add_eos_token": False}, f)
        self.tokenizer = AutoTokenizer.from_pretrained(self.temp_dir)

    @classmethod
    def from_pretrained(cls, tokenizer_data, **kwargs):
        return cls(tokenizer_json_bytes=tokenizer_data, **kwargs)

    def __call__(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return {"input_ids": ids}

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, **kwargs)


class Gemma4_E4BTokenizer(Gemma4_Tokenizer, sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_json = tokenizer_data.get("tokenizer_json", None)
        super().__init__(tokenizer_json, pad_with_end=False, embedding_size=2560, embedding_key='gemma4_e4b', tokenizer_class=Gemma4HFTokenizer, has_start_token=True, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_left=True, disable_weights=True, start_token=2, tokenizer_data=tokenizer_data)


class Gemma4Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="gemma4_e4b", tokenizer=Gemma4_E4BTokenizer)


class Gemma4_E4BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="all", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata
        self.dtypes = set()
        self.dtypes.add(dtype)
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, model_class=Gemma4_E4B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

    def process_tokens(self, tokens, device):
        embeds, _, _, embeds_info = super().process_tokens(tokens, device)
        scale = self.transformer.model.config.hidden_size ** 0.5
        # Undo text embedding scaling for multimodal tokens (vision/audio)
        for info in embeds_info:
            start_idx = info["index"]
            end_idx = start_idx + info["size"]
            embeds[:, start_idx:end_idx, :] /= scale
        return embeds

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = sd1_clip.SDClipModel.process_tokens(self, tokens_only, self.execution_device)
        # Build input_ids matching embeds length for per-layer embeddings
        # HF uses pad_token_id (0) at multimodal positions, not the placeholder ID
        base_ids = [t if isinstance(t, int) else 0 for t in tokens_only[0]]
        # Expand: each multimodal position was 1 token, now occupies `size` positions
        initial_token_ids = [base_ids]
        for info in sorted(embeds_info, key=lambda i: i["index"], reverse=True):
            idx, size = info["index"], info["size"]
            initial_token_ids[0] = initial_token_ids[0][:idx] + [0] * size + initial_token_ids[0][idx + 1:]
        scale = self.transformer.model.config.hidden_size ** 0.5
        for info in embeds_info:
            start_idx = info["index"]
            end_idx = start_idx + info["size"]
            embeds[:, start_idx:end_idx, :] /= scale
        input_ids = torch.tensor(initial_token_ids, device=self.execution_device)
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, stop_tokens=[106], initial_tokens=initial_token_ids[0], presence_penalty=presence_penalty, initial_input_ids=input_ids)


def gemma4_te(dtype_llama=None, llama_quantization_metadata=None):
    class Gemma4TEModel_(sd1_clip.SD1ClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, name="gemma4_e4b", clip_model=Gemma4_E4BModel, model_options=model_options)
    return Gemma4TEModel_
