# Fun ControlNet Model for Qwen Image
# Implements the QwenDiffSynth-style ControlNet architecture

import torch
import torch.nn as nn
import math

from .model import QwenImageTransformer2DModel, FeedForward
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.math import apply_rope1


class WeightOnlyNorm(nn.Module):
    """RMSNorm that only has a weight parameter (no learnable bias/eps)."""
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        # x: [B, H, T, D]
        return x * self.weight


class FunControlNetAttention(nn.Module):
    """
    Joint attention matching ComfyUI Qwen Attention (text + image streams).
    Implements the control_blocks.{i}.attn.* structure.
    """
    def __init__(self, dim=3072, heads=24, head_dim=128, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim

        # Image stream projections
        self.to_q = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.to_k = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.to_v = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

        # Text stream projections
        self.add_q_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.add_k_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.add_v_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

        # Q/K normalization
        self.norm_q = WeightOnlyNorm(head_dim, device=device, dtype=dtype)
        self.norm_k = WeightOnlyNorm(head_dim, device=device, dtype=dtype)
        self.norm_added_q = WeightOnlyNorm(head_dim, device=device, dtype=dtype)
        self.norm_added_k = WeightOnlyNorm(head_dim, device=device, dtype=dtype)

        # Output projections
        self.to_out = nn.Sequential(operations.Linear(dim, dim, bias=True, device=device, dtype=dtype))
        self.to_add_out = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, 
                image_rotary_emb=None, transformer_options=None):
        if encoder_hidden_states is None:
            raise ValueError("FunControlNetAttention requires encoder_hidden_states (text stream)")

        batch_size = hidden_states.shape[0]
        seq_img = hidden_states.shape[1]
        seq_txt = encoder_hidden_states.shape[1]

        # Project and reshape to [B, H, T, D]
        img_query = self.to_q(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
        img_key = self.to_k(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
        img_value = self.to_v(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2)

        txt_query = self.add_q_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
        txt_key = self.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
        txt_value = self.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2)

        # Q/K RMS norm (weight-only)
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        # Concatenate for joint attention: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        # Apply RoPE if provided
        if image_rotary_emb is not None:
            joint_query = apply_rope1(joint_query, image_rotary_emb)
            joint_key = apply_rope1(joint_key, image_rotary_emb)

        # Use optimized attention
        attention_mask = encoder_hidden_states_mask
        joint_hidden_states = optimized_attention_masked(
            joint_query,
            joint_key,
            joint_value,
            self.heads,
            attention_mask,
            transformer_options=transformer_options if transformer_options is not None else {},
            skip_reshape=True
        )

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class ProjWrap(nn.Module):
    """Wrapper to match the GELU projection pattern in FeedForward."""
    def __init__(self, linear):
        super().__init__()
        self.proj = linear

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FunControlNetBlock(nn.Module):
    """
    Full transformer block for Fun ControlNet.
    Implements the control_blocks.{i}.* structure.
    """
    def __init__(self, dim=3072, num_heads=24, head_dim=128, device=None, dtype=None, operations=None, 
                 block_id=0, has_before_proj=False):
        super().__init__()
        self.dim = dim
        self.block_id = block_id

        self.has_before_proj = has_before_proj
        if has_before_proj:
            self.before_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

        self.after_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.attn = FunControlNetAttention(dim, heads=num_heads, head_dim=head_dim, 
                                            device=device, dtype=dtype, operations=operations)

        # Normalization layers
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        # MLP layers (matching FeedForward structure)
        self.img_mlp = nn.Module()
        self.img_mlp.net = nn.ModuleList([
            ProjWrap(operations.Linear(dim, 4 * dim, device=device, dtype=dtype)),
            nn.Identity(),  # Placeholder for dropout
            operations.Linear(4 * dim, dim, device=device, dtype=dtype),
        ])

        self.txt_mlp = nn.Module()
        self.txt_mlp.net = nn.ModuleList([
            ProjWrap(operations.Linear(dim, 4 * dim, device=device, dtype=dtype)),
            nn.Identity(),
            operations.Linear(4 * dim, dim, device=device, dtype=dtype),
        ])

        # Modulation layers (output 6*dim for shift, scale, gate for both attention and MLP)
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, device=device, dtype=dtype)
        )
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, device=device, dtype=dtype)
        )

        # Initialize zero projections
        if has_before_proj:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def _mlp_forward(self, mlp_net, x):
        for module in mlp_net:
            x = module(x)
        return x

    def _modulate(self, x, mod_params):
        """Apply modulation: x * (1 + scale) + shift"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        modulated = x * (1 + scale) + shift
        return modulated, gate

    def forward(self, c, x, encoder_hidden_states, encoder_hidden_states_mask, 
                temb, image_rotary_emb, transformer_options=None):
        """
        Forward pass with VideoX's exact stacking logic.
        
        Args:
            c: Control tensor. For block 0: projected control context [B, S, D]
               For block 1+: stacked tensor from previous blocks [N, B, S, D]
            x: Hidden states from latent input [B, S, D] (used only in first block)
            encoder_hidden_states: Text features [B, T_txt, D]
            encoder_hidden_states_mask: Attention mask
            temb: Time embedding [B, D]
            image_rotary_emb: Rotary position embedding
            transformer_options: Options dict
        
        Returns:
            (encoder_hidden_states, c) where c is stacked [hints..., current_state]
        """
        if self.has_before_proj:
            # First block: combine control context with latent hidden states
            c = self.before_proj(c) + x
            all_c = []
        else:
            # Subsequent blocks: unpack stacked tensor, take last as input
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        
        # Now c is the control hidden states to process
        hidden_states = c
        
        # Get modulation parameters
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)
        
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Attention block
        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            transformer_options=transformer_options,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # MLP block
        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self._mlp_forward(self.img_mlp.net, img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self._mlp_forward(self.txt_mlp.net, txt_modulated2)

        # Project for hint output (zero-initialized)
        c_skip = self.after_proj(hidden_states)

        # Stack hints and current state for next block
        all_c = all_c + [c_skip, hidden_states]
        c = torch.stack(all_c)

        return encoder_hidden_states, c


class QwenImageFunControlNetModel(nn.Module):
    """
    Fun ControlNet model for Qwen Image.
    
    This implements the QwenDiffSynth-style ControlNet with full transformer blocks
    instead of simple linear projections.
    """
    def __init__(
        self,
        in_channels=64,
        control_hint_channels=64,
        inner_dim=3072,
        num_attention_heads=24,
        attention_head_dim=128,
        num_control_blocks=5,
        patch_size=2,
        joint_attention_dim=3584,  # Text encoder output dimension
        axes_dims_rope=(16, 56, 56),
        dtype=None,
        device=None,
        operations=None,
        **kwargs
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.inner_dim = inner_dim
        self.num_control_blocks = num_control_blocks
        self.main_model_double = 60  # Number of blocks in main model

        # Input projection for control hint
        # control_img_in expects: latent (64) + control hint (64) + extra (4) = 132 channels
        total_in_channels = in_channels + control_hint_channels + 4  # 132
        self.control_img_in = operations.Linear(total_in_channels, inner_dim, device=device, dtype=dtype)

        # Control blocks
        self.control_blocks = nn.ModuleList([
            FunControlNetBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                device=device,
                dtype=dtype,
                operations=operations,
                block_id=i,
                has_before_proj=(i == 0)  # First block has before_proj
            )
            for i in range(num_control_blocks)
        ])

        # Position embedding (shared with main model)

        # Position embedding (shared with main model)
        from comfy.ldm.flux.layers import EmbedND
        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

    def process_img(self, x):
        """Process image to patches, matching QwenImageTransformer2DModel.process_img"""
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        
        # Pad to patch size
        import comfy.ldm.common_dit
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (1, self.patch_size, self.patch_size))
        orig_shape = hidden_states.shape
        
        # Reshape to patches
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-3], 
                                           orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 3, 5, 1, 4, 6)
        hidden_states = hidden_states.reshape(orig_shape[0], 
                                              orig_shape[-3] * (orig_shape[-2] // 2) * (orig_shape[-1] // 2), 
                                              orig_shape[1] * 4)
        
        t_len = t
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device)
        if t_len > 1:
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, 
                                                                        device=x.device, dtype=x.dtype).unsqueeze(1).unsqueeze(1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, 
                                                                    device=x.device, dtype=x.dtype).unsqueeze(1).unsqueeze(0) - (h_len // 2)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, 
                                                                    device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0) - (w_len // 2)
        
        from einops import repeat
        return hidden_states, repeat(img_ids, "t h w c -> b (t h w) c", b=bs), orig_shape

    def forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        hint=None,
        transformer_options={},
        **kwargs
    ):
        """
        Forward pass for Fun ControlNet.
        
        Args:
            x: Latent input [B, C, T, H, W]
            timesteps: Timestep embeddings
            context: Text encoder hidden states
            attention_mask: Attention mask for text
            hint: Control hint image (preprocessed)
            transformer_options: Options dict
        
        Returns:
            dict with "input" key containing control signals for each main model block
        """
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        if encoder_hidden_states_mask is not None and not torch.is_floating_point(encoder_hidden_states_mask):
            encoder_hidden_states_mask = (encoder_hidden_states_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max

        # Borrow weights from main model if available
        # Fun ControlNet "Lite" architecture requires main model's input projections
        main_model = None
        if transformer_options is not None:
            # ComfyUI passes ModelPatcher in transformer_options["model"]
            model_patcher = transformer_options.get("model", None)
            if model_patcher is not None:
                 # patcher.model is the underlying diffusion model wrapper
                 main_model = getattr(model_patcher, "model", None)
                 # Unwrap if needed (e.g. if it's a wrapper like in SD3)
                 if hasattr(main_model, "diffusion_model"):
                     main_model = main_model.diffusion_model

        if main_model is None:
             # Fallback debug mode? No, we really need the weights.
             # But let's try to be helpful if user calls it weirdly.
             raise ValueError("Fun ControlNet requires access to main model weights (img_in, txt_in) but could not find model in transformer_options. Make sure you are using a standard Sampler node.")

        # 1. Process and project Latent Input (x)
        # Get patchified latent features [B, T_img, 64]
        latent_states, img_ids, orig_shape = self.process_img(x)
        # Project to inner_dim [B, T_img, 3072] using MAIN MODEL's weights
        x_inner = main_model.img_in(latent_states)

        # 2. Process and project Text Input (context)
        # Project from 3584 -> 3072 using MAIN MODEL's weights
        txt_hidden = main_model.txt_in(main_model.txt_norm(encoder_hidden_states))

        # 3. Process Control Input (hint)
        # Combine latent and hint with correct channel count (33 channels)
        # control_img_in expects 132 features = 33 channels × 4 (after patchification)
        hint_combined = torch.cat([x, hint], dim=1)  # [B, 32, T, H, W] (16+16)
        extra = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]), 
                           device=x.device, dtype=x.dtype)
        full_input = torch.cat([hint_combined, extra], dim=1)  # [B, 33, T, H, W]
        
        # Process to patches: 33 channels → 132 features
        full_states, _, _ = self.process_img(full_input)
        
        # Project control context to inner dimension [B, T_img, 3072]
        # This uses OUR OWN weights (present in checkpoint)
        c = self.control_img_in(full_states)

        # Create position embeddings
        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, 
                              ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).to(x.dtype).contiguous()

        # Create time embedding
        from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
        time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        timesteps_proj = time_proj(timesteps)
        # Simple time embedding projection
        temb = timesteps_proj.to(dtype=c.dtype)
        # Expand to inner_dim
        temb = temb.repeat(1, self.inner_dim // 256 + 1)[:, :self.inner_dim]

        # Run through control blocks with VideoX stacking logic
        for block in self.control_blocks:
            txt_hidden, c = block(
                c=c,
                x=x_inner,
                encoder_hidden_states=txt_hidden,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                transformer_options=transformer_options,
            )
            # x_inner is only used in first block, after that c carries through

        # Extract hints from stacked tensor (all but last element)
        # c is [N, B, S, D] where N = 2*num_blocks (hint + state per block)
        # hints are at even indices: [0, 2, 4, 6, 8] for 5 blocks
        all_outputs = list(torch.unbind(c))
        hints = all_outputs[:-1]  # All except final hidden state
        
        # The hints correspond to control_layers = [0, 12, 24, 36, 48]
        # We need to return them in a format ComfyUI's ControlNet expects
        # Map hints to the 60 main model layers
        control_layers = [0, 12, 24, 36, 48]
        controlnet_block_samples = []
        
        hint_idx = 0
        for layer_idx in range(self.main_model_double):
            if layer_idx in control_layers and hint_idx < len(hints):
                controlnet_block_samples.append(hints[hint_idx])
                hint_idx += 1
            else:
                # No hint for this layer - use zeros
                controlnet_block_samples.append(torch.zeros_like(hints[0]) if hints else None)
        
        # Filter out None values and return
        controlnet_block_samples = tuple(h for h in controlnet_block_samples if h is not None)
        
        return {"input": controlnet_block_samples}
