# https://github.com/QwenLM/Qwen-Image (Apache 2.0)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple
from einops import repeat, rearrange

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.layers import EmbedND
import comfy.ldm.common_dit
import comfy.patcher_extension

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, bias=bias, dtype=dtype, device=device)
        self.approximate = approximate

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(GELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype, device=device, operations=operations))
        self.net.append(nn.Dropout(dropout))
        self.net.append(operations.Linear(inner_dim, dim_out, bias=bias, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def apply_rotary_emb(x, freqs_cis):
    if x.shape[1] == 0:
        return x
    t_ = x.reshape(*x.shape[:-1], -1, 1, 2)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.reshape(*x.shape)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            dtype=dtype,
            device=device,
            operations=operations
        )

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        return timesteps_emb


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Image stream projections
        self.to_q = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Text stream projections
        self.add_q_proj = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Output projections
        self.to_out = nn.ModuleList([
            operations.Linear(self.inner_dim, self.out_dim, bias=out_bias, dtype=dtype, device=device),
            nn.Dropout(dropout)
        ])
        self.to_add_out = operations.Linear(self.inner_dim, self.out_context_dim, bias=out_bias, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_txt = encoder_hidden_states.shape[1]

        img_query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        img_key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        img_value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        txt_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        # Handle both tuple (EliGen) and single tensor (standard) RoPE formats
        if isinstance(image_rotary_emb, tuple):
            # EliGen path: Apply RoPE BEFORE concatenation (research-accurate)
            # txt/img query/key are in [b, s, h, d] format, compatible with apply_rotary_emb
            img_rope, txt_rope = image_rotary_emb

            # Add heads dimension to RoPE tensors for broadcasting
            # Shape: [s, features, 2, 2] -> [s, 1, features, 2, 2]
            # Also convert to match query dtype (e.g., bfloat16)
            txt_rope = txt_rope.unsqueeze(1).to(dtype=txt_query.dtype)
            img_rope = img_rope.unsqueeze(1).to(dtype=img_query.dtype)

            # Apply RoPE separately to text and image streams
            txt_query = apply_rotary_emb(txt_query, txt_rope)
            txt_key = apply_rotary_emb(txt_key, txt_rope)
            img_query = apply_rotary_emb(img_query, img_rope)
            img_key = apply_rotary_emb(img_key, img_rope)

            # Now concatenate
            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_key = torch.cat([txt_key, img_key], dim=1)
            joint_value = torch.cat([txt_value, img_value], dim=1)
        else:
            # Standard path: Concatenate first, then apply RoPE
            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_key = torch.cat([txt_key, img_key], dim=1)
            joint_value = torch.cat([txt_value, img_value], dim=1)

            joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
            joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        # Apply EliGen attention mask if present
        effective_mask = attention_mask
        if transformer_options is not None:
            eligen_mask = transformer_options.get("eligen_attention_mask", None)
            if eligen_mask is not None:
                effective_mask = eligen_mask

                # Validate shape
                expected_seq = joint_query.shape[1]
                if eligen_mask.shape[-1] != expected_seq:
                    raise ValueError(
                        f"EliGen attention mask shape mismatch: {eligen_mask.shape} "
                        f"doesn't match sequence length {expected_seq}"
                    )

        # Use ComfyUI's optimized attention
        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        joint_hidden_states = optimized_attention_masked(joint_query, joint_key, joint_value, self.heads, effective_mask, transformer_options=transformer_options)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = torch.chunk(mod_params, 3, dim=-1)
        return torch.addcmul(shift.unsqueeze(1), x, 1 + scale.unsqueeze(1)), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            transformer_options=transformer_options,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        hidden_states = torch.addcmul(hidden_states, img_gate2, self.img_mlp(img_modulated2))

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, self.txt_mlp(txt_modulated2))

        return encoder_hidden_states, hidden_states


class LastLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=False,
        eps=1e-6,
        bias=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = torch.addcmul(shift[:, None, :], self.norm(x), (1 + scale)[:, None, :])
        return x

class QwenImageTransformer2DModel(nn.Module):
    # Constants for EliGen processing
    LATENT_TO_PIXEL_RATIO = 8  # Latents are 8x downsampled from pixel space
    PATCH_TO_LATENT_RATIO = 2  # 2x2 patches in latent space
    PATCH_TO_PIXEL_RATIO = 16  # Combined: 2x2 patches on 8x downsampled latents = 16x in pixel space

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        image_model=None,
        final_layer=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))
        
        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                device=device,
                operations=operations
            )
            for _ in range(num_layers)
        ])

        if final_layer:
            self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
            self.proj_out = operations.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True, dtype=dtype, device=device)

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (1, self.patch_size, self.patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1) - (h_len // 2)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0) - (w_len // 2)
        return hidden_states, repeat(img_ids, "h w c -> b (h w) c", b=bs), orig_shape

    def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, entity_prompt_emb,
                            entity_prompt_emb_mask, entity_masks, height, width, image,
                            cond_or_uncond=None, batch_size=None):
        """
        Process entity masks and build spatial attention mask for EliGen.

        Concatenates entity+global prompts, builds RoPE embeddings, creates attention mask
        enforcing spatial restrictions, and handles CFG batching with separate masks.

        Based on: https://github.com/modelscope/DiffSynth-Studio
        """
        num_entities = len(entity_prompt_emb)
        actual_batch_size = latents.shape[0]

        has_positive = cond_or_uncond and 0 in cond_or_uncond
        has_negative = cond_or_uncond and 1 in cond_or_uncond
        is_cfg_batched = has_positive and has_negative

        logger.debug(
            f"[EliGen Model] Processing {num_entities} entities for {height}x{width}px, "
            f"batch_size={actual_batch_size}, CFG_batched={is_cfg_batched}"
        )

        # Concatenate entity + global prompts
        all_prompt_emb = entity_prompt_emb + [prompt_emb]
        all_prompt_emb = [self.txt_in(self.txt_norm(local_prompt_emb)) for local_prompt_emb in all_prompt_emb]
        all_prompt_emb = torch.cat(all_prompt_emb, dim=1)

        # Build RoPE embeddings
        patch_h = height // self.PATCH_TO_PIXEL_RATIO
        patch_w = width // self.PATCH_TO_PIXEL_RATIO

        entity_seq_lens = [int(mask.sum(dim=1)[0].item()) for mask in entity_prompt_emb_mask]

        if prompt_emb_mask is not None:
            global_seq_len = int(prompt_emb_mask.sum(dim=1)[0].item())
        else:
            global_seq_len = int(prompt_emb.shape[1])

        max_vid_index = max(patch_h // 2, patch_w // 2)

        # Generate per-entity text RoPE (each entity starts from same offset)
        entity_txt_embs = []
        for entity_seq_len in entity_seq_lens:
            entity_ids = torch.arange(
                max_vid_index,
                max_vid_index + entity_seq_len,
                device=latents.device
            ).reshape(1, -1, 1).repeat(1, 1, 3)

            entity_rope = self.pe_embedder(entity_ids).squeeze(1).squeeze(0)
            entity_txt_embs.append(entity_rope)

        # Generate global text RoPE
        global_ids = torch.arange(
            max_vid_index,
            max_vid_index + global_seq_len,
            device=latents.device
        ).reshape(1, -1, 1).repeat(1, 1, 3)
        global_rope = self.pe_embedder(global_ids).squeeze(1).squeeze(0)

        txt_rotary_emb = torch.cat(entity_txt_embs + [global_rope], dim=0)

        h_coords = torch.arange(-(patch_h - patch_h // 2), patch_h // 2, device=latents.device)
        w_coords = torch.arange(-(patch_w - patch_w // 2), patch_w // 2, device=latents.device)

        img_ids = torch.zeros((patch_h, patch_w, 3), device=latents.device)
        img_ids[:, :, 0] = 0
        img_ids[:, :, 1] = h_coords.unsqueeze(1)
        img_ids[:, :, 2] = w_coords.unsqueeze(0)
        img_ids = img_ids.reshape(1, -1, 3)

        img_rope = self.pe_embedder(img_ids).squeeze(1).squeeze(0)

        logger.debug(f"[EliGen Model] RoPE shapes - img: {img_rope.shape}, txt: {txt_rotary_emb.shape}")

        image_rotary_emb = (img_rope, txt_rotary_emb)

        # Prepare spatial masks
        repeat_dim = latents.shape[1]
        max_masks = entity_masks.shape[1]
        entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)

        padded_h = height // self.LATENT_TO_PIXEL_RATIO
        padded_w = width // self.LATENT_TO_PIXEL_RATIO
        if entity_masks.shape[3] != padded_h or entity_masks.shape[4] != padded_w:
            pad_h = padded_h - entity_masks.shape[3]
            pad_w = padded_w - entity_masks.shape[4]
            logger.debug(f"[EliGen Model] Padding masks by ({pad_h}, {pad_w})")
            entity_masks = torch.nn.functional.pad(entity_masks, (0, pad_w, 0, pad_h), mode='constant', value=0)

        entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]

        global_mask = torch.ones((entity_masks[0].shape[0], entity_masks[0].shape[1], padded_h, padded_w),
                                 device=latents.device, dtype=latents.dtype)
        entity_masks = entity_masks + [global_mask]

        # Patchify masks
        N = len(entity_masks)
        batch_size = int(entity_masks[0].shape[0])
        seq_lens = entity_seq_lens + [global_seq_len]
        total_seq_len = int(sum(seq_lens) + image.shape[1])

        logger.debug(f"[EliGen Model] total_seq={total_seq_len}")

        patched_masks = []
        for i in range(N):
            patched_mask = rearrange(
                entity_masks[i],
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=height // self.PATCH_TO_PIXEL_RATIO,
                W=width // self.PATCH_TO_PIXEL_RATIO,
                P=self.PATCH_TO_LATENT_RATIO,
                Q=self.PATCH_TO_LATENT_RATIO
            )
            patched_masks.append(patched_mask)

        # SECTION 5: Build attention mask matrix
        attention_mask = torch.ones(
            (batch_size, total_seq_len, total_seq_len),
            dtype=torch.bool
        ).to(device=entity_masks[0].device)

        # Calculate positions
        image_start = int(sum(seq_lens))
        image_end = int(total_seq_len)
        cumsum = [0]
        single_image_seq = int(image_end - image_start)

        for length in seq_lens:
            cumsum.append(cumsum[-1] + length)

        # RULE 1: Spatial restriction (prompt <-> image)
        for i in range(N):
            prompt_start = cumsum[i]
            prompt_end = cumsum[i+1]

            # Create binary mask for which image patches this entity can attend to
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, seq_lens[i], 1)

            # Always repeat mask to match image sequence length
            repeat_time = single_image_seq // image_mask.shape[-1]
            image_mask = image_mask.repeat(1, 1, repeat_time)

            # Bidirectional restriction:
            # - Entity prompt can only attend to its masked image regions
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            # - Image patches can only be updated by prompts that own them
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)

        # RULE 2: Entity isolation
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                start_i, end_i = cumsum[i], cumsum[i+1]
                start_j, end_j = cumsum[j], cumsum[j+1]
                attention_mask[:, start_i:end_i, start_j:end_j] = False

        # SECTION 6: Convert to additive bias and handle CFG batching
        attention_mask = attention_mask.float()
        num_valid_connections = (attention_mask == 1).sum().item()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        attention_mask = attention_mask.to(device=latents.device, dtype=latents.dtype)

        # Handle CFG batching: Create separate masks for positive and negative
        if is_cfg_batched and actual_batch_size > 1:
            # CFG batch: [positive, negative] - need different masks for each
            # Positive gets entity constraints, negative gets standard attention (all zeros)

            logger.debug(
                f"[EliGen Model] CFG batched detected - creating separate masks. "
                f"Positive (index 0) gets entity mask, Negative (index 1) gets standard mask"
            )

            # Create standard attention mask (all zeros = no constraints)
            standard_mask = torch.zeros_like(attention_mask)

            # Stack masks according to cond_or_uncond order
            mask_list = []
            for cond_type in cond_or_uncond:
                if cond_type == 0:  # Positive - use entity mask
                    mask_list.append(attention_mask[0:1])  # Take first (and only) entity mask
                else:  # Negative - use standard mask
                    mask_list.append(standard_mask[0:1])

            # Concatenate masks to match batch
            attention_mask = torch.cat(mask_list, dim=0)

            logger.debug(
                f"[EliGen Model] Created {len(mask_list)} masks for CFG batch. "
                f"Final shape: {attention_mask.shape}"
            )

        # Add head dimension: [B, 1, seq, seq]
        attention_mask = attention_mask.unsqueeze(1)

        logger.debug(
            f"[EliGen Model] Attention mask created: shape={attention_mask.shape}, "
            f"valid_connections={num_valid_connections}/{total_seq_len * total_seq_len}"
        )

        return all_prompt_emb, image_rotary_emb, attention_mask

    def forward(self, x, timestep, context, attention_mask=None, guidance=None, ref_latents=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, attention_mask, guidance, ref_latents, transformer_options, **kwargs)

    def _forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        control=None,
        **kwargs
    ):
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        # Extract EliGen entity data
        entity_prompt_emb = kwargs.get("entity_prompt_emb", None)
        entity_prompt_emb_mask = kwargs.get("entity_prompt_emb_mask", None)
        entity_masks = kwargs.get("entity_masks", None)

        # Detect batch composition for CFG handling
        cond_or_uncond = transformer_options.get("cond_or_uncond", []) if transformer_options else []
        is_positive_cond = 0 in cond_or_uncond
        is_negative_cond = 1 in cond_or_uncond
        batch_size = x.shape[0]

        if entity_prompt_emb is not None:
            logger.debug(
                f"[EliGen Forward] batch_size={batch_size}, cond_or_uncond={cond_or_uncond}, "
                f"has_positive={is_positive_cond}, has_negative={is_negative_cond}"
            )

        if entity_prompt_emb is not None and entity_masks is not None and entity_prompt_emb_mask is not None and is_positive_cond:
            # EliGen path
            height = int(orig_shape[-2] * 8)
            width = int(orig_shape[-1] * 8)

            encoder_hidden_states, image_rotary_emb, eligen_attention_mask = self.process_entity_masks(
                latents=x,
                prompt_emb=encoder_hidden_states,
                prompt_emb_mask=encoder_hidden_states_mask,
                entity_prompt_emb=entity_prompt_emb,
                entity_prompt_emb_mask=entity_prompt_emb_mask,
                entity_masks=entity_masks,
                height=height,
                width=width,
                image=hidden_states,
                cond_or_uncond=cond_or_uncond,
                batch_size=batch_size
            )

            hidden_states = self.img_in(hidden_states)

            if transformer_options is None:
                transformer_options = {}
            transformer_options["eligen_attention_mask"] = eligen_attention_mask

            del img_ids

        else:
            # Standard path
            txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
            txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
            ids = torch.cat((txt_ids, img_ids), dim=1)
            image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
            del ids, txt_ids, img_ids

            hidden_states = self.img_in(hidden_states)
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})

        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]
