# Original from: https://github.com/ace-step/ACE-Step/blob/main/models/ace_step_transformer.py

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, List, Union

import torch
from torch import nn

import comfy.model_management
import comfy.patcher_extension

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from .attention import LinearTransformerBlock, t2i_modulate
from .lyric_encoder import ConformerEncoder as LyricEncoder


def cross_norm(hidden_states, controlnet_input):
    # input N x T x c
    mean_hidden_states, std_hidden_states = hidden_states.mean(dim=(1,2), keepdim=True), hidden_states.std(dim=(1,2), keepdim=True)
    mean_controlnet_input, std_controlnet_input = controlnet_input.mean(dim=(1,2), keepdim=True), controlnet_input.std(dim=(1,2), keepdim=True)
    controlnet_input = (controlnet_input - mean_controlnet_input) * (std_hidden_states / (std_controlnet_input + 1e-12)) + mean_hidden_states
    return controlnet_input


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding with Mixtral->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, dtype=None, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class T2IFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size=[16, 1], out_channels=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True, dtype=dtype, device=device)
        self.scale_shift_table = nn.Parameter(torch.empty(2, hidden_size, dtype=dtype, device=device))
        self.out_channels = out_channels
        self.patch_size = patch_size

    def unpatchfy(
        self,
        hidden_states: torch.Tensor,
        width: int,
    ):
        # 4 unpatchify
        new_height, new_width = 1, hidden_states.size(1)
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], new_height, new_width, self.patch_size[0], self.patch_size[1], self.out_channels)
        ).contiguous()
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, new_height * self.patch_size[0], new_width * self.patch_size[1])
        ).contiguous()
        if width > new_width:
            output = torch.nn.functional.pad(output, (0, width - new_width, 0, 0), 'constant', 0)
        elif width < new_width:
            output = output[:, :, :, :width]
        return output

    def forward(self, x, t, output_length):
        shift, scale = (comfy.model_management.cast_to(self.scale_shift_table[None], device=t.device, dtype=t.dtype) + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # unpatchify
        output = self.unpatchfy(x, output_length)
        return output


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=16,
        width=4096,
        patch_size=(16, 1),
        in_channels=8,
        embed_dim=1152,
        bias=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        patch_size_h, patch_size_w = patch_size
        self.early_conv_layers = nn.Sequential(
            operations.Conv2d(in_channels, in_channels*256, kernel_size=patch_size, stride=patch_size, padding=0, bias=bias, dtype=dtype, device=device),
            operations.GroupNorm(num_groups=32, num_channels=in_channels*256, eps=1e-6, affine=True, dtype=dtype, device=device),
            operations.Conv2d(in_channels*256, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias, dtype=dtype, device=device)
        )
        self.patch_size = patch_size
        self.height, self.width = height // patch_size_h, width // patch_size_w
        self.base_size = self.width

    def forward(self, latent):
        # early convolutions, N x C x H x W -> N x 256 * sqrt(patch_size) x H/patch_size x W/patch_size
        latent = self.early_conv_layers(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return latent


class ACEStepTransformer2DModel(nn.Module):
    # _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: Optional[int] = 8,
        num_layers: int = 28,
        inner_dim: int = 1536,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        mlp_ratio: float = 4.0,
        out_channels: int = 8,
        max_position: int = 32768,
        rope_theta: float = 1000000.0,
        speaker_embedding_dim: int = 512,
        text_embedding_dim: int = 768,
        ssl_encoder_depths: List[int] = [9, 9],
        ssl_names: List[str] = ["mert", "m-hubert"],
        ssl_latent_dims: List[int] = [1024, 768],
        lyric_encoder_vocab_size: int = 6681,
        lyric_hidden_size: int = 1024,
        patch_size: List[int] = [16, 1],
        max_height: int = 16,
        max_width: int = 4096,
        audio_model=None,
        dtype=None, device=None, operations=None

    ):
        super().__init__()

        self.dtype = dtype
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.out_channels = out_channels
        self.max_position = max_position
        self.patch_size = patch_size

        self.rope_theta = rope_theta

        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=self.attention_head_dim,
            max_position_embeddings=self.max_position,
            base=self.rope_theta,
            dtype=dtype,
            device=device,
        )

        # 2. Define input layers
        self.in_channels = in_channels

        self.num_layers = num_layers
        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LinearTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    add_cross_attention=True,
                    add_cross_attention_dim=self.inner_dim,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for i in range(self.num_layers)
            ]
        )

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.t_block = nn.Sequential(nn.SiLU(), operations.Linear(self.inner_dim, 6 * self.inner_dim, bias=True, dtype=dtype, device=device))

        # speaker
        self.speaker_embedder = operations.Linear(speaker_embedding_dim, self.inner_dim, dtype=dtype, device=device)

        # genre
        self.genre_embedder = operations.Linear(text_embedding_dim, self.inner_dim, dtype=dtype, device=device)

        # lyric
        self.lyric_embs = operations.Embedding(lyric_encoder_vocab_size, lyric_hidden_size, dtype=dtype, device=device)
        self.lyric_encoder = LyricEncoder(input_size=lyric_hidden_size, static_chunk_size=0, dtype=dtype, device=device, operations=operations)
        self.lyric_proj = operations.Linear(lyric_hidden_size, self.inner_dim, dtype=dtype, device=device)

        projector_dim = 2 * self.inner_dim

        self.projectors = nn.ModuleList([
            nn.Sequential(
                operations.Linear(self.inner_dim, projector_dim, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Linear(projector_dim, projector_dim, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Linear(projector_dim, ssl_dim, dtype=dtype, device=device),
            ) for ssl_dim in ssl_latent_dims
        ])

        self.proj_in = PatchEmbed(
            height=max_height,
            width=max_width,
            patch_size=patch_size,
            embed_dim=self.inner_dim,
            bias=True,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.final_layer = T2IFinalLayer(self.inner_dim, patch_size=patch_size, out_channels=out_channels, dtype=dtype, device=device, operations=operations)

    def forward_lyric_encoder(
        self,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        out_dtype=None,
    ):
        # N x T x D
        lyric_embs = self.lyric_embs(lyric_token_idx, out_dtype=out_dtype)
        prompt_prenet_out, _mask = self.lyric_encoder(lyric_embs, lyric_mask, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        prompt_prenet_out = self.lyric_proj(prompt_prenet_out)
        return prompt_prenet_out

    def encode(
        self,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        lyrics_strength=1.0,
    ):

        bs = encoder_text_hidden_states.shape[0]
        device = encoder_text_hidden_states.device

        # speaker embedding
        encoder_spk_hidden_states = self.speaker_embedder(speaker_embeds).unsqueeze(1)

        # genre embedding
        encoder_text_hidden_states = self.genre_embedder(encoder_text_hidden_states)

        # lyric
        encoder_lyric_hidden_states = self.forward_lyric_encoder(
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
            out_dtype=encoder_text_hidden_states.dtype,
        )

        encoder_lyric_hidden_states *= lyrics_strength

        encoder_hidden_states = torch.cat([encoder_spk_hidden_states, encoder_text_hidden_states, encoder_lyric_hidden_states], dim=1)

        encoder_hidden_mask = None
        if text_attention_mask is not None:
            speaker_mask = torch.ones(bs, 1, device=device)
            encoder_hidden_mask = torch.cat([speaker_mask, text_attention_mask, lyric_mask], dim=1)

        return encoder_hidden_states, encoder_hidden_mask

    def decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_mask: torch.Tensor,
        timestep: Optional[torch.Tensor],
        output_length: int = 0,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        transformer_options={},
    ):
        embedded_timestep = self.timestep_embedder(self.time_proj(timestep).to(dtype=hidden_states.dtype))
        temb = self.t_block(embedded_timestep)

        hidden_states = self.proj_in(hidden_states)

        # controlnet logic
        if block_controlnet_hidden_states is not None:
            control_condi = cross_norm(hidden_states, block_controlnet_hidden_states)
            hidden_states = hidden_states + control_condi * controlnet_scale

        # inner_hidden_states = []

        rotary_freqs_cis = self.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])
        encoder_rotary_freqs_cis = self.rotary_emb(encoder_hidden_states, seq_len=encoder_hidden_states.shape[1])

        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_hidden_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                temb=temb,
                transformer_options=transformer_options,
            )

        output = self.final_layer(hidden_states, embedded_timestep, output_length)
        return output

    def forward(self,
        x,
        timestep,
        attention_mask=None,
        context: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        lyrics_strength=1.0,
        **kwargs
    ):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, kwargs.get("transformer_options", {}))
        ).execute(x, timestep, attention_mask, context, text_attention_mask, speaker_embeds, lyric_token_idx, lyric_mask, block_controlnet_hidden_states,
                  controlnet_scale, lyrics_strength, **kwargs)

    def _forward(
        self,
        x,
        timestep,
        attention_mask=None,
        context: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        lyrics_strength=1.0,
        **kwargs
    ):
        hidden_states = x
        encoder_text_hidden_states = context
        encoder_hidden_states, encoder_hidden_mask = self.encode(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
            lyrics_strength=lyrics_strength,
        )

        output_length = hidden_states.shape[-1]

        transformer_options = kwargs.get("transformer_options", {})
        output = self.decode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            timestep=timestep,
            output_length=output_length,
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            controlnet_scale=controlnet_scale,
            transformer_options=transformer_options,
        )

        return output
