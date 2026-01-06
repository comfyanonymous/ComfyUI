
import torch
import torch.nn as nn

import comfy.ops
from comfy.ldm.modules.diffusionmodules.mmdit import Mlp, TimestepEmbedder, PatchEmbed
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from torch.utils import checkpoint

from .attn_layers import Attention, CrossAttention
from .poolers import AttentionPool
from .posemb_layers import get_2d_rotary_pos_embed, get_fill_resize_and_crop

def calc_rope(x, patch_size, head_size):
    th = (x.shape[2] + (patch_size // 2)) // patch_size
    tw = (x.shape[3] + (patch_size // 2)) // patch_size
    base_size = 512 // 8 // patch_size
    start, stop = get_fill_resize_and_crop((th, tw), base_size)
    sub_args = [start, stop, (th, tw)]
    # head_size = HUNYUAN_DIT_CONFIG['DiT-g/2']['hidden_size'] // HUNYUAN_DIT_CONFIG['DiT-g/2']['num_heads']
    rope = get_2d_rotary_pos_embed(head_size, *sub_args)
    rope = (rope[0].to(x), rope[1].to(x))
    return rope


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class HunYuanDiTBlock(nn.Module):
    """
    A HunYuanDiT block with `add` conditioning.
    """
    def __init__(self,
                 hidden_size,
                 c_emb_size,
                 num_heads,
                 mlp_ratio=4.0,
                 text_states_dim=1024,
                 qk_norm=False,
                 norm_type="layer",
                 skip=False,
                 attn_precision=None,
                 dtype=None,
                 device=None,
                 operations=None,
                 ):
        super().__init__()
        use_ele_affine = True

        if norm_type == "layer":
            norm_layer = operations.LayerNorm
        elif norm_type == "rms":
            norm_layer = operations.RMSNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6, dtype=dtype, device=device)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6, dtype=dtype, device=device)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0, dtype=dtype, device=device, operations=operations)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(c_emb_size, hidden_size, bias=True, dtype=dtype, device=device)
        )

        # ========================= Cross-Attention =========================
        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True,
                                        qk_norm=qk_norm, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = norm_layer(2 * hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            self.skip_linear = operations.Linear(2 * hidden_size, hidden_size, dtype=dtype, device=device)
        else:
            self.skip_linear = None

        self.gradient_checkpointing = False

    def _forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1)
            if cat.dtype != x.dtype:
                cat = cat.to(x.dtype)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        attn_inputs = (
            self.norm1(x) + shift_msa, freq_cis_img,
        )
        x = x + self.attn1(*attn_inputs)[0]

        # Cross-Attention
        cross_inputs = (
            self.norm3(x), text_states, freq_cis_img
        )
        x = x + self.attn2(*cross_inputs)[0]

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)

        return x

    def forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward, x, c, text_states, freq_cis_img, skip)
        return self._forward(x, c, text_states, freq_cis_img, skip)


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """
    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(c_emb_size, 2 * final_hidden_size, bias=True, dtype=dtype, device=device)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class HunYuanDiT(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """
    #@register_to_config
    def __init__(self,
                 input_size: tuple = 32,
                 patch_size: int = 2,
                 in_channels: int = 4,
                 hidden_size: int = 1152,
                 depth: int = 28,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 text_states_dim = 1024,
                 text_states_dim_t5 = 2048,
                 text_len = 77,
                 text_len_t5 = 256,
                 qk_norm = True,# See http://arxiv.org/abs/2302.05442 for details.
                 size_cond = False,
                 use_style_cond = False,
                 learn_sigma = True,
                 norm = "layer",
                 log_fn: callable = print,
                 attn_precision=None,
                 dtype=None,
                 device=None,
                 operations=None,
                 **kwargs,
    ):
        super().__init__()
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = text_states_dim
        self.text_states_dim_t5 = text_states_dim_t5
        self.text_len = text_len
        self.text_len_t5 = text_len_t5
        self.size_cond = size_cond
        self.use_style_cond = use_style_cond
        self.norm = norm
        self.dtype = dtype
        #import pdb
        #pdb.set_trace()

        self.mlp_t5 = nn.Sequential(
            operations.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True, dtype=dtype, device=device),
        )
        # learnable replace
        self.text_embedding_padding = nn.Parameter(
            torch.empty(self.text_len + self.text_len_t5, self.text_states_dim, dtype=dtype, device=device))

        # Attention pooling
        pooler_out_dim = 1024
        self.pooler = AttentionPool(self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=pooler_out_dim, dtype=dtype, device=device, operations=operations)

        # Dimension of the extra input vectors
        self.extra_in_dim = pooler_out_dim

        if self.size_cond:
            # Image size and crop size conditions
            self.extra_in_dim += 6 * 256

        if self.use_style_cond:
            # Here we use a default learned embedder layer for future extension.
            self.style_embedder = operations.Embedding(1, hidden_size, dtype=dtype, device=device)
            self.extra_in_dim += hidden_size

        # Text embedding for `add`
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, dtype=dtype, device=device, operations=operations)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device, operations=operations)
        self.extra_embedder = nn.Sequential(
            operations.Linear(self.extra_in_dim, hidden_size * 4, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size * 4, hidden_size, bias=True, dtype=dtype, device=device),
        )

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=self.text_states_dim,
                            qk_norm=qk_norm,
                            norm_type=self.norm,
                            skip=layer > depth // 2,
                            attn_precision=attn_precision,
                            dtype=dtype,
                            device=device,
                            operations=operations,
                            )
            for layer in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)
        self.unpatchify_channels = self.out_channels



    def forward(self,
                x,
                t,
                context,#encoder_hidden_states=None,
                text_embedding_mask=None,
                encoder_hidden_states_t5=None,
                text_embedding_mask_t5=None,
                image_meta_size=None,
                style=None,
                return_dict=False,
                control=None,
                transformer_options={},
                ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """
        patches_replace = transformer_options.get("patches_replace", {})
        encoder_hidden_states = context
        text_states = encoder_hidden_states                     # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5               # 2,256,2048
        text_states_mask = text_embedding_mask.bool()           # 2,77
        text_states_t5_mask = text_embedding_mask_t5.bool()     # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)

        padding = comfy.ops.cast_to_input(self.text_embedding_padding, text_states)

        text_states[:,-self.text_len:] = torch.where(text_states_mask[:,-self.text_len:].unsqueeze(2), text_states[:,-self.text_len:], padding[:self.text_len])
        text_states_t5[:,-self.text_len_t5:] = torch.where(text_states_t5_mask[:,-self.text_len_t5:].unsqueeze(2), text_states_t5[:,-self.text_len_t5:], padding[self.text_len:])

        text_states = torch.cat([text_states, text_states_t5], dim=1)  # 2,205，1024
        # clip_t5_mask = torch.cat([text_states_mask, text_states_t5_mask], dim=-1)

        _, _, oh, ow = x.shape
        th, tw = (oh + (self.patch_size // 2)) // self.patch_size, (ow + (self.patch_size // 2)) // self.patch_size


        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = calc_rope(x, self.patch_size, self.hidden_size // self.num_heads) #(cos_cis_img, sin_cis_img)

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t, dtype=x.dtype)
        x = self.x_embedder(x)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        # Build image meta size tokens if applicable
        if self.size_cond:
            image_meta_size = timestep_embedding(image_meta_size.view(-1), 256).to(x.dtype)   # [B * 6, 256]
            image_meta_size = image_meta_size.view(-1, 6 * 256)
            extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)  # [B, D + 6 * 256]

        # Build style tokens
        if self.use_style_cond:
            if style is None:
                style = torch.zeros((extra_vec.shape[0],), device=x.device, dtype=torch.int)
            style_embedding = self.style_embedder(style, out_dtype=x.dtype)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        blocks_replace = patches_replace.get("dit", {})

        controls = None
        if control:
            controls = control.get("output", None)
        # ========================= Forward pass through HunYuanDiT blocks =========================
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.depth // 2:
                if controls is not None:
                    skip = skips.pop() + controls.pop().to(dtype=x.dtype)
                else:
                    skip = skips.pop()
            else:
                skip = None

            if ("double_block", layer) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], args["vec"], args["txt"], args["pe"], args["skip"])
                    return out

                out = blocks_replace[("double_block", layer)]({"img": x, "txt": text_states, "vec": c, "pe": freqs_cis_img, "skip": skip}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, c, text_states, freqs_cis_img, skip)   # (N, L, D)


            if layer < (self.depth // 2 - 1):
                skips.append(x)
        if controls is not None and len(controls) != 0:
            raise ValueError("The number of controls is not equal to the number of skip connections.")

        # ========================= Final layer =========================
        x = self.final_layer(x, c)                              # (N, L, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, th, tw)                          # (N, out_channels, H, W)

        if return_dict:
            return {'x': x}
        if self.learn_sigma:
            return x[:,:self.out_channels // 2,:oh,:ow]
        return x[:,:,:oh,:ow]

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
