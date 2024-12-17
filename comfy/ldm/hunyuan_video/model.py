#Based on Flux code because of weird hunyuan video code license.

import torch
import comfy.ldm.flux.layers
import comfy.ldm.modules.diffusionmodules.mmdit
from comfy.ldm.modules.attention import optimized_attention


from dataclasses import dataclass
from einops import repeat

from torch import Tensor, nn

from comfy.ldm.flux.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding
)

import comfy.ldm.common_dit


@dataclass
class HunyuanVideoParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: list
    qkv_bias: bool
    guidance_embed: bool


class SelfAttentionRef(nn.Module):
    def __init__(self, dim: int, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        super().__init__()
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)


class TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.heads = heads
        mlp_hidden_dim = hidden_size * 4

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device),
        )

        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = SelfAttentionRef(hidden_size, True, dtype=dtype, device=device, operations=operations)

        self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)

        self.mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x, c, mask):
        mod1, mod2 = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn.qkv(norm_x)
        q, k, v = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attn = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True)

        x = x + self.self_attn.proj(attn) * mod1.unsqueeze(1)
        x = x + self.mlp(self.norm2(x)) * mod2.unsqueeze(1)
        return x


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads,
        num_blocks,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads=heads,
                    dtype=dtype,
                    device=device,
                    operations=operations
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, c, mask):
        m = None
        if mask is not None:
            m = mask.view(mask.shape[0], 1, 1, mask.shape[1]).repeat(1, 1, mask.shape[1], 1)
            m = m + m.transpose(2, 3)

        for block in self.blocks:
            x = block(x, c, m)
        return x



class TokenRefiner(nn.Module):
    def __init__(
        self,
        text_dim,
        hidden_size,
        heads,
        num_blocks,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()

        self.input_embedder = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        self.t_embedder = MLPEmbedder(256, hidden_size, dtype=dtype, device=device, operations=operations)
        self.c_embedder = MLPEmbedder(text_dim, hidden_size, dtype=dtype, device=device, operations=operations)
        self.individual_token_refiner = IndividualTokenRefiner(hidden_size, heads, num_blocks, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        x,
        timesteps,
        mask,
    ):
        t = self.t_embedder(timestep_embedding(timesteps, 256, time_factor=1.0).to(x.dtype))
        # m = mask.float().unsqueeze(-1)
        # c = (x.float() * m).sum(dim=1) / m.sum(dim=1) #TODO: the following works when the x.shape is the same length as the tokens but might break otherwise
        c = x.sum(dim=1) / x.shape[1]

        c = t + self.c_embedder(c.to(x.dtype))
        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, mask)
        return x

class HunyuanVideo(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = HunyuanVideoParams(**kwargs)
        self.params = params
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.img_in = comfy.ldm.modules.diffusionmodules.mmdit.PatchEmbed(None, self.patch_size, self.in_channels, self.hidden_size, conv3d=True, dtype=dtype, device=device, operations=operations)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )

        self.txt_in = TokenRefiner(params.context_in_dim, self.hidden_size, self.num_heads, 2, dtype=dtype, device=device, operations=operations)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    flipped_img_txt=True,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, self.patch_size[-1], self.out_channels, dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((img, txt), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"])
                    return out

                out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, : img_len] += add

        img = img[:, : img_len]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape)
        return img

    def forward(self, x, timestep, context, y, guidance, attention_mask=None, control=None, transformer_options={}, **kwargs):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(x, img_ids, context, txt_ids, attention_mask, timestep, y, guidance, control, transformer_options)
        return out
