# Credits:
# Original Flux code can be found on: https://github.com/black-forest-labs/flux
# Chroma Radiance adaption referenced from https://github.com/lodestone-rock/flow

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import repeat
import comfy.ldm.common_dit

from comfy.ldm.flux.layers import (
    EmbedND,
    timestep_embedding,
)

from .layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    Approximator,
)
from .layers_dct import NerfEmbedder, NerfGLUBlock, NerfFinalLayer

from . import model as chroma_model

@dataclass
class ChromaRadianceParams(chroma_model.ChromaParams):
    patch_size: int
    nerf_hidden_size: int
    nerf_mlp_ratio: int
    nerf_depth: int
    nerf_max_freqs: int


class ChromaRadiance(chroma_model.Chroma):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        nn.Module.__init__(self)
        self.dtype = dtype
        params = ChromaRadianceParams(**kwargs)
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
        self.in_dim = params.in_dim
        self.out_dim = params.out_dim
        self.hidden_dim = params.hidden_dim
        self.n_layers = params.n_layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in_patch = operations.Conv2d(
            params.in_channels,
            params.hidden_size,
            kernel_size=params.patch_size,
            stride=params.patch_size,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)
        # set as nn identity for now, will overwrite it later.
        self.distilled_guidance_layer = Approximator(
                    in_dim=self.in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                    n_layers=self.n_layers,
                    dtype=dtype, device=device, operations=operations
                )


        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
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

        # pixel channel concat with DCT
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                dtype=dtype,
                device=device,
                operations=operations,
            ) for _ in range(params.nerf_depth)
        ])

        self.nerf_final_layer = NerfFinalLayer(
            params.nerf_hidden_size,
            out_channels=params.in_channels,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 4:
            raise ValueError("Input img tensor must be in [B, C, H, W] format.")
        if txt.ndim != 3:
            raise ValueError("Input txt tensors must have 3 dimensions.")
        B, C, H, W = img.shape

        # gemini gogogo idk how to unfold and pack the patch properly :P
        # Store the raw pixel values of each patch for the NeRF head later.
        # unfold creates patches: [B, C * P * P, NumPatches]
        nerf_pixels = nn.functional.unfold(img, kernel_size=self.params.patch_size, stride=self.params.patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2) # -> [B, NumPatches, C * P * P]

        # partchify ops
        img = self.img_in_patch(img) # -> [B, Hidden, H/P, W/P]
        num_patches = img.shape[2] * img.shape[3]
        # flatten into a sequence for the transformer.
        img = img.flatten(2).transpose(1, 2) # -> [B, NumPatches, Hidden]

        # distilled vector guidance
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        # guidance = guidance *
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                       txt=args["txt"],
                                                       vec=args["vec"],
                                                       pe=args["pe"],
                                                       attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img,
                                                               "txt": txt,
                                                               "vec": double_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                     txt=txt,
                                     vec=double_mod,
                                     pe=pe,
                                     attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img,
                                                               "vec": single_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]
        # aliasing
        nerf_hidden = img
        # reshape for per-patch processing
        nerf_hidden = nerf_hidden.reshape(B * num_patches, self.params.hidden_size)
        nerf_pixels = nerf_pixels.reshape(B * num_patches, C, self.params.patch_size**2).transpose(1, 2)

        # get DCT-encoded pixel embeddings [pixel-dct]
        img_dct = self.nerf_image_embedder(nerf_pixels)

        # pass through the dynamic MLP blocks (the NeRF)
        for i, block in enumerate(self.nerf_blocks):
            img_dct = block(img_dct, nerf_hidden)

        # final projection to get the output pixel values
        img_dct = self.nerf_final_layer(img_dct) # -> [B*NumPatches, P*P, C]

        # gemini gogogo idk how to fold this properly :P
        # Reassemble the patches into the final image.
        img_dct = img_dct.transpose(1, 2) # -> [B*NumPatches, C, P*P]
        # Reshape to combine with batch dimension for fold
        img_dct = img_dct.reshape(B, num_patches, -1) # -> [B, NumPatches, C*P*P]
        img_dct = img_dct.transpose(1, 2) # -> [B, C*P*P, NumPatches]
        img_dct = nn.functional.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=self.params.patch_size,
            stride=self.params.patch_size
        )

        return img_dct

    def forward(self, x, timestep, context, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        img = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        h_len = ((h + (self.patch_size // 2)) // self.patch_size)
        w_len = ((w + (self.patch_size // 2)) // self.patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        return self.forward_orig(img, img_ids, context, txt_ids, timestep, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
