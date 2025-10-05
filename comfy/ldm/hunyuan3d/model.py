import torch
from torch import nn
from comfy.ldm.flux.layers import (
    DoubleStreamBlock,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
import comfy.patcher_extension


class Hunyuan3Dv2(nn.Module):
    def __init__(
        self,
        in_channels=64,
        context_in_dim=1536,
        hidden_size=1024,
        mlp_ratio=4.0,
        num_heads=16,
        depth=16,
        depth_single_blocks=32,
        qkv_bias=True,
        guidance_embed=False,
        image_model=None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.dtype = dtype

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )

        self.max_period = 1000  # While reimplementing the model I noticed that they messed up. This 1000 value was meant to be the time_factor but they set the max_period instead
        self.latent_in = operations.Linear(in_channels, hidden_size, bias=True, dtype=dtype, device=device)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size, dtype=dtype, device=device, operations=operations)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=hidden_size, dtype=dtype, device=device, operations=operations) if guidance_embed else None
        )
        self.cond_in = operations.Linear(context_in_dim, hidden_size, dtype=dtype, device=device)
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(depth_single_blocks)
            ]
        )
        self.final_layer = LastLayer(hidden_size, 1, in_channels, dtype=dtype, device=device, operations=operations)

    def forward(self, x, timestep, context, guidance=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, guidance, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, guidance=None, transformer_options={}, **kwargs):
        x = x.movedim(-1, -2)
        timestep = 1.0 - timestep
        txt = context
        img = self.latent_in(x)

        vec = self.time_in(timestep_embedding(timestep, 256, self.max_period).to(dtype=img.dtype))
        if self.guidance_in is not None:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256, self.max_period).to(img.dtype))

        txt = self.cond_in(txt)
        pe = None
        attn_mask = None

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"),
                                                   transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask,
                                 transformer_options=transformer_options)

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"),
                                       transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

        img = img[:, txt.shape[1]:, ...]
        img = self.final_layer(img, vec)
        return img.movedim(-2, -1) * (-1.0)
