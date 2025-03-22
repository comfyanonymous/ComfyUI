#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit
from .common import pad_to_patch_size, rms_norm

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    Approximator,
    ModulationOut
)


@dataclass
class ChromaParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: int
    qkv_bias: bool
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int


class Chroma(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = ChromaParams(**kwargs)
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
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.time_in = MLPEmbedder(in_dim=64, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)
        # set as nn identity for now, will overwrite it later.
        self.distilled_guidance_layer = Approximator(
                    in_dim=self.in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                    n_layers=self.n_layers,
                    dtype=dtype, device=device, operations=operations
                )

        self.guidance_in = (
            MLPEmbedder(in_dim=64, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if self.distilled_guidance_layer else nn.Identity()
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

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False
    @staticmethod
    def distribute_modulations(tensor: torch.Tensor, single_block_count: int = 38, double_blocks_count: int = 19):
        """
        Distributes slices of the tensor into the block_dict as ModulationOut objects.

        Args:
            tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
        """
        batch_size, vectors, dim = tensor.shape

        block_dict = {}

        # HARD CODED VALUES! lookup table for the generated vectors
        # Add 38 single mod blocks
        for i in range(single_block_count):
            key = f"single_blocks.{i}.modulation.lin"
            block_dict[key] = None

        # Add 19 image double blocks
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.img_mod.lin"
            block_dict[key] = None

        # Add 19 text double blocks
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.txt_mod.lin"
            block_dict[key] = None

        # Add the final layer
        block_dict["final_layer.adaLN_modulation.1"] = None
        # # 6.2b version
        # block_dict["lite_double_blocks.4.img_mod.lin"] = None
        # block_dict["lite_double_blocks.4.txt_mod.lin"] = None


        idx = 0  # Index to keep track of the vector slices

        for key in block_dict.keys():
            if "single_blocks" in key:
                # Single block: 1 ModulationOut
                block_dict[key] = ModulationOut(
                    shift=tensor[:, idx:idx+1, :],
                    scale=tensor[:, idx+1:idx+2, :],
                    gate=tensor[:, idx+2:idx+3, :]
                )
                idx += 3  # Advance by 3 vectors

            elif "img_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block

            elif "txt_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block

            elif "final_layer" in key:
                # Final layer: 1 ModulationOut
                block_dict[key] = [
                    tensor[:, idx:idx+1, :],
                    tensor[:, idx+1:idx+2, :],
                ]
                idx += 2  # Advance by 2 vectors

            # elif "lite_double_blocks.4.img_mod" in key:
            #     # Double block: List of 2 ModulationOut
            #     double_block = []
            #     for _ in range(2):  # Create 2 ModulationOut objects
            #         double_block.append(
            #             ModulationOut(
            #                 shift=tensor[:, idx:idx+1, :],
            #                 scale=tensor[:, idx+1:idx+2, :],
            #                 gate=tensor[:, idx+2:idx+3, :]
            #             )
            #         )
            #         idx += 3  # Advance by 3 vectors per ModulationOut
            #     block_dict[key] = double_block

            # elif "lite_double_blocks.4.txt_mod" in key:
            #     # Double block: List of 2 ModulationOut
            #     double_block = []
            #     for _ in range(2):  # Create 2 ModulationOut objects
            #         double_block.append(
            #             ModulationOut(
            #                 shift=tensor[:, idx:idx+1, :],
            #                 scale=tensor[:, idx+1:idx+2, :],
            #                 gate=tensor[:, idx+2:idx+3, :]
            #             )
            #         )
            #         idx += 3  # Advance by 3 vectors per ModulationOut
            #     block_dict[key] = double_block

        return block_dict

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
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        # distilled vector guidance
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        # guidance = guidance *
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length), 32).to(img.device, img.dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        mod_vectors_dict = self.distribute_modulations(mod_vectors, 38, 19)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if i not in self.skip_mmdit:
                guidance_index = i
                # if lite we change block 4 guidance with lite guidance
                # and offset the guidance by 11 blocks after block 4
                # if self.lite and i == 4:
                #     img_mod = mod_vectors_dict[f"lite_double_blocks.4.img_mod.lin"]
                #     txt_mod = mod_vectors_dict[f"lite_double_blocks.4.txt_mod.lin"]
                # elif self.lite and i > 4:
                #     guidance_index = i + 11 
                #     img_mod = mod_vectors_dict[f"double_blocks.{guidance_index}.img_mod.lin"]
                #     txt_mod = mod_vectors_dict[f"double_blocks.{guidance_index}.txt_mod.lin"]
                # else:
                img_mod = mod_vectors_dict[f"double_blocks.{guidance_index}.img_mod.lin"]
                txt_mod = mod_vectors_dict[f"double_blocks.{guidance_index}.txt_mod.lin"]
                double_mod = [img_mod, txt_mod]

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
                single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
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
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
