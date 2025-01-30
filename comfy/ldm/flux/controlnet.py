#Original code can be found on: https://github.com/XLabs-AI/x-flux/blob/main/src/flux/controlnet.py
#modified to support different types of flux controlnets

import torch
import math
from torch import Tensor, nn
from einops import rearrange, repeat

from .layers import (timestep_embedding)

from .model import Flux
import comfy.ldm.common_dit

class MistolineCondDownsamplBlock(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.encoder = nn.Sequential(
            operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.encoder(x)

class MistolineControlnetBlock(nn.Module):
    def __init__(self, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.linear(x))


class ControlNetFlux(Flux):
    def __init__(self, latent_input=False, num_union_modes=0, mistoline=False, control_latent_channels=None, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__(final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)

        self.main_model_double = 19
        self.main_model_single = 38

        self.mistoline = mistoline
        # add ControlNet blocks
        if self.mistoline:
            control_block = lambda : MistolineControlnetBlock(self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            control_block = lambda : operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(self.params.depth):
            self.controlnet_blocks.append(control_block())

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(self.params.depth_single_blocks):
            self.controlnet_single_blocks.append(control_block())

        self.num_union_modes = num_union_modes
        self.controlnet_mode_embedder = None
        if self.num_union_modes > 0:
            self.controlnet_mode_embedder = operations.Embedding(self.num_union_modes, self.hidden_size, dtype=dtype, device=device)

        self.gradient_checkpointing = False
        self.latent_input = latent_input
        if control_latent_channels is None:
            control_latent_channels = self.in_channels
        else:
            control_latent_channels *= 2 * 2 #patch size

        self.pos_embed_input = operations.Linear(control_latent_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        if not self.latent_input:
            if self.mistoline:
                self.input_cond_block = MistolineCondDownsamplBlock(dtype=dtype, device=device, operations=operations)
            else:
                self.input_hint_block = nn.Sequential(
                    operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device)
                )

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control_type: Tensor = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        if self.controlnet_mode_embedder is not None and len(control_type) > 0:
            control_cond = self.controlnet_mode_embedder(torch.tensor(control_type, device=img.device), out_dtype=img.dtype).unsqueeze(0).repeat((txt.shape[0], 1, 1))
            txt = torch.cat([control_cond, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:,:1], txt_ids], dim=1)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        controlnet_double = ()

        for i in range(len(self.double_blocks)):
            img, txt = self.double_blocks[i](img=img, txt=txt, vec=vec, pe=pe)
            controlnet_double = controlnet_double + (self.controlnet_blocks[i](img),)

        img = torch.cat((txt, img), 1)

        controlnet_single = ()

        for i in range(len(self.single_blocks)):
            img = self.single_blocks[i](img, vec=vec, pe=pe)
            controlnet_single = controlnet_single + (self.controlnet_single_blocks[i](img[:, txt.shape[1] :, ...]),)

        repeat = math.ceil(self.main_model_double / len(controlnet_double))
        if self.latent_input:
            out_input = ()
            for x in controlnet_double:
                    out_input += (x,) * repeat
        else:
            out_input = (controlnet_double * repeat)

        out = {"input": out_input[:self.main_model_double]}
        if len(controlnet_single) > 0:
            repeat = math.ceil(self.main_model_single / len(controlnet_single))
            out_output = ()
            if self.latent_input:
                for x in controlnet_single:
                        out_output += (x,) * repeat
            else:
                out_output = (controlnet_single * repeat)
            out["output"] = out_output[:self.main_model_single]
        return out

    def forward(self, x, timesteps, context, y, guidance=None, hint=None, **kwargs):
        patch_size = 2
        if self.latent_input:
            hint = comfy.ldm.common_dit.pad_to_patch_size(hint, (patch_size, patch_size))
        elif self.mistoline:
            hint = hint * 2.0 - 1.0
            hint = self.input_cond_block(hint)
        else:
            hint = hint * 2.0 - 1.0
            hint = self.input_hint_block(hint)

        hint = rearrange(hint, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        return self.forward_orig(img, img_ids, hint, context, txt_ids, timesteps, y, guidance, control_type=kwargs.get("control_type", []))
