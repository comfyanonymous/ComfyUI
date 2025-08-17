import re
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn import Module, Linear, init
from typing import Any, Mapping

from .PixArt import PixArt, get_2d_sincos_pos_embed
from .PixArtMS import PixArtMSBlock, PixArtMS
from .utils import auto_grad_checkpoint

# The implementation of ControlNet-Half architrecture
# https://github.com/lllyasviel/ControlNet/discussions/188
class ControlT2IDitBlockHalf(Module):
    def __init__(self, base_block: PixArtMSBlock, block_index: 0) -> None:
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        
        self.hidden_size = hidden_size = base_block.hidden_size
        if self.block_index == 0:
            self.before_proj = Linear(hidden_size, hidden_size)
            init.zeros_(self.before_proj.weight)
            init.zeros_(self.before_proj.bias)
        self.after_proj = Linear(hidden_size, hidden_size) 
        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, mask=None, c=None):
        
        if self.block_index == 0:
            # the first block
            c = self.before_proj(c)
            c = self.copied_block(x + c, y, t, mask)
            c_skip = self.after_proj(c)
        else:
            # load from previous c and produce the c for skip connection
            c = self.copied_block(c, y, t, mask)
            c_skip = self.after_proj(c)
        
        return c, c_skip
        

# The implementation of ControlPixArtHalf net
class ControlPixArtHalf(Module):
    # only support single res model
    def __init__(self, base_model: PixArt, copy_blocks_num: int = 13) -> None:
        super().__init__()
        self.dtype = torch.get_default_dtype()
        self.base_model = base_model.eval()
        self.controlnet = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.blocks)
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Copy first copy_blocks_num block
        for i in range(copy_blocks_num):
            self.controlnet.append(ControlT2IDitBlockHalf(base_model.blocks[i], i))
        self.controlnet = nn.ModuleList(self.controlnet)
    
    def __getattr__(self, name: str) -> Tensor or Module:
        if name in ['forward', 'forward_with_dpmsolver', 'forward_with_cfg', 'forward_c', 'load_state_dict']:
            return self.__dict__[name]
        elif name in ['base_model', 'controlnet']:
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    def forward_c(self, c):
        self.h, self.w = c.shape[-2]//self.patch_size, c.shape[-1]//self.patch_size
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(c.device).to(self.dtype)
        return self.x_embedder(c) + pos_embed if c is not None else c

    # def forward(self, x, t, c, **kwargs):
    #     return self.base_model(x, t, c=self.forward_c(c), **kwargs)
    def forward_raw(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        # modify the original PixArtMS forward function
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = auto_grad_checkpoint(self.controlnet[index - 1], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        
            # update x
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward(self, x, timesteps, context, cn_hint=None, **kwargs):
        """
        Forward pass that adapts comfy input to original forward function
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timesteps: (N,) tensor of diffusion timesteps
        context: (N, 1, 120, C) conditioning
        cn_hint: controlnet hint
        """
        ## Still accepts the input w/o that dim but returns garbage
        if len(context.shape) == 3:
            context = context.unsqueeze(1)

        ## run original forward pass
        out = self.forward_raw(
            x = x.to(self.dtype),
            timestep = timesteps.to(self.dtype),
            y = context.to(self.dtype),
            c = cn_hint,
        )

        ## only return EPS
        out = out.to(torch.float)
        eps, rest = out[:, :self.in_channels], out[:, self.in_channels:]
        return eps

    def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
        model_out = self.forward_raw(x, t, y, data_info=data_info, c=c, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    # def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
    #     return self.base_model.forward_with_dpmsolver(x, t, y, data_info=data_info, c=self.forward_c(c), **kwargs)

    def forward_with_cfg(self, x, t, y, cfg_scale, data_info, c, **kwargs):
        return self.base_model.forward_with_cfg(x, t, y, cfg_scale, data_info, c=self.forward_c(c), **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if all((k.startswith('base_model') or k.startswith('controlnet')) for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict)
        else:
            new_key = {}
            for k in state_dict.keys():
                new_key[k] = re.sub(r"(blocks\.\d+)(.*)", r"\1.base_block\2", k)
            for k, v in new_key.items():
                if k != v:
                    print(f"replace {k} to {v}")
                    state_dict[v] = state_dict.pop(k)

            return self.base_model.load_state_dict(state_dict, strict)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    # @property
    # def dtype(self):
        ## 返回模型参数的数据类型
        # return next(self.parameters()).dtype


# The implementation for PixArtMS_Half + 1024 resolution
class ControlPixArtMSHalf(ControlPixArtHalf):
    # support multi-scale res model (multi-scale model can also be applied to single reso training & inference)
    def __init__(self, base_model: PixArtMS, copy_blocks_num: int = 13) -> None:
        super().__init__(base_model=base_model, copy_blocks_num=copy_blocks_num)

    def forward_raw(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        # modify the original PixArtMS forward function
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size

        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(x.device).to(self.dtype)
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        csize = self.csize_embedder(c_size, bs)  # (N, D)
        ar = self.ar_embedder(ar, bs)  # (N, D)
        t = t + torch.cat([csize, ar], dim=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = auto_grad_checkpoint(self.controlnet[index - 1], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        
            # update x
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward(self, x, timesteps, context, img_hw=None, aspect_ratio=None, cn_hint=None, **kwargs):
        """
        Forward pass that adapts comfy input to original forward function
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timesteps: (N,) tensor of diffusion timesteps
        context: (N, 1, 120, C) conditioning
        img_hw: height|width conditioning
        aspect_ratio: aspect ratio conditioning
        cn_hint: controlnet hint
        """
        ## size/ar from cond with fallback based on the latent image shape.
        bs = x.shape[0]
        data_info = {}
        if img_hw is None:
            data_info["img_hw"] = torch.tensor(
                [[x.shape[2]*8, x.shape[3]*8]],
                dtype=self.dtype,
                device=x.device
            ).repeat(bs, 1)
        else:
            data_info["img_hw"] = img_hw.to(x.dtype)
        if aspect_ratio is None or True:
            data_info["aspect_ratio"] = torch.tensor(
                [[x.shape[2]/x.shape[3]]],
                dtype=self.dtype,
                device=x.device
            ).repeat(bs, 1)
        else:
            data_info["aspect_ratio"] = aspect_ratio.to(x.dtype)

        ## Still accepts the input w/o that dim but returns garbage
        if len(context.shape) == 3:
            context = context.unsqueeze(1)

        ## run original forward pass
        out = self.forward_raw(
            x = x.to(self.dtype),
            timestep = timesteps.to(self.dtype),
            y = context.to(self.dtype),
            c = cn_hint,
            data_info=data_info,
        )

        ## only return EPS
        out = out.to(torch.float)
        eps, rest = out[:, :self.in_channels], out[:, self.in_channels:]
        return eps
