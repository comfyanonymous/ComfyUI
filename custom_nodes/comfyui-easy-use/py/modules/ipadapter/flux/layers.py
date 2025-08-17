import torch
from torch import Tensor, nn

from .math import attention
from ..attention_processor import IPAFluxAttnProcessor2_0
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from comfy import model_management as mm

class DoubleStreamBlockIPA(nn.Module):
    def __init__(self, original_block: DoubleStreamBlock, ip_adapter, image_emb):
        super().__init__()

        mlp_hidden_dim = original_block.img_mlp[0].out_features
        mlp_ratio = mlp_hidden_dim / original_block.hidden_size
        mlp_hidden_dim = int(original_block.hidden_size * mlp_ratio)
        self.num_heads = original_block.num_heads
        self.hidden_size = original_block.hidden_size
        self.img_mod = original_block.img_mod
        self.img_norm1 = original_block.img_norm1
        self.img_attn = original_block.img_attn

        self.img_norm2 = original_block.img_norm2
        self.img_mlp = original_block.img_mlp

        self.txt_mod = original_block.txt_mod
        self.txt_norm1 = original_block.txt_norm1
        self.txt_attn = original_block.txt_attn

        self.txt_norm2 = original_block.txt_norm2
        self.txt_mlp = original_block.txt_mlp
        self.flipped_img_txt = original_block.flipped_img_txt

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = mm.get_torch_device()

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, t: Tensor, attn_mask=None):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        if self.flipped_img_txt:
            # run actual attention
            attn = attention(torch.cat((img_q, txt_q), dim=2),
                             torch.cat((img_k, txt_k), dim=2),
                             torch.cat((img_v, txt_v), dim=2),
                             pe=pe, mask=attn_mask)

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
        else:
            # run actual attention
            attn = attention(torch.cat((txt_q, img_q), dim=2),
                             torch.cat((txt_k, img_k), dim=2),
                             torch.cat((txt_v, img_v), dim=2),
                             pe=pe, mask=attn_mask)

            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        for adapter, image in zip(self.ip_adapter, self.image_emb):
            # this does a separate attention for each adapter
            ip_hidden_states = adapter(self.num_heads, img_q, image, t)
            if ip_hidden_states is not None:
                ip_hidden_states = ip_hidden_states.to(self.device)
                img_attn = img_attn + ip_hidden_states

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class SingleStreamBlockIPA(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, original_block: SingleStreamBlock, ip_adapter, image_emb):
        super().__init__()
        self.hidden_dim = original_block.hidden_size
        self.num_heads = original_block.num_heads
        self.scale = original_block.scale

        self.mlp_hidden_dim = original_block.mlp_hidden_dim
        # qkv and mlp_in
        self.linear1 = original_block.linear1
        # proj and mlp_out
        self.linear2 = original_block.linear2

        self.norm = original_block.norm

        self.hidden_size = original_block.hidden_size
        self.pre_norm = original_block.pre_norm

        self.mlp_act = original_block.mlp_act
        self.modulation = original_block.modulation

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = mm.get_torch_device()

    def add_adapter(self, ip_adapter: IPAFluxAttnProcessor2_0, image_emb):
        self.ip_adapter.append(ip_adapter)
        self.image_emb.append(image_emb)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, t: Tensor, attn_mask=None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask)

        for adapter, image in zip(self.ip_adapter, self.image_emb):
            # this does a separate attention for each adapter
            # maybe we want a single joint attention call for all adapters?
            ip_hidden_states = adapter(self.num_heads, q, image, t)
            if ip_hidden_states is not None:
                ip_hidden_states = ip_hidden_states.to(self.device)
                attn = attn + ip_hidden_states

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x