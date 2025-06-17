import torch
from torch import Tensor, nn

from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import (
    MLPEmbedder,
    RMSNorm,
    QKNorm,
    SelfAttention,
    ModulationOut,
)



class ChromaModulationOut(ModulationOut):
    @classmethod
    def from_offset(cls, tensor: torch.Tensor, offset: int = 0) -> ModulationOut:
        return cls(
            shift=tensor[:, offset : offset + 1, :],
            scale=tensor[:, offset + 1 : offset + 2, :],
            gate=tensor[:, offset + 2 : offset + 3, :],
        )




class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers = 5, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_proj = operations.Linear(in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim, dtype=dtype, device=device, operations=operations) for x in range( n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim, dtype=dtype, device=device, operations=operations) for x in range( n_layers)])
        self.out_proj = operations.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, flipped_img_txt=False, dtype=None, device=None, operations=None):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)

        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)

        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.flipped_img_txt = flipped_img_txt

    def forward(self, img: Tensor, txt: Tensor, pe: Tensor, vec: Tensor, attn_mask=None):
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        # prepare image for attention
        img_modulated = torch.addcmul(img_mod1.shift, 1 + img_mod1.scale, self.img_norm1(img))
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = torch.addcmul(txt_mod1.shift, 1 + txt_mod1.scale, self.txt_norm1(txt))
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        attn = attention(torch.cat((txt_q, img_q), dim=2),
                         torch.cat((txt_k, img_k), dim=2),
                         torch.cat((txt_v, img_v), dim=2),
                         pe=pe, mask=attn_mask)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img.addcmul_(img_mod1.gate, self.img_attn.proj(img_attn))
        img.addcmul_(img_mod2.gate, self.img_mlp(torch.addcmul(img_mod2.shift, 1 + img_mod2.scale, self.img_norm2(img))))

        # calculate the txt bloks
        txt.addcmul_(txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt.addcmul_(txt_mod2.gate, self.txt_mlp(torch.addcmul(txt_mod2.shift, 1 + txt_mod2.scale, self.txt_norm2(txt))))

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype, device=device)
        # proj and mlp_out
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device)

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor, pe: Tensor, vec: Tensor, attn_mask=None) -> Tensor:
        mod = vec
        x_mod = torch.addcmul(mod.shift, 1 + mod.scale, self.pre_norm(x))
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x.addcmul_(mod.gate, output)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = torch.addcmul(shift[:, None, :], 1 + scale[:, None, :], self.norm_final(x))
        x = self.linear(x)
        return x
