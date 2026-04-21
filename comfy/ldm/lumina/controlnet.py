import torch
from torch import nn

from .model import JointTransformerBlock

class ZImageControlTransformerBlock(JointTransformerBlock):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=0,
        operation_settings=None,
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm, modulation, z_image_modulation=True, operation_settings=operation_settings)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = operation_settings.get("operations").Linear(self.dim, self.dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.after_proj = operation_settings.get("operations").Linear(self.dim, self.dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c_skip, c

class ZImage_Control(torch.nn.Module):
    def __init__(
        self,
        dim: int = 3840,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = (8.0 / 3.0),
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        n_control_layers=6,
        control_in_dim=16,
        additional_in_dim=0,
        broken=False,
        refiner_control=False,
        dtype=None,
        device=None,
        operations=None,
        **kwargs
    ):
        super().__init__()
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        self.broken = broken
        self.additional_in_dim = additional_in_dim
        self.control_in_dim = control_in_dim
        n_refiner_layers = 2
        self.n_control_layers = n_control_layers
        self.control_layers = nn.ModuleList(
            [
                ZImageControlTransformerBlock(
                    i,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    block_id=i,
                    operation_settings=operation_settings,
                )
                for i in range(self.n_control_layers)
            ]
        )

        all_x_embedder = {}
        patch_size = 2
        f_patch_size = 1
        x_embedder = operations.Linear(f_patch_size * patch_size * patch_size * (self.control_in_dim + self.additional_in_dim), dim, bias=True, device=device, dtype=dtype)
        all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

        self.refiner_control = refiner_control

        self.control_all_x_embedder = nn.ModuleDict(all_x_embedder)
        if self.refiner_control:
            self.control_noise_refiner = nn.ModuleList(
                [
                    ZImageControlTransformerBlock(
                        layer_id,
                        dim,
                        n_heads,
                        n_kv_heads,
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps,
                        qk_norm,
                        block_id=layer_id,
                        operation_settings=operation_settings,
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )
        else:
            self.control_noise_refiner = nn.ModuleList(
                [
                    JointTransformerBlock(
                        layer_id,
                        dim,
                        n_heads,
                        n_kv_heads,
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps,
                        qk_norm,
                        modulation=True,
                        z_image_modulation=True,
                        operation_settings=operation_settings,
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )

    def forward(self, cap_feats, control_context, x_freqs_cis, adaln_input):
        patch_size = 2
        f_patch_size = 1
        pH = pW = patch_size
        B, C, H, W = control_context.shape
        control_context = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

        x_attn_mask = None
        if not self.refiner_control:
            for layer in self.control_noise_refiner:
                control_context = layer(control_context, x_attn_mask, x_freqs_cis[:control_context.shape[0], :control_context.shape[1]], adaln_input)

        return control_context

    def forward_noise_refiner_block(self, layer_id, control_context, x, x_attn_mask, x_freqs_cis, adaln_input):
        if self.refiner_control:
            if self.broken:
                if layer_id == 0:
                    return self.control_layers[layer_id](control_context, x, x_mask=x_attn_mask, freqs_cis=x_freqs_cis[:control_context.shape[0], :control_context.shape[1]], adaln_input=adaln_input)
                if layer_id > 0:
                    out = None
                    for i in range(1, len(self.control_layers)):
                        o, control_context = self.control_layers[i](control_context, x, x_mask=x_attn_mask, freqs_cis=x_freqs_cis[:control_context.shape[0], :control_context.shape[1]], adaln_input=adaln_input)
                        if out is None:
                            out = o

                    return (out, control_context)
            else:
                return self.control_noise_refiner[layer_id](control_context, x, x_mask=x_attn_mask, freqs_cis=x_freqs_cis[:control_context.shape[0], :control_context.shape[1]], adaln_input=adaln_input)
        else:
            return (None, control_context)

    def forward_control_block(self, layer_id, control_context, x, x_attn_mask, x_freqs_cis, adaln_input):
        return self.control_layers[layer_id](control_context, x, x_mask=x_attn_mask, freqs_cis=x_freqs_cis[:control_context.shape[0], :control_context.shape[1]], adaln_input=adaln_input)
