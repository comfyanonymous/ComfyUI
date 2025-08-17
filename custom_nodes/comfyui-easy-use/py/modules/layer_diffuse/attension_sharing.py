# Currently only sd15

import functools
import torch
import einops

from comfy import model_management, utils
from comfy.ldm.modules.attention import optimized_attention

module_mapping_sd15 = {
    0: "input_blocks.1.1.transformer_blocks.0.attn1",
    1: "input_blocks.1.1.transformer_blocks.0.attn2",
    2: "input_blocks.2.1.transformer_blocks.0.attn1",
    3: "input_blocks.2.1.transformer_blocks.0.attn2",
    4: "input_blocks.4.1.transformer_blocks.0.attn1",
    5: "input_blocks.4.1.transformer_blocks.0.attn2",
    6: "input_blocks.5.1.transformer_blocks.0.attn1",
    7: "input_blocks.5.1.transformer_blocks.0.attn2",
    8: "input_blocks.7.1.transformer_blocks.0.attn1",
    9: "input_blocks.7.1.transformer_blocks.0.attn2",
    10: "input_blocks.8.1.transformer_blocks.0.attn1",
    11: "input_blocks.8.1.transformer_blocks.0.attn2",
    12: "output_blocks.3.1.transformer_blocks.0.attn1",
    13: "output_blocks.3.1.transformer_blocks.0.attn2",
    14: "output_blocks.4.1.transformer_blocks.0.attn1",
    15: "output_blocks.4.1.transformer_blocks.0.attn2",
    16: "output_blocks.5.1.transformer_blocks.0.attn1",
    17: "output_blocks.5.1.transformer_blocks.0.attn2",
    18: "output_blocks.6.1.transformer_blocks.0.attn1",
    19: "output_blocks.6.1.transformer_blocks.0.attn2",
    20: "output_blocks.7.1.transformer_blocks.0.attn1",
    21: "output_blocks.7.1.transformer_blocks.0.attn2",
    22: "output_blocks.8.1.transformer_blocks.0.attn1",
    23: "output_blocks.8.1.transformer_blocks.0.attn2",
    24: "output_blocks.9.1.transformer_blocks.0.attn1",
    25: "output_blocks.9.1.transformer_blocks.0.attn2",
    26: "output_blocks.10.1.transformer_blocks.0.attn1",
    27: "output_blocks.10.1.transformer_blocks.0.attn2",
    28: "output_blocks.11.1.transformer_blocks.0.attn1",
    29: "output_blocks.11.1.transformer_blocks.0.attn2",
    30: "middle_block.1.transformer_blocks.0.attn1",
    31: "middle_block.1.transformer_blocks.0.attn2",
}


def compute_cond_mark(cond_or_uncond, sigmas):
    cond_or_uncond_size = int(sigmas.shape[0])

    cond_mark = []
    for cx in cond_or_uncond:
        cond_mark += [cx] * cond_or_uncond_size

    cond_mark = torch.Tensor(cond_mark).to(sigmas)
    return cond_mark


class LoRALinearLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 256, org=None):
        super().__init__()
        self.down = torch.nn.Linear(in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_features, bias=False)
        self.org = [org]

    def forward(self, h):
        org_weight = self.org[0].weight.to(h)
        org_bias = self.org[0].bias.to(h) if self.org[0].bias is not None else None
        down_weight = self.down.weight
        up_weight = self.up.weight
        final_weight = org_weight + torch.mm(up_weight, down_weight)
        return torch.nn.functional.linear(h, final_weight, org_bias)


class AttentionSharingUnit(torch.nn.Module):
    # `transformer_options` passed to the most recent BasicTransformerBlock.forward
    # call.
    transformer_options: dict = {}

    def __init__(self, module, frames=2, use_control=True, rank=256):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = (
            module.to_q.in_features,
            module.to_q.out_features,
        )
        k_in_channels, k_out_channels = (
            module.to_k.in_features,
            module.to_k.out_features,
        )
        v_in_channels, v_out_channels = (
            module.to_v.in_features,
            module.to_v.out_features,
        )
        o_in_channels, o_out_channels = (
            module.to_out[0].in_features,
            module.to_out[0].out_features,
        )

        hidden_size = k_out_channels

        self.to_q_lora = [
            LoRALinearLayer(q_in_channels, q_out_channels, rank, module.to_q)
            for _ in range(self.frames)
        ]
        self.to_k_lora = [
            LoRALinearLayer(k_in_channels, k_out_channels, rank, module.to_k)
            for _ in range(self.frames)
        ]
        self.to_v_lora = [
            LoRALinearLayer(v_in_channels, v_out_channels, rank, module.to_v)
            for _ in range(self.frames)
        ]
        self.to_out_lora = [
            LoRALinearLayer(o_in_channels, o_out_channels, rank, module.to_out[0])
            for _ in range(self.frames)
        ]

        self.to_q_lora = torch.nn.ModuleList(self.to_q_lora)
        self.to_k_lora = torch.nn.ModuleList(self.to_k_lora)
        self.to_v_lora = torch.nn.ModuleList(self.to_v_lora)
        self.to_out_lora = torch.nn.ModuleList(self.to_out_lora)

        self.temporal_i = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.temporal_n = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6
        )
        self.temporal_q = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.temporal_k = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.temporal_v = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.temporal_o = torch.nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )

        self.control_convs = None

        if use_control:
            self.control_convs = [
                torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(256, hidden_size, kernel_size=1),
                )
                for _ in range(self.frames)
            ]
            self.control_convs = torch.nn.ModuleList(self.control_convs)

        self.control_signals = None

    def forward(self, h, context=None, value=None):
        transformer_options = self.transformer_options

        modified_hidden_states = einops.rearrange(
            h, "(b f) d c -> f b d c", f=self.frames
        )

        if self.control_convs is not None:
            context_dim = int(modified_hidden_states.shape[2])
            control_outs = []
            for f in range(self.frames):
                control_signal = self.control_signals[context_dim].to(
                    modified_hidden_states
                )
                control = self.control_convs[f](control_signal)
                control = einops.rearrange(control, "b c h w -> b (h w) c")
                control_outs.append(control)
            control_outs = torch.stack(control_outs, dim=0)
            modified_hidden_states = modified_hidden_states + control_outs.to(
                modified_hidden_states
            )

        if context is None:
            framed_context = modified_hidden_states
        else:
            framed_context = einops.rearrange(
                context, "(b f) d c -> f b d c", f=self.frames
            )

        framed_cond_mark = einops.rearrange(
            compute_cond_mark(
                transformer_options["cond_or_uncond"],
                transformer_options["sigmas"],
            ),
            "(b f) -> f b",
            f=self.frames,
        ).to(modified_hidden_states)

        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]

            if context is not None:
                cond_overwrite = transformer_options.get("cond_overwrite", [])
                if len(cond_overwrite) > f:
                    cond_overwrite = cond_overwrite[f]
                else:
                    cond_overwrite = None
                if cond_overwrite is not None:
                    cond_mark = framed_cond_mark[f][:, None, None]
                    fcf = cond_overwrite.to(fcf) * (1.0 - cond_mark) + fcf * cond_mark

            q = self.to_q_lora[f](modified_hidden_states[f])
            k = self.to_k_lora[f](fcf)
            v = self.to_v_lora[f](fcf)
            o = optimized_attention(q, k, v, self.heads)
            o = self.to_out_lora[f](o)
            o = self.original_module[0].to_out[1](o)
            attn_outs.append(o)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(
            modified_hidden_states
        )
        modified_hidden_states = einops.rearrange(
            modified_hidden_states, "f b d c -> (b f) d c", f=self.frames
        )

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        q = self.temporal_q(x)
        k = self.temporal_k(x)
        v = self.temporal_v(x)

        x = optimized_attention(q, k, v, self.heads)
        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        return modified_hidden_states - h

    @classmethod
    def hijack_transformer_block(cls):
        def register_get_transformer_options(func):
            @functools.wraps(func)
            def forward(self, x, context=None, transformer_options={}):
                cls.transformer_options = transformer_options
                return func(self, x, context, transformer_options)

            return forward

        from comfy.ldm.modules.attention import BasicTransformerBlock

        BasicTransformerBlock.forward = register_get_transformer_options(
            BasicTransformerBlock.forward
        )


AttentionSharingUnit.hijack_transformer_block()


class AdditionalAttentionCondsEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks_0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 64*64*256

        self.blocks_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 32*32*256

        self.blocks_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 16*16*256

        self.blocks_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 8*8*256

        self.blks = [self.blocks_0, self.blocks_1, self.blocks_2, self.blocks_3]

    def __call__(self, h):
        results = {}
        for b in self.blks:
            h = b(h)
            results[int(h.shape[2]) * int(h.shape[3])] = h
        return results


class HookerLayers(torch.nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layers = torch.nn.ModuleList(layer_list)


class AttentionSharingPatcher(torch.nn.Module):
    def __init__(self, unet, frames=2, use_control=True, rank=256):
        super().__init__()
        model_management.unload_model_clones(unet)

        units = []
        for i in range(32):
            real_key = module_mapping_sd15[i]
            attn_module = utils.get_attr(unet.model.diffusion_model, real_key)
            u = AttentionSharingUnit(
                attn_module, frames=frames, use_control=use_control, rank=rank
            )
            units.append(u)
            unet.add_object_patch("diffusion_model." + real_key, u)

        self.hookers = HookerLayers(units)

        if use_control:
            self.kwargs_encoder = AdditionalAttentionCondsEncoder()
        else:
            self.kwargs_encoder = None

        self.dtype = torch.float32
        if model_management.should_use_fp16(model_management.get_torch_device()):
            self.dtype = torch.float16
            self.hookers.half()
        return

    def set_control(self, img):
        img = img.cpu().float() * 2.0 - 1.0
        signals = self.kwargs_encoder(img)
        for m in self.hookers.layers:
            m.control_signals = signals
        return
