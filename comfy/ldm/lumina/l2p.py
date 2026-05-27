import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.patcher_extension
from .model import NextDiT


class MicroDiffusionModel(nn.Module):
    """L2P pixel-space U-Net decoder."""

    def __init__(self, in_channels: int, si_t_hidden_size: int,
                 dtype=None, device=None, operations=None):
        super().__init__()
        conv = operations.Conv2d

        def conv_silu(c_in, c_out, k, p):
            return nn.Sequential(
                conv(c_in, c_out, kernel_size=k, padding=p, dtype=dtype, device=device),
                nn.SiLU(),
            )

        self.enc1 = conv_silu(in_channels, 64, 3, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = conv_silu(64, 128, 3, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = conv_silu(128, 256, 3, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.enc4 = conv_silu(256, 512, 3, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.bottleneck = nn.Sequential(
            conv(512 + si_t_hidden_size, 512, kernel_size=1, dtype=dtype, device=device),
            nn.SiLU(),
        )

        def up_block(c_in, c_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv(c_in, c_out, kernel_size=3, padding=1, dtype=dtype, device=device),
            )

        self.up4 = up_block(512, 512)
        self.dec4 = conv_silu(512 + 512, 256, 3, 1)
        self.up3 = up_block(256, 256)
        self.dec3 = conv_silu(256 + 256, 128, 3, 1)
        self.up2 = up_block(128, 128)
        self.dec2 = conv_silu(128 + 128, 64, 3, 1)
        self.up1 = up_block(64, 64)
        self.dec1 = conv_silu(64 + 64, 64, 3, 1)

        self.out_conv = conv(64, in_channels, kernel_size=1, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        enc1_out = self.enc1(x)
        p1 = self.pool1(enc1_out)
        enc2_out = self.enc2(p1)
        p2 = self.pool2(enc2_out)
        enc3_out = self.enc3(p2)
        p3 = self.pool3(enc3_out)
        enc4_out = self.enc4(p3)
        p4 = self.pool4(enc4_out)

        if c.shape[-2:] != p4.shape[-2:]:
            c = F.interpolate(c, size=p4.shape[-2:], mode='nearest')
        bottleneck_out = self.bottleneck(torch.cat([p4, c.to(p4.dtype)], dim=1))

        d4 = self.up4(bottleneck_out)
        d4 = self.dec4(torch.cat([d4, enc4_out], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, enc3_out], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, enc2_out], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, enc1_out], dim=1))

        return self.out_conv(d1)


class NextDiTL2P(NextDiT):
    """Z-Image DiT backbone with the L2P pixel-space U-Net decoder.

    Sibling of :class:`NextDiTPixelSpace`. Same backbone, RoPE, refiners, and
    text-conditioning path. Two head-level differences:
      * decoder architecture: an image-wide convolutional U-Net consuming the
        noisy RGB plus a ``(B, dim, H/P, W/P)`` feature map, rather than a
        per-patch SimpleMLPAdaLN.
      * output parameterization: L2P's decoder predicts ``x_0 - noise``
        (the negative flow-matching velocity), so ``_forward`` returns
        ``-decoded = noise - x_0 = v`` and ``forward`` passes it through
        directly. NextDiTPixelSpace's DCT decoder predicts ``-x_0`` and its
        ``forward`` reconstructs v via ``(x + img_out)/sigma``; that wrapper
        does NOT apply here.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.final_layer

        self.local_decoder = MicroDiffusionModel(
            in_channels=kwargs.get("in_channels", 3),
            si_t_hidden_size=kwargs.get("dim", 3840),
            dtype=kwargs.get("dtype"),
            device=kwargs.get("device"),
            operations=kwargs.get("operations"),
        )

    def _forward(self, x, timesteps, context, num_tokens, attention_mask=None,
                 ref_latents=[], ref_contexts=[], siglip_feats=[],
                 transformer_options={}, **kwargs):
        if len(ref_latents) > 0:
            timesteps = torch.cat([timesteps * 0, timesteps], dim=0)

        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        _, _, h_pad, w_pad = x.shape

        t = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t

        if self.clip_text_pooled_proj is not None:
            pooled = kwargs.get("clip_text_pooled", None)
            if pooled is not None:
                pooled = self.clip_text_pooled_proj(pooled)
            else:
                pooled = torch.zeros((x.shape[0], self.clip_text_dim), device=x.device, dtype=x.dtype)
            adaln_input = self.time_text_embed(torch.cat((t, pooled), dim=-1))

        patches = transformer_options.get("patches", {})
        img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = self.patchify_and_embed(
            x, cap_feats, cap_mask, adaln_input, num_tokens,
            ref_latents=ref_latents, ref_contexts=ref_contexts,
            siglip_feats=siglip_feats, transformer_options=transformer_options,
        )
        freqs_cis = freqs_cis.to(img.device)

        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "double"
        img_input = img
        for i, layer in enumerate(self.layers):
            transformer_options["block_index"] = i
            img = layer(img, mask, freqs_cis, adaln_input,
                        timestep_zero_index=timestep_zero_index,
                        transformer_options=transformer_options)
            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": img[:, cap_size[0]:], "img_input": img_input[:, cap_size[0]:],
                             "txt": img[:, :cap_size[0]], "pe": freqs_cis[:, cap_size[0]:],
                             "vec": adaln_input, "x": x, "block_index": i,
                             "transformer_options": transformer_options})
                    if "img" in out:
                        img[:, cap_size[0]:] = out["img"]
                    if "txt" in out:
                        img[:, :cap_size[0]] = out["txt"]

        # Build the conditioning feature map from real (unpadded) image tokens.
        pH = self.patch_size
        Ht, Wt = h_pad // pH, w_pad // pH
        n_real = Ht * Wt
        begin = cap_size[0]

        img_hidden = img[:, begin:begin + n_real, :]
        feat_map = img_hidden.reshape(bs, Ht, Wt, self.dim).permute(0, 3, 1, 2).contiguous()

        decoded = self.local_decoder(x, feat_map)
        decoded = decoded[:, :, :h, :w]
        return -decoded

    def forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                kwargs.get("transformer_options", {}),
            ),
        ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)
