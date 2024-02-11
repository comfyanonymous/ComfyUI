# pylint: skip-file
# type: ignore
import math
import random

import torch
from torch import nn

from .gfpganv1_arch import ResUpBlock
from .stylegan2_bilinear_arch import (
    ConvLayer,
    EqualConv2d,
    EqualLinear,
    ResBlock,
    ScaledLeakyReLU,
    StyleGAN2GeneratorBilinear,
)


class StyleGAN2GeneratorBilinearSFT(StyleGAN2GeneratorBilinear):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).
    It is the bilinear version. It does not use the complicated UpFirDnSmooth function that is not friendly for
    deployment. It can be easily converted to the clean version: StyleGAN2GeneratorCSFT.
    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(
        self,
        out_size,
        num_style_feat=512,
        num_mlp=8,
        channel_multiplier=2,
        lr_mlp=0.01,
        narrow=1,
        sft_half=False,
    ):
        super(StyleGAN2GeneratorBilinearSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            lr_mlp=lr_mlp,
            narrow=narrow,
        )
        self.sft_half = sft_half

    def forward(
        self,
        styles,
        conditions,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        truncation=1,
        truncation_latent=None,
        inject_index=None,
        return_latents=False,
    ):
        """Forward function for StyleGAN2GeneratorBilinearSFT.
        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [
                    getattr(self.noises, f"noise{i}") for i in range(self.num_layers)
                ]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = (
                styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            )
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2],
            self.style_convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
        ):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class GFPGANBilinear(nn.Module):
    """The GFPGAN architecture: Unet + StyleGAN2 decoder with SFT.
    It is the bilinear version and it does not use the complicated UpFirDnSmooth function that is not friendly for
    deployment. It can be easily converted to the clean version: GFPGANv1Clean.
    Ref: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.
    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        decoder_load_path (str): The path to the pre-trained decoder model (usually, the StyleGAN2). Default: None.
        fix_decoder (bool): Whether to fix the decoder. Default: True.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Whether to use different latent w for different layers. Default: False.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(
        self,
        out_size,
        num_style_feat=512,
        channel_multiplier=1,
        decoder_load_path=None,
        fix_decoder=True,
        # for stylegan decoder
        num_mlp=8,
        lr_mlp=0.01,
        input_is_latent=False,
        different_w=False,
        narrow=1,
        sft_half=False,
    ):
        super(GFPGANBilinear, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat
        self.min_size_restriction = 512

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            "4": int(512 * unet_narrow),
            "8": int(512 * unet_narrow),
            "16": int(512 * unet_narrow),
            "32": int(512 * unet_narrow),
            "64": int(256 * channel_multiplier * unet_narrow),
            "128": int(128 * channel_multiplier * unet_narrow),
            "256": int(64 * channel_multiplier * unet_narrow),
            "512": int(32 * channel_multiplier * unet_narrow),
            "1024": int(16 * channel_multiplier * unet_narrow),
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2 ** (int(math.log(out_size, 2)))

        self.conv_body_first = ConvLayer(
            3, channels[f"{first_out_size}"], 1, bias=True, activate=True
        )

        # downsample
        in_channels = channels[f"{first_out_size}"]
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f"{2**(i - 1)}"]
            self.conv_body_down.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.final_conv = ConvLayer(
            in_channels, channels["4"], 3, bias=True, activate=True
        )

        # upsample
        in_channels = channels["4"]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(
                EqualConv2d(
                    channels[f"{2**i}"],
                    3,
                    1,
                    stride=1,
                    padding=0,
                    bias=True,
                    bias_init_val=0,
                )
            )

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = EqualLinear(
            channels["4"] * 4 * 4,
            linear_out_channel,
            bias=True,
            bias_init_val=0,
            lr_mul=1,
            activation=None,
        )

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorBilinearSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            lr_mlp=lr_mlp,
            narrow=narrow,
            sft_half=sft_half,
        )

        # load pre-trained stylegan2 model if necessary
        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(
                    decoder_load_path, map_location=lambda storage, loc: storage
                )["params_ema"]
            )
        # fix decoder without updating params
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(
                        out_channels,
                        out_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(
                        out_channels,
                        sft_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=1,
                    ),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(
                        out_channels,
                        out_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(
                        out_channels,
                        sft_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                )
            )

    def forward(self, x, return_latents=False, return_rgb=True, randomize_noise=True):
        """Forward function for GFPGANBilinear.
        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        unet_skips = []
        out_rgbs = []

        # encoder
        feat = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)

        feat = self.final_conv(feat)

        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        # decoder
        image, _ = self.stylegan_decoder(
            [style_code],
            conditions,
            return_latents=return_latents,
            input_is_latent=self.input_is_latent,
            randomize_noise=randomize_noise,
        )

        return image, out_rgbs
