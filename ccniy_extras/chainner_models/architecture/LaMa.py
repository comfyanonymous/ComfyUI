# pylint: skip-file
"""
Model adapted from advimman's lama project: https://github.com/advimman/lama
"""

# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, rotate


class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(
                self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x)
            )
        else:
            raise ValueError(f"Unexpected input type {type(x)}")

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode="reflect")
        x_padded_rotated = rotate(
            x_padded, self.angle.to(x_padded), InterpolationMode.BILINEAR, fill=0
        )

        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(
            y_padded_rotated,
            -self.angle.to(y_padded_rotated),
            InterpolationMode.BILINEAR,
            fill=0,
        )
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        spatial_scale_factor=None,
        spatial_scale_mode="bilinear",
        spectral_pos_encoding=False,
        use_se=False,
        se_kwargs=None,
        ffc3d=False,
        fft_norm="ortho",
    ):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        half_check = False
        if x.type() == "torch.cuda.HalfTensor":
            # half only works on gpu anyway
            half_check = True

        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(
                x,
                scale_factor=self.spatial_scale_factor,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        if half_check == True:
            ffted = torch.fft.rfftn(
                x.float(), dim=fft_dim, norm=self.fft_norm
            )  # .type(torch.cuda.HalfTensor)
        else:
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)

        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = (
                torch.linspace(0, 1, height)[None, None, :, None]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            coords_hor = (
                torch.linspace(0, 1, width)[None, None, None, :]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        if half_check == True:
            ffted = self.conv_layer(ffted.half())  # (batch, c*2, h, w/2+1)
        else:
            ffted = self.conv_layer(
                ffted
            )  # .type(torch.cuda.FloatTensor)  # (batch, c*2, h, w/2+1)

        ffted = self.relu(self.bn(ffted))
        # forcing to be always float
        ffted = ffted.float()

        ffted = (
            ffted.view(
                (
                    batch,
                    -1,
                    2,
                )
                + ffted.size()[2:]
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # (batch,c, t, h, w/2+1, 2)

        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        if half_check == True:
            output = output.half()

        if self.spatial_scale_factor is not None:
            output = F.interpolate(
                output,
                size=orig_size,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        return output


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        enable_lfu=True,
        separable_fu=False,
        **fu_kwargs,
    ):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        fu_class = FourierUnit
        self.fu = fu_class(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = fu_class(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            _, c, h, _ = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(
                torch.split(x[:, : c // 4], split_s, dim=-2), dim=1
            ).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        enable_lfu=True,
        padding_type="reflect",
        gated=False,
        **spectral_kwargs,
    ):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(
            in_cl,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(
            in_cl,
            out_cg,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(
            in_cg,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg,
            out_cg,
            stride,
            1 if groups == 1 else groups // 2,
            enable_lfu,
            **spectral_kwargs,
        )

        self.gated = gated
        module = (
            nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        )
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
        padding_type="reflect",
        enable_lfu=True,
        **kwargs,
    ):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            padding_type=padding_type,
            **kwargs,
        )
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        activation_layer=nn.ReLU,
        dilation=1,
        spatial_transform_kwargs=None,
        inline=False,
        **conv_kwargs,
    ):
        super().__init__()
        self.conv1 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        self.conv2 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(
                self.conv1, **spatial_transform_kwargs
            )
            self.conv2 = LearnableSpatialTransformWrapper(
                self.conv2, **spatial_transform_kwargs
            )
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = (
                x[:, : -self.conv1.ffc.global_in_num],
                x[:, -self.conv1.ffc.global_in_num :],
            )
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
        activation_layer=nn.ReLU,
        up_norm_layer=nn.BatchNorm2d,
        up_activation=nn.ReLU(True),
        init_conv_kwargs={},
        downsample_conv_kwargs={},
        resnet_conv_kwargs={},
        spatial_transform_layers=None,
        spatial_transform_kwargs={},
        max_features=1024,
        out_ffc=False,
        out_ffc_kwargs={},
    ):
        assert n_blocks >= 0
        super().__init__()
        """
        init_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False}
        downsample_conv_kwargs = {'ratio_gin': '${generator.init_conv_kwargs.ratio_gout}', 'ratio_gout': '${generator.downsample_conv_kwargs.ratio_gin}', 'enable_lfu': False}
        resnet_conv_kwargs = {'ratio_gin': 0.75, 'ratio_gout': '${generator.resnet_conv_kwargs.ratio_gin}', 'enable_lfu': False}
        spatial_transform_kwargs = {}
        out_ffc_kwargs = {}
        """
        """
        print(input_nc, output_nc, ngf, n_downsampling, n_blocks, norm_layer,
                padding_type, activation_layer,
                up_norm_layer, up_activation,
                spatial_transform_layers,
                add_out_act, max_features, out_ffc, file=sys.stderr)

        4 3 64 3 18 <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
        reflect <class 'torch.nn.modules.activation.ReLU'>
        <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
        ReLU(inplace=True)
        None sigmoid 1024 False
        """
        init_conv_kwargs = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
        downsample_conv_kwargs = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
        resnet_conv_kwargs = {
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
            "enable_lfu": False,
        }
        spatial_transform_kwargs = {}
        out_ffc_kwargs = {}

        model = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                input_nc,
                ngf,
                kernel_size=7,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                **init_conv_kwargs,
            ),
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [
                FFC_BN_ACT(
                    min(max_features, ngf * mult),
                    min(max_features, ngf * mult * 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    **cur_conv_kwargs,
                )
            ]

        mult = 2**n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(
                feats_num_bottleneck,
                padding_type=padding_type,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                **resnet_conv_kwargs,
            )
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(
                    cur_resblock, **spatial_transform_kwargs
                )
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(max_features, ngf * mult),
                    min(max_features, int(ngf * mult / 2)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                up_norm_layer(min(max_features, int(ngf * mult / 2))),
                up_activation,
            ]

        if out_ffc:
            model += [
                FFCResnetBlock(
                    ngf,
                    padding_type=padding_type,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    inline=True,
                    **out_ffc_kwargs,
                )
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        ]
        model.append(nn.Sigmoid())
        self.model = nn.Sequential(*model)

    def forward(self, image, mask):
        return self.model(torch.cat([image, mask], dim=1))


class LaMa(nn.Module):
    def __init__(self, state_dict) -> None:
        super(LaMa, self).__init__()
        self.model_arch = "LaMa"
        self.sub_type = "Inpaint"
        self.in_nc = 4
        self.out_nc = 3
        self.scale = 1

        self.min_size = None
        self.pad_mod = 8
        self.pad_to_square = False

        self.model = FFCResNetGenerator(self.in_nc, self.out_nc)
        self.state = {
            k.replace("generator.model", "model.model"): v
            for k, v in state_dict.items()
        }

        self.supports_fp16 = False
        self.support_bf16 = True

        self.load_state_dict(self.state, strict=False)

    def forward(self, img, mask):
        masked_img = img * (1 - mask)
        inpainted_mask = mask * self.model.forward(masked_img, mask)
        result = inpainted_mask + (1 - mask) * img
        return result
