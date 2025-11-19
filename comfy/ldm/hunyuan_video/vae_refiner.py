import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.modules.diffusionmodules.model import ResnetBlock, AttnBlock, VideoConv3d, Normalize
import comfy.ops
import comfy.ldm.models.autoencoder
ops = comfy.ops.disable_weight_init

class RMS_norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        shape = (dim, 1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.empty(shape))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma

class DnSmpl(nn.Module):
    def __init__(self, ic, oc, tds=True, refiner_vae=True, op=VideoConv3d):
        super().__init__()
        fct = 2 * 2 * 2 if tds else 1 * 2 * 2
        assert oc % fct == 0
        self.conv = op(ic, oc // fct, kernel_size=3, stride=1, padding=1)
        self.refiner_vae = refiner_vae

        self.tds = tds
        self.gs = fct * ic // oc

    def forward(self, x):
        r1 = 2 if self.tds else 1
        h = self.conv(x)

        if self.tds and self.refiner_vae:
            hf = h[:, :, :1, :, :]
            b, c, f, ht, wd = hf.shape
            hf = hf.reshape(b, c, f, ht // 2, 2, wd // 2, 2)
            hf = hf.permute(0, 4, 6, 1, 2, 3, 5)
            hf = hf.reshape(b, 2 * 2 * c, f, ht // 2, wd // 2)
            hf = torch.cat([hf, hf], dim=1)

            hn = h[:, :, 1:, :, :]
            b, c, frms, ht, wd = hn.shape
            nf = frms // r1
            hn = hn.reshape(b, c, nf, r1, ht // 2, 2, wd // 2, 2)
            hn = hn.permute(0, 3, 5, 7, 1, 2, 4, 6)
            hn = hn.reshape(b, r1 * 2 * 2 * c, nf, ht // 2, wd // 2)

            h = torch.cat([hf, hn], dim=2)

            xf = x[:, :, :1, :, :]
            b, ci, f, ht, wd = xf.shape
            xf = xf.reshape(b, ci, f, ht // 2, 2, wd // 2, 2)
            xf = xf.permute(0, 4, 6, 1, 2, 3, 5)
            xf = xf.reshape(b, 2 * 2 * ci, f, ht // 2, wd // 2)
            B, C, T, H, W = xf.shape
            xf = xf.view(B, h.shape[1], self.gs // 2, T, H, W).mean(dim=2)

            xn = x[:, :, 1:, :, :]
            b, ci, frms, ht, wd = xn.shape
            nf = frms // r1
            xn = xn.reshape(b, ci, nf, r1, ht // 2, 2, wd // 2, 2)
            xn = xn.permute(0, 3, 5, 7, 1, 2, 4, 6)
            xn = xn.reshape(b, r1 * 2 * 2 * ci, nf, ht // 2, wd // 2)
            B, C, T, H, W = xn.shape
            xn = xn.view(B, h.shape[1], self.gs, T, H, W).mean(dim=2)
            sc = torch.cat([xf, xn], dim=2)
        else:
            b, c, frms, ht, wd = h.shape

            nf = frms // r1
            h = h.reshape(b, c, nf, r1, ht // 2, 2, wd // 2, 2)
            h = h.permute(0, 3, 5, 7, 1, 2, 4, 6)
            h = h.reshape(b, r1 * 2 * 2 * c, nf, ht // 2, wd // 2)

            b, ci, frms, ht, wd = x.shape
            nf = frms // r1
            sc = x.reshape(b, ci, nf, r1, ht // 2, 2, wd // 2, 2)
            sc = sc.permute(0, 3, 5, 7, 1, 2, 4, 6)
            sc = sc.reshape(b, r1 * 2 * 2 * ci, nf, ht // 2, wd // 2)
            B, C, T, H, W = sc.shape
            sc = sc.view(B, h.shape[1], self.gs, T, H, W).mean(dim=2)

        return h + sc


class UpSmpl(nn.Module):
    def __init__(self, ic, oc, tus=True, refiner_vae=True, op=VideoConv3d):
        super().__init__()
        fct = 2 * 2 * 2 if tus else 1 * 2 * 2
        self.conv = op(ic, oc * fct, kernel_size=3, stride=1, padding=1)
        self.refiner_vae = refiner_vae

        self.tus = tus
        self.rp = fct * oc // ic

    def forward(self, x):
        r1 = 2 if self.tus else 1
        h = self.conv(x)

        if self.tus and self.refiner_vae:
            hf = h[:, :, :1, :, :]
            b, c, f, ht, wd = hf.shape
            nc = c // (2 * 2)
            hf = hf.reshape(b, 2, 2, nc, f, ht, wd)
            hf = hf.permute(0, 3, 4, 5, 1, 6, 2)
            hf = hf.reshape(b, nc, f, ht * 2, wd * 2)
            hf = hf[:, : hf.shape[1] // 2]

            hn = h[:, :, 1:, :, :]
            b, c, frms, ht, wd = hn.shape
            nc = c // (r1 * 2 * 2)
            hn = hn.reshape(b, r1, 2, 2, nc, frms, ht, wd)
            hn = hn.permute(0, 4, 5, 1, 6, 2, 7, 3)
            hn = hn.reshape(b, nc, frms * r1, ht * 2, wd * 2)

            h = torch.cat([hf, hn], dim=2)

            xf = x[:, :, :1, :, :]
            b, ci, f, ht, wd = xf.shape
            xf = xf.repeat_interleave(repeats=self.rp // 2, dim=1)
            b, c, f, ht, wd = xf.shape
            nc = c // (2 * 2)
            xf = xf.reshape(b, 2, 2, nc, f, ht, wd)
            xf = xf.permute(0, 3, 4, 5, 1, 6, 2)
            xf = xf.reshape(b, nc, f, ht * 2, wd * 2)

            xn = x[:, :, 1:, :, :]
            xn = xn.repeat_interleave(repeats=self.rp, dim=1)
            b, c, frms, ht, wd = xn.shape
            nc = c // (r1 * 2 * 2)
            xn = xn.reshape(b, r1, 2, 2, nc, frms, ht, wd)
            xn = xn.permute(0, 4, 5, 1, 6, 2, 7, 3)
            xn = xn.reshape(b, nc, frms * r1, ht * 2, wd * 2)
            sc = torch.cat([xf, xn], dim=2)
        else:
            b, c, frms, ht, wd = h.shape
            nc = c // (r1 * 2 * 2)
            h = h.reshape(b, r1, 2, 2, nc, frms, ht, wd)
            h = h.permute(0, 4, 5, 1, 6, 2, 7, 3)
            h = h.reshape(b, nc, frms * r1, ht * 2, wd * 2)

            sc = x.repeat_interleave(repeats=self.rp, dim=1)
            b, c, frms, ht, wd = sc.shape
            nc = c // (r1 * 2 * 2)
            sc = sc.reshape(b, r1, 2, 2, nc, frms, ht, wd)
            sc = sc.permute(0, 4, 5, 1, 6, 2, 7, 3)
            sc = sc.reshape(b, nc, frms * r1, ht * 2, wd * 2)

        return h + sc

class Encoder(nn.Module):
    def __init__(self, in_channels, z_channels, block_out_channels, num_res_blocks,
                 ffactor_spatial, ffactor_temporal, downsample_match_channel=True, refiner_vae=True, **_):
        super().__init__()
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.ffactor_temporal = ffactor_temporal

        self.refiner_vae = refiner_vae
        if self.refiner_vae:
            conv_op = VideoConv3d
            norm_op = RMS_norm
        else:
            conv_op = ops.Conv3d
            norm_op = Normalize

        self.conv_in = conv_op(in_channels, block_out_channels[0], 3, 1, 1)

        self.down = nn.ModuleList()
        ch = block_out_channels[0]
        depth = (ffactor_spatial >> 1).bit_length()
        depth_temporal = ((ffactor_spatial // self.ffactor_temporal) >> 1).bit_length()

        for i, tgt in enumerate(block_out_channels):
            stage = nn.Module()
            stage.block = nn.ModuleList([ResnetBlock(in_channels=ch if j == 0 else tgt,
                                                     out_channels=tgt,
                                                     temb_channels=0,
                                                     conv_op=conv_op, norm_op=norm_op)
                                        for j in range(num_res_blocks)])
            ch = tgt
            if i < depth:
                nxt = block_out_channels[i + 1] if i + 1 < len(block_out_channels) and downsample_match_channel else ch
                stage.downsample = DnSmpl(ch, nxt, tds=i >= depth_temporal, refiner_vae=self.refiner_vae, op=conv_op)
                ch = nxt
            self.down.append(stage)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=conv_op, norm_op=norm_op)
        self.mid.attn_1 = AttnBlock(ch, conv_op=ops.Conv3d, norm_op=norm_op)
        self.mid.block_2 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=conv_op, norm_op=norm_op)

        self.norm_out = norm_op(ch)
        self.conv_out = conv_op(ch, z_channels << 1, 3, 1, 1)

        self.regul = comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer()

    def forward(self, x):
        if not self.refiner_vae and x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)

        x = self.conv_in(x)

        for stage in self.down:
            for blk in stage.block:
                x = blk(x)
            if hasattr(stage, 'downsample'):
                x = stage.downsample(x)

        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))

        b, c, t, h, w = x.shape
        grp = c // (self.z_channels << 1)
        skip = x.view(b, c // grp, grp, t, h, w).mean(2)

        out = self.conv_out(F.silu(self.norm_out(x))) + skip

        if self.refiner_vae:
            out = self.regul(out)[0]

            out = torch.cat((out[:, :, :1], out), dim=2)
            out = out.permute(0, 2, 1, 3, 4)
            b, f_times_2, c, h, w = out.shape
            out = out.reshape(b, f_times_2 // 2, 2 * c, h, w)
            out = out.permute(0, 2, 1, 3, 4).contiguous()

        return out

class Decoder(nn.Module):
    def __init__(self, z_channels, out_channels, block_out_channels, num_res_blocks,
                 ffactor_spatial, ffactor_temporal, upsample_match_channel=True, refiner_vae=True, **_):
        super().__init__()
        block_out_channels = block_out_channels[::-1]
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        self.refiner_vae = refiner_vae
        if self.refiner_vae:
            conv_op = VideoConv3d
            norm_op = RMS_norm
        else:
            conv_op = ops.Conv3d
            norm_op = Normalize

        ch = block_out_channels[0]
        self.conv_in = conv_op(z_channels, ch, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=conv_op, norm_op=norm_op)
        self.mid.attn_1 = AttnBlock(ch, conv_op=ops.Conv3d, norm_op=norm_op)
        self.mid.block_2 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=conv_op, norm_op=norm_op)

        self.up = nn.ModuleList()
        depth = (ffactor_spatial >> 1).bit_length()
        depth_temporal = (ffactor_temporal >> 1).bit_length()

        for i, tgt in enumerate(block_out_channels):
            stage = nn.Module()
            stage.block = nn.ModuleList([ResnetBlock(in_channels=ch if j == 0 else tgt,
                                                     out_channels=tgt,
                                                     temb_channels=0,
                                                     conv_op=conv_op, norm_op=norm_op)
                                        for j in range(num_res_blocks + 1)])
            ch = tgt
            if i < depth:
                nxt = block_out_channels[i + 1] if i + 1 < len(block_out_channels) and upsample_match_channel else ch
                stage.upsample = UpSmpl(ch, nxt, tus=i < depth_temporal, refiner_vae=self.refiner_vae, op=conv_op)
                ch = nxt
            self.up.append(stage)

        self.norm_out = norm_op(ch)
        self.conv_out = conv_op(ch, out_channels, 3, stride=1, padding=1)

    def forward(self, z):
        if self.refiner_vae:
            z = z.permute(0, 2, 1, 3, 4)
            b, f, c, h, w = z.shape
            z = z.reshape(b, f, 2, c // 2, h, w)
            z = z.permute(0, 1, 2, 3, 4, 5).reshape(b, f * 2, c // 2, h, w)
            z = z.permute(0, 2, 1, 3, 4)
            z = z[:, :, 1:]

        x = self.conv_in(z) + z.repeat_interleave(self.block_out_channels[0] // self.z_channels, 1)
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))

        for stage in self.up:
            for blk in stage.block:
                x = blk(x)
            if hasattr(stage, 'upsample'):
                x = stage.upsample(x)

        out = self.conv_out(F.silu(self.norm_out(x)))

        if not self.refiner_vae:
            if z.shape[-3] == 1:
                out = out[:, :, -1:]

        return out
