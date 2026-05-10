import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.modules.diffusionmodules.model import ResnetBlock, AttnBlock
import comfy.ops
ops = comfy.ops.disable_weight_init


class PixelShuffle2D(nn.Module):
    def __init__(self, in_dim, out_dim, op=ops.Conv2d):
        super().__init__()
        self.conv = op(in_dim, out_dim >> 2, 3, 1, 1)
        self.ratio = (in_dim << 2) // out_dim

    def forward(self, x):
        b, c, h, w = x.shape
        h2, w2 = h >> 1, w >> 1
        y = self.conv(x).view(b, -1, h2, 2, w2, 2).permute(0, 3, 5, 1, 2, 4).reshape(b, -1, h2, w2)
        r = x.view(b, c, h2, 2, w2, 2).permute(0, 3, 5, 1, 2, 4).reshape(b, c << 2, h2, w2)
        return y + r.view(b, y.shape[1], self.ratio, h2, w2).mean(2)


class PixelUnshuffle2D(nn.Module):
    def __init__(self, in_dim, out_dim, op=ops.Conv2d):
        super().__init__()
        self.conv = op(in_dim, out_dim << 2, 3, 1, 1)
        self.scale = (out_dim << 2) // in_dim

    def forward(self, x):
        b, c, h, w = x.shape
        h2, w2 = h << 1, w << 1
        y = self.conv(x).view(b, 2, 2, -1, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, -1, h2, w2)
        r = x.repeat_interleave(self.scale, 1).view(b, 2, 2, -1, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, -1, h2, w2)
        return y + r


class Encoder(nn.Module):
    def __init__(self, in_channels, z_channels, block_out_channels, num_res_blocks,
                 ffactor_spatial, downsample_match_channel=True, **_):
        super().__init__()
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.conv_in = ops.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)

        self.down = nn.ModuleList()
        ch = block_out_channels[0]
        depth = (ffactor_spatial >> 1).bit_length()

        for i, tgt in enumerate(block_out_channels):
            stage = nn.Module()
            stage.block = nn.ModuleList([ResnetBlock(in_channels=ch if j == 0 else tgt,
                                                     out_channels=tgt,
                                                     temb_channels=0,
                                                     conv_op=ops.Conv2d)
                                        for j in range(num_res_blocks)])
            ch = tgt
            if i < depth:
                nxt = block_out_channels[i + 1] if i + 1 < len(block_out_channels) and downsample_match_channel else ch
                stage.downsample = PixelShuffle2D(ch, nxt, ops.Conv2d)
                ch = nxt
            self.down.append(stage)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=ops.Conv2d)
        self.mid.attn_1 = AttnBlock(ch, conv_op=ops.Conv2d)
        self.mid.block_2 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=ops.Conv2d)

        self.norm_out = ops.GroupNorm(32, ch, 1e-6, True)
        self.conv_out = ops.Conv2d(ch, z_channels << 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)

        for stage in self.down:
            for blk in stage.block:
                x = blk(x)
            if hasattr(stage, 'downsample'):
                x = stage.downsample(x)

        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))

        b, c, h, w = x.shape
        grp = c // (self.z_channels << 1)
        skip = x.view(b, c // grp, grp, h, w).mean(2)

        return self.conv_out(F.silu(self.norm_out(x))) + skip


class Decoder(nn.Module):
    def __init__(self, z_channels, out_channels, block_out_channels, num_res_blocks,
                 ffactor_spatial, upsample_match_channel=True, **_):
        super().__init__()
        block_out_channels = block_out_channels[::-1]
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        ch = block_out_channels[0]
        self.conv_in = ops.Conv2d(z_channels, ch, 3, 1, 1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=ops.Conv2d)
        self.mid.attn_1 = AttnBlock(ch, conv_op=ops.Conv2d)
        self.mid.block_2 = ResnetBlock(in_channels=ch, out_channels=ch, temb_channels=0, conv_op=ops.Conv2d)

        self.up = nn.ModuleList()
        depth = (ffactor_spatial >> 1).bit_length()

        for i, tgt in enumerate(block_out_channels):
            stage = nn.Module()
            stage.block = nn.ModuleList([ResnetBlock(in_channels=ch if j == 0 else tgt,
                                                     out_channels=tgt,
                                                     temb_channels=0,
                                                     conv_op=ops.Conv2d)
                                        for j in range(num_res_blocks + 1)])
            ch = tgt
            if i < depth:
                nxt = block_out_channels[i + 1] if i + 1 < len(block_out_channels) and upsample_match_channel else ch
                stage.upsample = PixelUnshuffle2D(ch, nxt, ops.Conv2d)
                ch = nxt
            self.up.append(stage)

        self.norm_out = ops.GroupNorm(32, ch, 1e-6, True)
        self.conv_out = ops.Conv2d(ch, out_channels, 3, 1, 1)

    def forward(self, z):
        x = self.conv_in(z) + z.repeat_interleave(self.block_out_channels[0] // self.z_channels, 1)
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))

        for stage in self.up:
            for blk in stage.block:
                x = blk(x)
            if hasattr(stage, 'upsample'):
                x = stage.upsample(x)

        return self.conv_out(F.silu(self.norm_out(x)))
