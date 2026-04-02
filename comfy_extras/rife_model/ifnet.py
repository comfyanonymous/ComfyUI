import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

ops = comfy.ops.disable_weight_init

_warp_grid_cache = {}
_WARP_GRID_CACHE_MAX = 4


def clear_warp_cache():
    _warp_grid_cache.clear()


def warp(img, flow):
    B, _, H, W = img.shape
    dtype = img.dtype
    img = img.float()
    flow = flow.float()
    cache_key = (H, W, flow.device)
    if cache_key not in _warp_grid_cache:
        if len(_warp_grid_cache) >= _WARP_GRID_CACHE_MAX:
            _warp_grid_cache.pop(next(iter(_warp_grid_cache)))
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=torch.float32),
            indexing="ij",
        )
        _warp_grid_cache[cache_key] = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
    grid = _warp_grid_cache[cache_key].expand(B, -1, -1, -1)
    flow_norm = torch.cat([
        flow[:, 0:1] / ((W - 1) / 2),
        flow[:, 1:2] / ((H - 1) / 2),
    ], dim=1)
    grid = (grid + flow_norm).permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True).to(dtype)


class Head(nn.Module):
    def __init__(self, out_ch=4, device=None, dtype=None, operations=ops):
        super().__init__()
        self.cnn0 = operations.Conv2d(3, 16, 3, 2, 1, device=device, dtype=dtype)
        self.cnn1 = operations.Conv2d(16, 16, 3, 1, 1, device=device, dtype=dtype)
        self.cnn2 = operations.Conv2d(16, 16, 3, 1, 1, device=device, dtype=dtype)
        self.cnn3 = operations.ConvTranspose2d(16, out_ch, 4, 2, 1, device=device, dtype=dtype)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c, device=None, dtype=None, operations=ops):
        super().__init__()
        self.conv = operations.Conv2d(c, c, 3, 1, 1, device=device, dtype=dtype)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1), device=device, dtype=dtype))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(torch.addcmul(x, self.conv(x), self.beta))


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, device=None, dtype=None, operations=ops):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Sequential(
                operations.Conv2d(in_planes, c // 2, 3, 2, 1, device=device, dtype=dtype),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                operations.Conv2d(c // 2, c, 3, 2, 1, device=device, dtype=dtype),
                nn.LeakyReLU(0.2, True),
            ),
        )
        self.convblock = nn.Sequential(
            *(ResConv(c, device=device, dtype=dtype, operations=operations) for _ in range(8))
        )
        self.lastconv = nn.Sequential(
            operations.ConvTranspose2d(c, 4 * 13, 4, 2, 1, device=device, dtype=dtype),
            nn.PixelShuffle(2),
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear").div_(scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear")
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet(nn.Module):
    def __init__(self, head_ch=4, channels=None, device=None, dtype=None, operations=ops):
        super().__init__()
        if channels is None:
            channels = [192, 128, 96, 64, 32]
        self.encode = Head(out_ch=head_ch, device=device, dtype=dtype, operations=operations)
        block_in = [7 + 2 * head_ch] + [8 + 4 + 8 + 2 * head_ch] * 4
        self.blocks = nn.ModuleList([
            IFBlock(block_in[i], channels[i], device=device, dtype=dtype, operations=operations)
            for i in range(5)
        ])
        self.scale_list = [16, 8, 4, 2, 1]

    def get_dtype(self):
        return self.encode.cnn0.weight.dtype

    def forward(self, img0, img1, timestep=0.5):
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]),
                                  timestep, device=img0.device, dtype=img0.dtype)
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        flow = mask = feat = None
        warped_img0 = img0
        warped_img1 = img1
        for i, block in enumerate(self.blocks):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0, img1, f0, f1, timestep), 1),
                    None, scale=self.scale_list[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, mask, feat = block(
                    torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask, feat), 1),
                    flow, scale=self.scale_list[i],
                )
                flow = flow.add_(fd)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return torch.lerp(warped_img1, warped_img0, mask)


def detect_rife_config(state_dict):
    # Determine head output channels from encode.cnn3 (ConvTranspose2d)
    # ConvTranspose2d weight shape is (in_ch, out_ch, kH, kW)
    head_ch = state_dict["encode.cnn3.weight"].shape[1]

    # Read per-block channel widths from conv0 second layer output channels
    # conv0 is Sequential(Sequential(Conv2d, ReLU), Sequential(Conv2d, ReLU))
    # conv0.1.0.weight shape is (c, c//2, 3, 3)
    channels = []
    for i in range(5):
        key = f"blocks.{i}.conv0.1.0.weight"
        if key in state_dict:
            channels.append(state_dict[key].shape[0])

    if len(channels) != 5:
        raise ValueError(f"Unsupported RIFE model: expected 5 blocks, found {len(channels)}")

    return head_ch, channels
