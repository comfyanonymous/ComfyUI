# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The patcher and unpatcher implementation for 2D and 3D data.

The idea of Haar wavelet is to compute LL, LH, HL, HH component as two 1D convolutions.
One on the rows and one on the columns.
For example, in 1D signal, we have [a, b], then the low-freq compoenent is [a + b] / 2 and high-freq is [a - b] / 2.
We can use a 1D convolution with kernel [1, 1] and stride 2 to represent the L component.
For H component, we can use a 1D convolution with kernel [1, -1] and stride 2.
Although in principle, we typically only do additional Haar wavelet over the LL component. But here we do it for all
   as we need to support downsampling for more than 2x.
For example, 4x downsampling can be done by 2x Haar and additional 2x Haar, and the shape would be.
   [3, 256, 256] -> [12, 128, 128] -> [48, 64, 64]
"""

import torch
import torch.nn.functional as F
from einops import rearrange

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = False


class Patcher(torch.nn.Module):
    """A module to convert image tensors into patches using torch operations.

    The main difference from `class Patching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Patching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer(
            "wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT
        )
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._haar(x)
        elif self.patch_method == "rearrange":
            return self._arrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets.to(device=x.device)

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange.to(device=x.device))).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x):
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        ).contiguous()
        return x


class Patcher3D(Patcher):
    """A 3D discrete wavelet transform for video data, expects 5D tensor, i.e. a batch of videos."""

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__(patch_method=patch_method, patch_size=patch_size)
        self.register_buffer(
            "patch_size_buffer",
            patch_size * torch.ones([1], dtype=torch.int32),
            persistent=_PERSISTENT,
        )

    def _dwt(self, x, wavelet, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets.to(device=x.device)

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange.to(device=x.device))).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # Handles temporal axis.
        x = F.pad(
            x, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode
        ).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes.
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out / (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    def _haar(self, x):
        xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
        x = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        for _ in self.range:
            x = self._dwt(x, "haar", rescale=True)
        return x

    def _arrange(self, x):
        xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
        x = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        x = rearrange(
            x,
            "b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        ).contiguous()
        return x


class UnPatcher(torch.nn.Module):
    """A module to convert patches into image tensorsusing torch operations.

    The main difference from `class Unpatching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Unpatching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer(
            "wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT
        )
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._ihaar(x)
        elif self.patch_method == "rearrange":
            return self._iarrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets.to(device=x.device)
        n = h.shape[0]

        g = x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange.to(device=x.device))).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)

        # Inverse transform.
        yl = torch.nn.functional.conv_transpose2d(
            xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yl += torch.nn.functional.conv_transpose2d(
            xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh = torch.nn.functional.conv_transpose2d(
            xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh += torch.nn.functional.conv_transpose2d(
            xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        y = torch.nn.functional.conv_transpose2d(
            yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )
        y += torch.nn.functional.conv_transpose2d(
            yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x


class UnPatcher3D(UnPatcher):
    """A 3D inverse discrete wavelet transform for video wavelet decompositions."""

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__(patch_method=patch_method, patch_size=patch_size)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets.to(device=x.device)

        g = x.shape[1] // 8  # split into 8 spatio-temporal filtered tesnors.
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange.to(device=x.device))).reshape(1, 1, -1).repeat(g, 1, 1)
        hl = hl.to(dtype=dtype)
        hh = hh.to(dtype=dtype)

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)
        del x

        # Height height transposed convolutions.
        xll = F.conv_transpose3d(
            xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xlll

        xll += F.conv_transpose3d(
            xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xllh

        xlh = F.conv_transpose3d(
            xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xlhl

        xlh += F.conv_transpose3d(
            xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xlhh

        xhl = F.conv_transpose3d(
            xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xhll

        xhl += F.conv_transpose3d(
            xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xhlh

        xhh = F.conv_transpose3d(
            xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xhhl

        xhh += F.conv_transpose3d(
            xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        del xhhh

        # Handles width transposed convolutions.
        xl = F.conv_transpose3d(
            xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        del xll

        xl += F.conv_transpose3d(
            xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        del xlh

        xh = F.conv_transpose3d(
            xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        del xhl

        xh += F.conv_transpose3d(
            xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        del xhh

        # Handles time axis transposed convolutions.
        x = F.conv_transpose3d(
            xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )
        del xl

        x += F.conv_transpose3d(
            xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )

        if rescale:
            x = x * (2 * torch.sqrt(torch.tensor(2.0)))
        return x

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        x = x[:, :, self.patch_size - 1 :, ...]
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        )
        x = x[:, :, self.patch_size - 1 :, ...]
        return x
