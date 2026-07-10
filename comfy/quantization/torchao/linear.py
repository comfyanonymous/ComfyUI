"""
TINT4Linear — INT4 weight-only Linear with LoRA + QuaRot.

Backed by torchao Int4PlainInt32Tensor. Activations stay FP16.
LoRA entries are injected at forward time — no kernel modification needed.
QuaRot activation rotation is applied online when enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization.quantize_.workflows.int4.int4_plain_int32_tensor import (
    Int4PlainInt32Tensor,
)


class TINT4Linear(nn.Module):
    """
    INT4 weight-only linear layer.

    Stores raw int32 qdata + fp16 scales + int8 zero-points.
    Int4PlainInt32Tensor is lazily constructed and rebuilt on device change.

    Slots (set externally by loaders):
      _use_quarot: bool         — enable activation Hadamard rotation
      _group_size: int          — QuaRot group size
      _hadamard_H: Tensor|None  — normalized Hadamard matrix
      _tint4_lora_entries: dict|None — {lora_name: [(type, ...), ...]}
    """

    def __init__(self, in_features: int, out_features: int,
                 qdata: torch.Tensor, scale: torch.Tensor,
                 zp: torch.Tensor, block_size: tuple,
                 bias: torch.Tensor | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

        self._qdata = qdata           # int32  (out_f, in_f // 8)
        self._scale = scale           # fp16   (n_blocks, out_f)
        self._zp = zp                 # int8   (n_blocks, out_f)
        self._block_size = block_size # (b0, b1)
        self._qt: Int4PlainInt32Tensor | None = None

        # QuaRot (optional)
        self._use_quarot: bool = False
        self._group_size: int = 128
        self._hadamard_H: torch.Tensor | None = None

        # LoRA entries slot
        self._tint4_lora_entries: dict | None = None

    @property
    def weight(self) -> Int4PlainInt32Tensor:
        if self._qt is None:
            self._qt = Int4PlainInt32Tensor(
                self._qdata, self._scale, self._zp,
                self._block_size, [self.out_features, self.in_features],
            )
        return self._qt

    @weight.setter
    def weight(self, value: Int4PlainInt32Tensor) -> None:
        if isinstance(value, Int4PlainInt32Tensor):
            self._qt = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, x.shape[-1])

        # ── QuaRot activation rotation (online) ──────────────────
        if self._use_quarot and self._hadamard_H is not None:
            if x_flat.shape[-1] % self._group_size != 0:
                raise ValueError(
                    f"activation dim {x_flat.shape[-1]} not divisible "
                    f"by QuaRot group_size {self._group_size}"
                )
            H_dev = self._hadamard_H.to(device=x.device, dtype=x.dtype)
            n_groups = x_flat.shape[-1] // self._group_size
            x_g = x_flat.view(-1, n_groups, self._group_size)
            x_flat = torch.matmul(x_g, H_dev).view(-1, n_groups * self._group_size)

        dev = x.device

        # ── Rebuild Int4PlainInt32Tensor on device change ────────
        if self._qt is None or self._qt.device != dev:
            self._qt = Int4PlainInt32Tensor(
                self._qdata.to(dev), self._scale.to(dev), self._zp.to(dev),
                self._block_size, [self.out_features, self.in_features],
            )

        out = F.linear(x_flat, self._qt, None)

        # ── LoRA forward injection ───────────────────────────────
        entries = self._tint4_lora_entries
        if entries is not None and entries:
            cd = (x.dtype if x.dtype in (torch.float16, torch.bfloat16)
                  else torch.float16)
            for lora_entries in entries.values():
                for e in lora_entries:
                    # ── LoKr path ────────────────────────────────
                    if isinstance(e[0], str) and e[0] == "lokr":
                        _, w1, w2, mult, factor = e[:5]
                        sl = e[5] if len(e) > 5 else None
                        se = e[6] if len(e) > 6 else None
                        w1d = w1.to(device=dev, dtype=cd)
                        w2d = w2.to(device=dev, dtype=cd)
                        of2, if2 = w2d.shape
                        w1x = w1d.repeat_interleave(
                            of2 // factor, dim=0).repeat_interleave(
                            if2 // factor, dim=1)
                        dw = (w1x * w2d).mul_(mult)
                        lo = x_flat.to(cd) @ dw.T
                        if sl is not None:
                            if lo.shape[1] != (se - sl):
                                raise ValueError(
                                    f"LoKr output width {lo.shape[1]} "
                                    f"doesn't match target slice width "
                                    f"{se - sl}"
                                )
                            out[:, sl:se] += lo
                        elif lo.shape == out.shape:
                            out += lo
                        else:
                            raise ValueError(
                                f"LoKr output shape {lo.shape} "
                                f"incompatible with {out.shape}"
                            )
                        continue

                    # ── Standard LoRA path ────────────────────────
                    A, B, mult = e[:3]
                    sl = e[3] if len(e) > 3 else None
                    se = e[4] if len(e) > 4 else None
                    Ad = A.to(device=dev, dtype=cd)
                    Bd = B.to(device=dev, dtype=cd)
                    lo = (x_flat.to(cd) @ Ad.T) @ Bd.T * mult
                    if sl is not None:
                        if lo.shape[1] != (se - sl):
                            raise ValueError(
                                f"LoRA output width {lo.shape[1]} "
                                f"doesn't match target slice width "
                                f"{se - sl}"
                            )
                        out[:, sl:se] += lo
                    elif lo.shape[1] == out.shape[1]:
                        out += lo
                    else:
                        raise ValueError(
                            f"LoRA output shape {lo.shape} "
                            f"incompatible with {out.shape}"
                        )

        if self.bias is not None:
            out += self.bias.to(device=dev, dtype=out.dtype)

        return out.reshape(*x.shape[:-1], out.shape[-1])

    def release(self) -> None:
        """Drop cached Int4PlainInt32Tensor to free VRAM."""
        self._qt = None

    def __del__(self):
        self._qdata = None
        self._scale = None
        self._zp = None
        self._qt = None
        self._tint4_lora_entries = None
