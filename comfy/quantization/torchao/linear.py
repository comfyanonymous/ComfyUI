"""
TINT4Linear — INT4 weight-only Linear with LoRA + QuaRot.

Backed by torchao Int4PlainInt32Tensor. Activations stay FP16.
LoRA entries are injected at forward time — no kernel modification needed.
QuaRot activation rotation is applied online; LoRA A/w2 matrices are
rotated into the QuaRot basis inside forward() automatically unless
_lora_prerotated is set (for callers that pre-rotate at injection time).
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
	  _lora_prerotated: bool    — True if caller already rotated A/w2
								  into QuaRot basis (default False).
								  When False, rotation is done in forward().
	"""

	def __init__(self, in_features: int, out_features: int,
				 qdata: torch.Tensor, scale: torch.Tensor,
				 zp: torch.Tensor, block_size: tuple,
				 bias: torch.Tensor | None = None):
		"""
		Initialize TINT4Linear from raw torchao INT4 tensors.

		Args:
			in_features: Input feature dimension.
			out_features: Output feature dimension.
			qdata: Packed int32 quantized weight data (out_f, in_f // 8).
			scale: Per-block fp16 scales (n_blocks, out_f).
			zp: Per-block int8 zero-points (n_blocks, out_f).
			block_size: (b0, b1) tuple defining quantization block shape.
			bias: Optional bias tensor (out_f,).
		"""
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features

		if bias is not None:
			self.bias = nn.Parameter(bias)
		else:
			self.register_parameter('bias', None)

		self.register_buffer('_qdata', qdata)
		self.register_buffer('_scale', scale)
		self.register_buffer('_zp', zp)
		self._block_size = block_size
		self._qt: Int4PlainInt32Tensor | None = None

		# QuaRot (optional)
		self._use_quarot: bool = False
		self._group_size: int = 128
		self._hadamard_H: torch.Tensor | None = None

		# LoRA entries slot
		self._tint4_lora_entries: dict | None = None

		# Pre-rotation flag
		self._lora_prerotated: bool = False

	@property
	def weight(self) -> Int4PlainInt32Tensor:
		"""Lazily construct and return the Int4PlainInt32Tensor view."""
		if self._qt is None:
			self._qt = Int4PlainInt32Tensor(
				self._qdata, self._scale, self._zp,
				self._block_size, [self.out_features, self.in_features],
			)
		return self._qt

	@weight.setter
	def weight(self, value: Int4PlainInt32Tensor) -> None:
		"""Replace the underlying Int4PlainInt32Tensor."""
		if isinstance(value, Int4PlainInt32Tensor):
			self._qt = value

	def _rotate_into_quarot(self, tensor: torch.Tensor) -> torch.Tensor:
		"""
		Rotate a LoRA A or w2 matrix into the QuaRot Hadamard basis.

		Skipped when _lora_prerotated is True (caller already rotated).
		Returns unchanged tensor if QuaRot is disabled or dimensions
		are not divisible by _group_size.
		"""
		if self._lora_prerotated:
			return tensor
		if not self._use_quarot or self._hadamard_H is None:
			return tensor
		if tensor.shape[1] % self._group_size != 0:
			return tensor
		Ht = self._hadamard_H.T.to(device=tensor.device, dtype=tensor.dtype)
		ng = tensor.shape[1] // self._group_size
		return (tensor.reshape(tensor.shape[0], ng, self._group_size)
				@ Ht).reshape(tensor.shape[0], -1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass with optional QuaRot activation rotation and LoRA injection.

		Args:
			x: Input tensor, shape (..., in_features).

		Returns:
			Output tensor, shape (..., out_features).

		Raises:
			ValueError: If activation dim is not divisible by QuaRot group_size,
						or if LoRA output shapes are incompatible.
		"""
		x_flat = x.reshape(-1, x.shape[-1])

		# QuaRot activation rotation (online)
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

		# Rebuild Int4PlainInt32Tensor on device change
		if self._qt is None or self._qt.device != dev:
			self._qt = Int4PlainInt32Tensor(
				self._qdata.to(dev), self._scale.to(dev), self._zp.to(dev),
				self._block_size, [self.out_features, self.in_features],
			)

		out = F.linear(x_flat, self._qt, None)

		# LoRA forward injection
		entries = self._tint4_lora_entries
		if entries is not None and entries:
			cd = (x.dtype if x.dtype in (torch.float16, torch.bfloat16)
				  else torch.float16)
			for lora_entries in entries.values():
				for e in lora_entries:
					# ── LoKr path ──
					# ★ FIX 2: 用 torch.kron 替代 repeat_interleave
					# 旧方案要求 w2 维度能被 factor 整除（例如 64÷60=不整除→报错）
					# 新方案 kron 展开 + pad/trim 到目标维度，兼容任意分解
					if isinstance(e[0], str) and e[0] == "lokr":
						_, w1, w2, mult, factor = e[:5]
						sl = e[5] if len(e) > 5 else None
						se = e[6] if len(e) > 6 else None
）
						w2 = self._rotate_into_quarot(w2)
						w1d = w1.to(device=dev, dtype=cd)
						w2d = w2.to(device=dev, dtype=cd)

						# kron 展开为完整 delta（代替 repeat_interleave）
						dw = torch.kron(w1d, w2d)

						# Pad/trim to layer (or slice) dimensions
						tgt_out = (se - sl) if sl is not None else self.out_features
						if dw.shape[0] < tgt_out:
							dw = dw.repeat(
								(tgt_out + dw.shape[0] - 1) // dw.shape[0], 1)
						if dw.shape[0] > tgt_out:
							dw = dw[:tgt_out, :]
						if dw.shape[1] < self.in_features:
							dw = dw.repeat(
								1, (self.in_features + dw.shape[1] - 1) // dw.shape[1])
						if dw.shape[1] > self.in_features:
							dw = dw[:, :self.in_features]

						dw = dw.contiguous().mul_(mult)
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

					# Standard LoRA path
					A, B, mult = e[:3]
					sl = e[3] if len(e) > 3 else None
					se = e[4] if len(e) > 4 else None
					A = self._rotate_into_quarot(A)
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
