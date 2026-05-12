"""High-level loader and inference wrapper for MoGe v1 / v2 checkpoints.

Mirrors the structure of :mod:`comfy.clip_vision`: owns the ``nn.Module`` and
a :class:`comfy.model_patcher.CoreModelPatcher`, exposes a
:meth:`MoGeModel.infer` that runs preprocessing, forward, and post-processing.
"""

from __future__ import annotations

from numbers import Number
from typing import Dict, Optional, Union

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .ldm.moge.geometry import (
    depth_map_to_point_map,
    intrinsics_from_focal_center,
    recover_focal_shift,
)
from .ldm.moge.model import detect_and_build
from .ldm.moge.state_dict import remap_moge_state_dict


class MoGeModel:
    """Loaded MoGe model + ComfyUI memory management."""

    def __init__(self, state_dict: dict):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)

        sd = remap_moge_state_dict(state_dict)
        self.model = detect_and_build(sd, dtype=self.dtype, device=offload_device,
                                      operations=comfy.ops.manual_cast)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model, load_device=self.load_device, offload_device=offload_device
        )
        self.version = "v2" if hasattr(self.model, "encoder") else "v1"
        self.mask_threshold = float(getattr(self.model, "mask_threshold", 0.5))
        nt = getattr(self.model, "num_tokens_range", (1200, 2500 if self.version == "v1" else 3600))
        self.num_tokens_range = (int(nt[0]), int(nt[1]))

    @torch.inference_mode()
    def infer(self,
              image: torch.Tensor,
              num_tokens: Optional[int] = None,
              resolution_level: int = 9,
              fov_x: Optional[Union[Number, torch.Tensor]] = None,
              force_projection: bool = True,
              apply_mask: bool = True) -> Dict[str, torch.Tensor]:
        """Run a single MoGe forward + post-process pass.

        ``image`` must already be ``(B, 3, H, W)`` in ``[0, 1]`` on any device.
        Returns a dict with at least ``points``, ``depth``, ``intrinsics``,
        ``mask``; v2 checkpoints additionally produce ``normal``.
        """
        comfy.model_management.load_model_gpu(self.patcher)
        device = self.load_device
        image = image.to(device=device, dtype=self.dtype)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        H, W = image.shape[-2:]
        aspect_ratio = W / H

        if num_tokens is None:
            lo, hi = self.num_tokens_range
            num_tokens = int(lo + (resolution_level / 9) * (hi - lo))

        out = self.model.forward(image, num_tokens=num_tokens)
        points = out.get("points")
        normal = out.get("normal")
        mask = out.get("mask")
        metric_scale = out.get("metric_scale")

        # Post-processing always runs in fp32 for numerical stability.
        if points is not None: points = points.float()
        if normal is not None: normal = normal.float()
        if mask is not None: mask = mask.float()
        if metric_scale is not None: metric_scale = metric_scale.float()

        mask_binary = (mask > self.mask_threshold) if mask is not None else None

        depth = None
        intrinsics = None
        if points is not None:
            if fov_x is None:
                focal, shift = recover_focal_shift(points, mask_binary)
                # The unconstrained least-squares solver inside recover_focal_shift
                # can converge to a degenerate solution where (z + shift) is
                # negative for most pixels, which flips the sign of the
                # estimated focal.  Detect that and fall back to a sensible
                # 60-degree-FoV default rather than emitting garbage geometry.
                bad = ~torch.isfinite(focal) | (focal <= 0)
                if bool(bad.any()):
                    default_fov = 60.0
                    fov_t = torch.as_tensor(default_fov, device=points.device, dtype=points.dtype)
                    fallback_focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 \
                                     / torch.tan(torch.deg2rad(fov_t / 2))
                    fallback_focal = fallback_focal.expand_as(focal).clone()
                    focal = torch.where(bad, fallback_focal, focal)
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
            else:
                fov_t = torch.as_tensor(fov_x, device=points.device, dtype=points.dtype)
                focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(fov_t / 2))
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _, shift = recover_focal_shift(points, mask_binary, focal=focal)
            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            half = torch.tensor(0.5, device=points.device, dtype=points.dtype)
            intrinsics = intrinsics_from_focal_center(fx, fy, half, half)
            points[..., 2] = points[..., 2] + shift[..., None, None]
            # v2 upstream additionally filters mask by depth > 0 as a safeguard
            # against negative-depth artifacts from the metric-scale path; v1
            # does not, and applying it there can cut out the foreground when
            # shift recovery overshoots slightly.
            if mask_binary is not None and self.version == "v2":
                mask_binary = mask_binary & (points[..., 2] > 0)
            depth = points[..., 2].clone()

        if force_projection and depth is not None and intrinsics is not None:
            points = depth_map_to_point_map(depth, intrinsics=intrinsics)

        if metric_scale is not None:
            if points is not None:
                points = points * metric_scale[:, None, None, None]
            if depth is not None:
                depth = depth * metric_scale[:, None, None]

        if apply_mask and mask_binary is not None:
            if points is not None:
                points = torch.where(mask_binary[..., None], points,
                                     torch.full_like(points, float("inf")))
            if depth is not None:
                depth = torch.where(mask_binary, depth,
                                    torch.full_like(depth, float("inf")))
            if normal is not None:
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

        result = {
            "points": points,
            "depth": depth,
            "intrinsics": intrinsics,
            "mask": mask_binary,
            "normal": normal,
        }
        return {k: v for k, v in result.items() if v is not None}


def load(ckpt_path: str) -> MoGeModel:
    """Load a MoGe ``.pt`` / ``.safetensors`` checkpoint into a :class:`MoGeModel`."""
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    if isinstance(sd, dict) and "model" in sd and "model_config" in sd:
        sd = sd["model"]
    return MoGeModel(sd)
