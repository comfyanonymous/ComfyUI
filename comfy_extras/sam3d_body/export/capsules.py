"""3D capsule rendering for OpenPose-style skeletons (SCAIL-Pose look).

Each limb is a true 3D capsule (cylinder + hemispherical caps), projected
through the per-person camera (`pred_cam_t` + `focal_length` + image_size) so
closer limbs appear thicker/brighter. Self-contained analytic ray-capsule
renderer. Output: (H, W, 3) fp32 torch.Tensor in [0, 1].
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import comfy.model_management

from .glb_shared import (
    OPENPOSE_18_PAIRS,
    OPENPOSE_RAINBOW_18,
    SCAIL_LIMB_COLORS_17,
    OPENPOSE_HAND_PAIRS,
    OPENPOSE_HAND_COLORS_21,
    openpose_render_keypoints,
)


def _limb_palette_rgb01(palette: str) -> np.ndarray:
    """17 per-limb RGB colors in [0,1] for the OpenPose-18 body limbs."""
    if palette == "scail":
        return SCAIL_LIMB_COLORS_17.astype(np.float32)
    return OPENPOSE_RAINBOW_18[: len(OPENPOSE_18_PAIRS)].astype(np.float32)


def _build_specs_from_pose(
    persons: List[Dict[str, Any]],
    pose_data: Dict[str, Any],
    *,
    include_hands: bool,
    palette: str,
    person_brightness_falloff: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten body + optional hand limbs for one frame into (starts, ends,
    colors_rgba, is_hand) in camera coords (Y-down, +Z forward). Drops non-finite
    or behind-camera endpoints; `is_hand` lets the renderer draw hands thinner.

    `person_brightness_falloff` mixes each per-person color toward white by
    `1 - falloff^k` for track k (track 0 stays vivid)."""
    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    is_hand: List[bool] = []

    body_limb_colors = _limb_palette_rgb01(palette)
    hand_limb_colors = OPENPOSE_HAND_COLORS_21.astype(np.float32)

    falloff = max(0.0, min(1.0, float(person_brightness_falloff)))

    for k, person in enumerate(persons):
        cam_t = person.get("pred_cam_t")
        body_op = openpose_render_keypoints(person, pose_data, "body", dim=3)
        if body_op is None or cam_t is None:
            continue
        cam_t_np = np.asarray(cam_t, dtype=np.float32).reshape(3)
        # op-keypoints are camera frame; add cam_t to place the subject in front.
        body_kp = body_op + cam_t_np[None, :]

        pastel = 0.0 if k == 0 else (1.0 - falloff ** k)

        def _tint(rgb: np.ndarray) -> np.ndarray:
            if pastel <= 0:
                return rgb
            return rgb * (1.0 - pastel) + pastel

        # SCAIL skips face bones (13..16) and redirects limb 12 into a short
        # head stub blending spine + neck→nose direction.
        body_limb_count = 13 if palette == "scail" else len(OPENPOSE_18_PAIRS)
        spine_dir = None
        if palette == "scail":
            mid_hip = 0.5 * (body_kp[8] + body_kp[11])  # 8=RHip, 11=LHip
            sd = body_kp[1] - mid_hip                    # 1=Neck
            sd_len = float(np.linalg.norm(sd))
            if np.all(np.isfinite(sd)) and sd_len > 1e-6:
                spine_dir = sd / sd_len
        for limb_i, (a, b) in enumerate(OPENPOSE_18_PAIRS[:body_limb_count]):
            sa, sb = body_kp[a], body_kp[b]
            if not (np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))):
                continue
            if sa[2] <= 0 or sb[2] <= 0:
                continue
            if palette == "scail" and limb_i == 12:
                nose_vec = sb - sa
                nose_len = float(np.linalg.norm(nose_vec))
                if nose_len > 1e-6 and spine_dir is not None:
                    nose_dir = nose_vec / nose_len
                    mixed = 0.6 * spine_dir + 0.4 * nose_dir
                    mixed = mixed / max(float(np.linalg.norm(mixed)), 1e-6)
                    sb = sa + mixed * (nose_len * 0.5)
                elif nose_len > 1e-6:
                    sb = sa + nose_vec * 0.5
                elif spine_dir is not None:
                    sb = sa + spine_dir * (sd_len * 0.3)
            starts.append(sa)
            ends.append(sb)
            is_hand.append(False)
            color_rgb = _tint(body_limb_colors[limb_i])
            colors.append(np.array([color_rgb[0], color_rgb[1], color_rgb[2], 1.0],
                                   dtype=np.float32))

        if include_hands:
            hand_ops = [openpose_render_keypoints(person, pose_data, p, dim=3)
                        for p in ("hand_r", "hand_l")]
            hand_kps = [h + cam_t_np[None, :] for h in hand_ops if h is not None]
            for limb_i, (a, b) in enumerate(OPENPOSE_HAND_PAIRS):
                for hand_kp in hand_kps:
                    sa, sb = hand_kp[a], hand_kp[b]
                    if not (np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))):
                        continue
                    if sa[2] <= 0 or sb[2] <= 0:
                        continue
                    starts.append(sa)
                    ends.append(sb)
                    is_hand.append(True)
                    color_rgb = _tint(hand_limb_colors[(a + b) % len(hand_limb_colors)])
                    colors.append(np.array([color_rgb[0], color_rgb[1], color_rgb[2], 1.0],
                                           dtype=np.float32))

    if not starts:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=bool))
    return (np.stack(starts).astype(np.float32),
            np.stack(ends).astype(np.float32),
            np.stack(colors).astype(np.float32),
            np.asarray(is_hand, dtype=bool))


def _ray_capsule_t(
    ray_dirs: torch.Tensor,    # (K, 3) unit rays from camera origin
    starts: torch.Tensor,      # (M, 3)
    ends: torch.Tensor,        # (M, 3)
    ba_norm: torch.Tensor,     # (M, 3) unit axis (A → B)
    ba_len: torch.Tensor,      # (M,) segment length
    radius: torch.Tensor,      # (M,) per-capsule radius
) -> torch.Tensor:
    """Closed-form ray-capsule intersection -> (K, M) ray params t to the nearest
    valid hit per capsule, +inf on miss. Capsule = union of (cylinder, hemisphere
    at A, hemisphere at B), each a quadratic root-find."""
    INF = float("inf")
    r_sq = radius * radius                      # (M,)

    # Cached dot products.
    dn = ray_dirs @ ba_norm.transpose(0, 1)     # (K, M) — d·n
    dA = ray_dirs @ starts.transpose(0, 1)      # (K, M) — d·A
    dB = ray_dirs @ ends.transpose(0, 1)        # (K, M) — d·B
    An = (starts * ba_norm).sum(-1)             # (M,)   — A·n
    A_sq = (starts * starts).sum(-1)            # (M,)   — |A|²
    B_sq = (ends * ends).sum(-1)                # (M,)   — |B|²

    # Cylinder body: project onto plane ⊥ n and solve |P_⊥(t)|² = r².
    a_c = 1.0 - dn * dn                         # (K, M)
    b_c = -2.0 * (dA - dn * An)                 # (K, M)
    c_c = A_sq - An * An - r_sq                 # (M,)
    disc_c = b_c * b_c - 4.0 * a_c * c_c
    safe_a = a_c.clamp(min=1e-9)
    sqrt_c = torch.sqrt(disc_c.clamp(min=0.0))
    t_cyl = (-b_c - sqrt_c) / (2.0 * safe_a)
    s_cyl = t_cyl * dn - An                     # axial projection from A
    cyl_ok = (disc_c >= 0) & (a_c > 1e-7) & (t_cyl > 1e-6) & \
             (s_cyl >= 0.0) & (s_cyl <= ba_len)
    t_cyl = torch.where(cyl_ok, t_cyl, torch.full_like(t_cyl, INF))

    # Sphere at A, restricted to the hemisphere with axial projection ≤ 0.
    disc_a = dA * dA - (A_sq - r_sq)
    sqrt_a = torch.sqrt(disc_a.clamp(min=0.0))
    t_sa = dA - sqrt_a
    s_a = t_sa * dn - An
    a_ok = (disc_a >= 0) & (t_sa > 1e-6) & (s_a <= 0.0)
    t_sa = torch.where(a_ok, t_sa, torch.full_like(t_sa, INF))

    # Sphere at B, restricted to the hemisphere with axial projection ≥ ba_len.
    disc_b = dB * dB - (B_sq - r_sq)
    sqrt_b = torch.sqrt(disc_b.clamp(min=0.0))
    t_sb = dB - sqrt_b
    s_b = t_sb * dn - An
    b_ok = (disc_b >= 0) & (t_sb > 1e-6) & (s_b >= ba_len)
    t_sb = torch.where(b_ok, t_sb, torch.full_like(t_sb, INF))

    return torch.minimum(torch.minimum(t_cyl, t_sa), t_sb)


def _render_capsules_torch(
    starts: torch.Tensor,
    ends: torch.Tensor,
    colors: torch.Tensor,
    H: int, W: int,
    fx: float, fy: float, cx: float, cy: float,
    radius: torch.Tensor,     # scalar or (M,) per-capsule radius
    background_rgb: Optional[torch.Tensor],
    device: torch.device,
    flat_shade: bool = False,
) -> torch.Tensor:
    """Analytic ray-capsule renderer for a union of capsules. Camera at
    origin looking down +Z; pixels in y-down screen coords."""
    M = int(starts.shape[0])
    if M == 0:
        if background_rgb is not None:
            return background_rgb.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        return torch.zeros(H, W, 3, dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = (xx - cx) / fx
    v = (yy - cy) / fy
    z = torch.ones_like(u)
    ray_dirs = torch.stack([u, v, z], dim=-1)
    ray_dirs = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True)
    flat_dirs = ray_dirs.view(-1, 3)
    N = flat_dirs.shape[0]

    radius = torch.as_tensor(radius, device=device, dtype=torch.float32)
    if radius.ndim == 0:
        radius = radius.expand(M)

    ba = ends - starts
    ba_len = torch.linalg.norm(ba, dim=1).clamp(min=1e-6)
    ba_norm = ba / ba_len.unsqueeze(1)

    z_min = float(min(starts[:, 2].min().item(), ends[:, 2].min().item()))
    z_near = max(0.05, z_min - float(radius.max().item()))

    # Union of per-capsule screen-space bboxes — pixels outside can't hit any
    # capsule, so intersection only runs on the relevant subset of the canvas.
    sz = starts[:, 2].clamp(min=z_near)
    ez = ends[:, 2].clamp(min=z_near)
    sx_p = starts[:, 0] * fx / sz + cx
    sy_p = starts[:, 1] * fy / sz + cy
    ex_p = ends[:, 0] * fx / ez + cx
    ey_p = ends[:, 1] * fy / ez + cy
    # Projected radius using the closer endpoint — conservative bbox.
    r_pix = radius * fx / torch.minimum(sz, ez)
    pad = 2.0
    xmin_t = (torch.minimum(sx_p, ex_p) - r_pix - pad).floor().long().clamp(min=0, max=W)
    xmax_t = (torch.maximum(sx_p, ex_p) + r_pix + pad).ceil().long().clamp(min=0, max=W)
    ymin_t = (torch.minimum(sy_p, ey_p) - r_pix - pad).floor().long().clamp(min=0, max=H)
    ymax_t = (torch.maximum(sy_p, ey_p) + r_pix + pad).ceil().long().clamp(min=0, max=H)
    # One stack→tolist sync amortizes the GPU→CPU read over all M bboxes.
    bboxes_cpu = torch.stack([xmin_t, ymin_t, xmax_t, ymax_t], dim=1).tolist()
    coarse_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    for xmin_i, ymin_i, xmax_i, ymax_i in bboxes_cpu:
        if xmax_i > xmin_i and ymax_i > ymin_i:
            coarse_mask[ymin_i:ymax_i, xmin_i:xmax_i] = True

    # Analytic ray-capsule intersection, one pass over the masked pixels.
    INF = float("inf")
    flat_t = torch.full((N,), INF, device=device, dtype=torch.float32)
    flat_m_idx = torch.full((N,), -1, device=device, dtype=torch.long)
    active_idx = torch.nonzero(coarse_mask.view(-1), as_tuple=False).squeeze(1)
    if active_idx.numel() > 0:
        # Cap per-chunk (K, M) tensors to ~4M elements to bound peak memory.
        chunk_max = max(1, int(4_000_000 / max(M, 1)))
        for i0 in range(0, active_idx.numel(), chunk_max):
            sub = active_idx[i0 : i0 + chunk_max]
            t_KM = _ray_capsule_t(
                flat_dirs[sub], starts, ends, ba_norm, ba_len, radius,
            )
            t_min, m_idx = t_KM.min(dim=1)
            hit = t_min < INF
            if hit.any():
                winners = sub[hit]
                flat_t[winners] = t_min[hit]
                flat_m_idx[winners] = m_idx[hit]

    # Shade via analytic normal (P - closest point on segment).
    out = torch.zeros((N, 3), dtype=torch.float32, device=device)
    if background_rgb is not None:
        out = background_rgb.to(device=device, dtype=torch.float32).reshape(N, 3).clone()
    hit_idx = torch.nonzero(flat_m_idx >= 0, as_tuple=False).squeeze(1)
    if hit_idx.numel() > 0:
        rd = flat_dirs[hit_idx]
        t_h = flat_t[hit_idx]
        m_h = flat_m_idx[hit_idx]
        p_hit = rd * t_h.unsqueeze(-1)

        A_h = starts[m_h]
        n_h = ba_norm[m_h]
        L_h = ba_len[m_h]
        proj = ((p_hit - A_h) * n_h).sum(-1).clamp(min=0.0)
        proj = torch.minimum(proj, L_h)
        C_h = A_h + proj.unsqueeze(-1) * n_h
        normals = p_hit - C_h
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        col = colors[m_h, :3]
        if flat_shade:
            # Solid per-limb color (OpenPose look) — no lighting/depth.
            out[hit_idx] = col
            return out.view(H, W, 3).clamp(0.0, 1.0)
        # SCAIL Blinn-Phong, headlight along +Z.
        diff = torch.clamp(-(normals[:, 2]), min=0.0)
        diffuse = 0.45 + 0.55 * diff

        view_dir = -rd
        half_dir = view_dir.clone()
        half_dir[:, 2] -= 1.0
        half_dir = half_dir / half_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        spec = torch.clamp((normals * half_dir).sum(dim=-1), min=0.0).pow(32)

        # Mild depth fade.
        z_vals = p_hit[:, 2]
        z_lo, z_hi = float(z_vals.min().item()), float(z_vals.max().item())
        if z_hi - z_lo > 1e-6:
            fade = 1.0 - (z_vals - z_lo) / (z_hi - z_lo)
            depth_factor = 0.85 + 0.15 * fade
        else:
            depth_factor = torch.ones_like(z_vals)

        base = col * diffuse.unsqueeze(-1) * depth_factor.unsqueeze(-1)
        highlight = (0.5 * spec * depth_factor).unsqueeze(-1)
        out[hit_idx] = base + highlight

    return out.view(H, W, 3).clamp(0.0, 1.0)


def render_pose_data_capsules(
    pose_data: Dict[str, Any],
    *,
    frame_idx: int,
    W: int,
    H: int,
    background: Optional[torch.Tensor] = None,
    composite: str = "over",
    radius_m: float = 0.025,
    include_hands: bool = False,
    palette: str = "scail",
    person_brightness_falloff: float = 0.0,
    flat_shade: bool = False,
    hand_radius_scale: float = 0.4,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Render a frame's pose_data as 3D capsules through the per-person camera.
    Returns (H, W, 3) fp32 in [0, 1].

    `composite='over'` paints over `background` (black if None); 'mesh_only'
    uses a black canvas. `radius_m` is in meters; hand limbs use
    `radius_m * hand_radius_scale`. fx/fy come from each person's `focal_length`.
    """
    persons = pose_data["frames"][frame_idx]
    if device is None:
        device = comfy.model_management.get_torch_device()

    # SAM3DBody shares one camera across the clip — use the first valid person.
    fx = fy = float(min(H, W))
    for person in persons:
        f = person.get("focal_length")
        if f is None:
            continue
        fx = fy = float(np.asarray(f, dtype=np.float32).reshape(-1)[0])
        break
    cx, cy = W * 0.5, H * 0.5

    starts_np, ends_np, colors_np, is_hand_np = _build_specs_from_pose(
        persons, pose_data, include_hands=include_hands, palette=palette,
        person_brightness_falloff=person_brightness_falloff,
    )

    bg_t: Optional[torch.Tensor] = None
    if composite == "over" and background is not None:
        bg_t = background.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if bg_t.shape[:2] != (H, W):
            bg_t = bg_t.permute(2, 0, 1).unsqueeze(0)
            bg_t = torch.nn.functional.interpolate(
                bg_t, size=(H, W), mode="bilinear", align_corners=False,
            )
            bg_t = bg_t.squeeze(0).permute(1, 2, 0).contiguous()

    if starts_np.shape[0] == 0:
        if bg_t is not None:
            return bg_t
        return torch.zeros(H, W, 3, dtype=torch.float32, device=device)

    starts_t = torch.from_numpy(starts_np).to(device=device, dtype=torch.float32)
    ends_t = torch.from_numpy(ends_np).to(device=device, dtype=torch.float32)
    colors_t = torch.from_numpy(colors_np).to(device=device, dtype=torch.float32)
    radii_np = np.where(is_hand_np, radius_m * hand_radius_scale, radius_m).astype(np.float32)
    radii_t = torch.from_numpy(radii_np).to(device=device, dtype=torch.float32)

    return _render_capsules_torch(
        starts_t, ends_t, colors_t,
        H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
        radius=radii_t,
        background_rgb=bg_t,
        device=device,
        flat_shade=flat_shade,
    )
