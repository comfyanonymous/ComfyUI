import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from comfy.ldm.sam3d_body.utils import prepare_batch
from comfy.ldm.sam3.tracker import unpack_masks
from comfy.ldm.sam3d_body.model.model import SAM3DBody


import comfy.model_management
from comfy_api.latest import io
import comfy.utils

from tqdm import tqdm

def _bbox_from_mask(mask: torch.Tensor) -> Optional[torch.Tensor]:
    """xyxy bounds of a binary mask, with sub-5px speckles filtered out."""
    m = mask[..., 0] if mask.dim() == 3 else mask
    m_bool = m > 0
    if not m_bool.any():
        return None
    t = m_bool.to(torch.float32)[None, None]
    eroded = -F.max_pool2d(-t, kernel_size=5, stride=1, padding=2)
    keep = eroded[0, 0] > 0.5
    if not keep.any():
        keep = m_bool
    ys, xs = torch.where(keep)
    return torch.stack([
        xs.min().float(), ys.min().float(),
        (xs.max() + 1).float(), (ys.max() + 1).float(),
    ])


def inputs_from_sam3_track(track_data, B: int, H: int, W: int):
    """Unpack SAM3_TRACK_DATA into per-frame per-object bboxes + masks at image
    resolution. Returns (per_frame_bboxes, per_frame_masks) or
    (None, None) when the track is empty / frame count doesn't match"""

    packed = track_data.get("packed_masks") if isinstance(track_data, dict) else None
    if packed is None:
        return None, None
    unpacked = unpack_masks(packed)  # (N, K, Hm, Wm) bool
    N, K = unpacked.shape[:2]
    if N != B or K == 0:
        return None, None
    Hm, Wm = unpacked.shape[2], unpacked.shape[3]
    resized = F.interpolate(
        unpacked.float().reshape(N * K, 1, Hm, Wm),
        size=(H, W), mode="bilinear", align_corners=False,
    )
    arr = (resized > 0.5).to(torch.uint8).reshape(N, K, H, W).cpu()

    per_frame_masks = [arr[f, :, :, :, None].contiguous() for f in range(N)]
    full_frame_bbox = torch.tensor([0.0, 0.0, float(W), float(H)], dtype=torch.float32)
    per_frame_bboxes = []
    for f in range(N):
        derived = []
        for k in range(K):
            b = _bbox_from_mask(arr[f, k])
            derived.append(b if b is not None else full_frame_bbox)
        per_frame_bboxes.append(torch.stack(derived, dim=0))
    return per_frame_bboxes, per_frame_masks


# Soft budget for the batched Predict path
BATCHED_CROPS_PER_CHUNK = 64


def cam_int_from_fov(height: int, width: int, fov_degrees: float) -> Optional[torch.Tensor]:
    """(1,3,3) intrinsic matrix from a vertical FOV in degrees. Matches MoGe2's
    convention (vertical focal for both axes). Returns None for fov<=0 so the
    caller falls back to prepare_batch's diagonal-focal default."""
    if fov_degrees <= 0:
        return None
    focal = height / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return torch.tensor(
        [[[focal, 0.0, width / 2.0],
          [0.0, focal, height / 2.0],
          [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )


def cam_int_from_moge(moge_geometry, height: int, width: int) -> Optional[torch.Tensor]:
    """(1,3,3) intrinsic matrix from a MoGe geometry payload. Uses MoGe's
    vertical focal for both axes; forces principal point to image center
    (overrides MoGe's predicted cx/cy to match prepare_batch's convention)."""
    if moge_geometry is None:
        return None
    # MOGE_GEOMETRY is a dict with optional keys (see comfy_extras/nodes_moge.py).
    K_norm = moge_geometry.get("intrinsics") if isinstance(moge_geometry, dict) else None
    if K_norm is None:
        return None
    if K_norm.ndim == 3:
        K_norm = K_norm[0]
    # MoGe stores fy in height-units (multiply by H to get pixels); vfov = fy.
    fy_norm = float(K_norm[1, 1].item())
    focal = fy_norm * height
    return torch.tensor(
        [[[focal, 0.0, width / 2.0],
          [0.0, focal, height / 2.0],
          [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )


def run_batched_single_chunk(
    inner: SAM3DBody,
    frames_rgb: List[torch.Tensor],
    per_frame_boxes: List[torch.Tensor],
    per_frame_masks: Optional[List[torch.Tensor]],
    image_size: Tuple[int, int],
    inference_type: str,
    K: int,
    cam_int: Optional[torch.Tensor] = None,
) -> List[List[Dict[str, Any]]]:
    """Run a SINGLE chunk of frames through run_inference in one forward."""
    N = len(frames_rgb)
    total = N * K

    # Reset stateful caches on the model
    for attr in ("batch", "image_embeddings", "output"):
        if hasattr(inner, attr):
            setattr(inner, attr, None)
    inner.prev_prompt = []

    boxes_stacked = torch.stack(
        [per_frame_boxes[f][k] for f in range(N) for k in range(K)], dim=0
    )
    img_per_crop = [frames_rgb[f] for f in range(N) for _ in range(K)]

    if per_frame_masks is not None:
        # Broadcast a single-mask bundle to per-bbox: when the user supplied one
        # mask but multiple bboxes per frame, each bbox gets the same mask.
        flat_masks = []
        for f in range(N):
            mf = per_frame_masks[f]
            if mf.shape[0] == 1 and K > 1:
                mf = mf.repeat_interleave(K, dim=0)
            flat_masks.extend([mf[k] for k in range(K)])
        masks_stacked = torch.stack(flat_masks, dim=0)
        masks_score = torch.ones(total, dtype=torch.float32)
    else:
        masks_stacked = None
        masks_score = None

    batch = prepare_batch(
        img_per_crop, boxes_stacked,
        input_size=image_size,
        masks=masks_stacked, masks_score=masks_score, cam_int=cam_int,
    )
    device = comfy.model_management.get_torch_device()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    inner._initialize_batch(batch)

    outputs = inner.run_inference(
        img_per_crop,
        batch,
        inference_type=inference_type,
        thresh_wrist_angle=1.4,
    )

    if inference_type == "full":
        pose_output, batch_lhand, batch_rhand, _, _ = outputs
    else:
        pose_output = outputs
        batch_lhand = batch_rhand = None

    out = {k: v.cpu().numpy() for k, v in pose_output["mhr"].items()
           if v is not None and k != "faces"}

    # Snapshot batch['bbox'] to CPU before we release `batch` references
    batch_bbox_cpu = batch["bbox"][0].cpu().numpy()
    lhand_bboxes = rhand_bboxes = None
    if inference_type == "full" and batch_lhand is not None and batch_rhand is not None:
        lhand_bboxes = [_bbox_from_center_scale(batch_lhand, i) for i in range(total)]
        rhand_bboxes = [_bbox_from_center_scale(batch_rhand, i) for i in range(total)]

    del pose_output, batch, batch_lhand, batch_rhand, outputs

    frames_out: List[List[Dict[str, Any]]] = []
    for f in range(N):
        persons: List[Dict[str, Any]] = []
        for k in range(K):
            idx = f * K + k
            p: Dict[str, Any] = {
                "bbox": batch_bbox_cpu[idx],
                "focal_length": out["focal_length"][idx],
                "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                "pred_vertices": out["pred_vertices"][idx],
                "pred_cam_t": out["pred_cam_t"][idx],
                "pred_pose_raw": out["pred_pose_raw"][idx],
                "global_rot": out["global_rot"][idx],
                "body_pose_params": out["body_pose"][idx],
                "hand_pose_params": out["hand"][idx],
                "scale_params": out["scale"][idx],
                "shape_params": out["shape"][idx],
                "expr_params": out["face"][idx],
                "mask": (per_frame_masks[f][k] if per_frame_masks[f].shape[0] > 1 else per_frame_masks[f][0])
                        if per_frame_masks is not None else None,
                "pred_joint_coords": out["pred_joint_coords"][idx],
                "pred_global_rots": out["joint_global_rots"][idx],
                "mhr_model_params": out["mhr_model_params"][idx],
                # 238 face landmarks from sapiens-308 (indices 70..308 of the pre-slice keypoint tensor).
                "pred_face_keypoints_3d": out["pred_face_keypoints_3d"][idx] if "pred_face_keypoints_3d" in out else None,
                "pred_face_keypoints_2d": out["pred_face_keypoints_2d"][idx] if "pred_face_keypoints_2d" in out else None,
            }
            if lhand_bboxes is not None:
                p["lhand_bbox"] = lhand_bboxes[idx]
                p["rhand_bbox"] = rhand_bboxes[idx]
            persons.append(p)
        frames_out.append(persons)
    return frames_out


def run_batched_frames(
    inner: SAM3DBody,
    frames_rgb: List[torch.Tensor],
    per_frame_boxes: List[torch.Tensor],
    per_frame_masks: Optional[List[torch.Tensor]],
    image_size: Tuple[int, int],
    inference_type: str,
    cam_int: Optional[torch.Tensor] = None,
    pbar: Optional[comfy.utils.ProgressBar] = None,
    crops_per_chunk: int = BATCHED_CROPS_PER_CHUNK,
) -> List[List[Dict[str, Any]]]:
    """Run the clip through chunked batched run_inference calls.

    Supports K persons per frame (K must be the same across frames — padded
    externally). Splits frames into chunks so chunk_frames * K <= budget; each
    chunk is one body forward + optional hand forwards over its person-crops
    """
    N = len(frames_rgb)
    assert N > 0, "empty frame list"
    K_set = {len(b) for b in per_frame_boxes}
    assert len(K_set) == 1, f"batched path requires same bbox count per frame, got {K_set}"
    K = K_set.pop()
    assert K >= 1, "need at least one bbox per frame"

    chunk_frames = max(1, crops_per_chunk // K)
    results: List[List[Dict[str, Any]]] = []
    with tqdm(total=N, desc="SAM3D body inference") as t:
        for start in range(0, N, chunk_frames):
            end = min(N, start + chunk_frames)
            sub_frames = frames_rgb[start:end]
            sub_boxes = per_frame_boxes[start:end]
            sub_masks = None if per_frame_masks is None else per_frame_masks[start:end]
            chunk_result = run_batched_single_chunk(
                inner, sub_frames, sub_boxes, sub_masks,
                image_size, inference_type, K,
                cam_int=cam_int,
            )
            results.extend(chunk_result)
            t.update(end - start)
            if pbar is not None:
                pbar.update(end - start)
            # Drop GPU caches so the next chunk starts from a clean allocator state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def _bbox_from_center_scale(batch, idx: int) -> np.ndarray:
    cx = batch["bbox_center"].flatten(0, 1)[idx][0].item()
    cy = batch["bbox_center"].flatten(0, 1)[idx][1].item()
    sx = batch["bbox_scale"].flatten(0, 1)[idx][0].item()
    sy = batch["bbox_scale"].flatten(0, 1)[idx][1].item()
    return np.array([cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2], dtype=np.float32)

# Wire types and small helpers shared across the SAM 3D Body node modules.

def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI image tensor (any shape, float 0..1) → uint8 tensor in [0, 255] on CPU."""
    return (image * 255.0).clamp(0.0, 255.0).to(dtype=torch.uint8, device="cpu")


def compute_canonical_colors(model) -> Dict[str, np.ndarray]:
    """Canonical rest-pose data for shader color lookups: positions (Nv,3),
    norm (Nv,3 in [0,1]), face_mask, head_mask, and face_region_rgb
    (per-region painted color from the .safetensors)."""
    verts = model.head_pose.canonical_vertices().float().cpu().numpy()
    faces = model.head_pose.faces.cpu().numpy()

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    vn = np.zeros_like(verts, dtype=np.float32)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    ln[ln < 1e-8] = 1.0
    vn = vn / ln
    norm_map = ((vn + 1.0) * 0.5).astype(np.float32)

    face_mask = _compute_face_mask(model)
    # Head: above jaw-neck (y>1.43) and narrower than shoulders (|x|<0.11).
    # Ears reach |x|≈0.09; shoulders start at |x|≈0.20.
    head_mask = (verts[:, 1] > 1.43) & (np.abs(verts[:, 0]) < 0.11)

    # Painted per-vertex face region RGB ships in the model .safetensors as
    # `head_pose.face_region_rgb` and gets loaded by load_state_dict.
    face_region_rgb = model.head_pose.face_region_rgb.detach().float().cpu().numpy()

    return {
        "positions": verts.astype(np.float32),
        "norm": norm_map,
        "face_mask": face_mask,
        "head_mask": head_mask,
        "face_region_rgb": face_region_rgb,
    }


def compute_hand_vert_mask(model, hand_radius_m: float = 0.15, weight_threshold: float = 0.5) -> np.ndarray:
    """(Nv,) bool mask of hand-region verts. Picks joints within `hand_radius_m`
    of the mhr70 hand keypoint clusters (indices 21..62), then sums sparse LBS
    weights across them; verts above `weight_threshold` are hand verts."""
    head = model.head_pose
    mhr = head.mhr
    device = head.scale_mean.device

    zeros = lambda *s: torch.zeros(1, *s, device=device)
    out = head.mhr_forward(
        global_trans=zeros(3),
        global_rot=zeros(3),
        body_pose_params=zeros(130),
        hand_pose_params=zeros(head.num_hand_comps * 2),
        scale_params=zeros(head.num_scale_comps),
        shape_params=zeros(head.num_shape_comps),
        expr_params=zeros(head.num_face_comps),
        return_keypoints=True,
        return_joint_coords=True,
    )
    # Output order with these flags: (verts, kp, jcoords). See mhr_head.mhr_forward.
    _, kp, jcoords = out[0], out[1], out[2]
    kp = kp[0, :70].cpu().numpy()
    jcoords = jcoords[0].cpu().numpy()

    right_center = kp[21:42].mean(axis=0)
    left_center = kp[42:63].mean(axis=0)
    j_dist_r = np.linalg.norm(jcoords - right_center, axis=1)
    j_dist_l = np.linalg.norm(jcoords - left_center, axis=1)
    is_hand_joint = (j_dist_r < hand_radius_m) | (j_dist_l < hand_radius_m)

    lbs_w = mhr.lbs_skin_weights.cpu().numpy()
    lbs_v = mhr.lbs_vert_indices.cpu().numpy()
    lbs_j = mhr.lbs_skin_indices.cpu().numpy()
    is_hand_joint_f = is_hand_joint.astype(np.float32)

    n_verts = mhr.NUM_VERTS
    hand_mass = np.zeros(n_verts, dtype=np.float32)
    np.add.at(hand_mass, lbs_v, lbs_w * is_hand_joint_f[lbs_j])
    return hand_mass >= weight_threshold


def _compute_face_mask(model, disp_threshold_m: float = 1e-4) -> np.ndarray:
    """(Nv,) bool mask of verts that move with face expression. Sweeps each of
    the 72 expression axes at coef=+1.0 and flags any vert that moves more
    than `disp_threshold_m` for at least one axis."""
    head = model.head_pose
    device = head.scale_mean.device
    num_face = head.num_face_comps

    zeros = lambda *s: torch.zeros(1, *s, device=device)
    neutral_kw = dict(
        global_trans=zeros(3),
        global_rot=zeros(3),
        body_pose_params=zeros(130),
        hand_pose_params=zeros(head.num_hand_comps * 2),
        scale_params=zeros(head.num_scale_comps),
        shape_params=zeros(head.num_shape_comps),
        expr_params=zeros(num_face),
    )
    v0 = head.mhr_forward(**neutral_kw).cpu().numpy()[0]  # (Nv, 3)

    face_mask = np.zeros(v0.shape[0], dtype=bool)
    for axis in range(num_face):
        expr = zeros(num_face)
        expr[0, axis] = 1.0
        kw = dict(neutral_kw)
        kw["expr_params"] = expr
        v = head.mhr_forward(**kw).cpu().numpy()[0]
        face_mask |= (np.linalg.norm(v - v0, axis=1) > disp_threshold_m)
    return face_mask


def jet_colormap(s: np.ndarray) -> np.ndarray:
    """matplotlib jet, (N,) in [0,1] -> (N, 3) float32 RGB."""
    s = np.asarray(s, dtype=np.float32).clip(0.0, 1.0)
    r = np.interp(s, [0.0, 0.35, 0.66, 0.89, 1.0], [0.0, 0.0, 1.0, 1.0, 0.5])
    g = np.interp(s, [0.0, 0.125, 0.375, 0.64, 0.91, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    b = np.interp(s, [0.0, 0.11, 0.34, 0.65, 1.0], [0.5, 1.0, 1.0, 0.0, 0.0])
    return np.stack([r, g, b], axis=-1).astype(np.float32)
