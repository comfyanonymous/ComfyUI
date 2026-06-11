"""SAM 3D Body — Predict + Smooth nodes and their inference helpers."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from scipy.signal import savgol_coeffs

import comfy.model_management
import comfy.utils
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override
import folder_paths

from comfy.ldm.sam3d_body.model.model import SAM3DBody
from comfy_extras.sam3d_body.utils import (
        apply_camera_override,
        cam_int_from_fov,
        inputs_from_sam3_track,
        run_batched_frames,
        run_batched_single_chunk,
        compute_canonical_colors,
        compute_hand_vert_mask,
    )

from comfy_extras.sam3d_body.rasterizer import render_pose_data_torch as render_pose_data
from comfy_extras.sam3d_body.export.capsules import render_pose_data_capsules
from comfy_extras.sam3d_body.export.openpose_2d import render_pose_data_openpose
from comfy_extras.sam3d_body import face_expression as fx
from comfy_extras.sam3d_body.utils import image_to_uint8


SAM3TrackData = io.Custom("SAM3_TRACK_DATA")
# MHRPoseData = SAM3DBody_Predict's native output (carries mhr_model_params,
# shape_params, expr_params, MHR70 keypoint layout, canonical_colors keyed to
# MHR mesh, hand_vert_mask from MHR LBS). The export-side consumers
# (BuildPoseGLB / SavePoseBVH in comfy_extras/nodes_save_3d.py) also accept
# KIMODO_POSE_DATA via a MultiType union — those types are mirrored there.
MHRPoseData = io.Custom("MHR_POSE_DATA")
SAM3DBodyModel = io.Custom("SAM3D_BODY_MODEL")

# Loader

class SAM3DBody_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBody_Loader",
            display_name="Load SAM3D Body Model",
            category="image/detection",
            inputs=[
                io.Combo.Input(
                    "model_file",
                    options=folder_paths.get_filename_list("detection"),

                ),
            ],
            outputs=[SAM3DBodyModel.Output(display_name="sam3d_body_model")],
        )


    @classmethod
    def execute(cls, model_file) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("detection", model_file)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        sd = {k.replace(".layers.0.0.", ".layers.0."): v for k, v in sd.items()}

        load_device = comfy.model_management.get_torch_device()
        weight_dtype = comfy.utils.weight_dtype(sd)
        torch_dtype = comfy.model_management.unet_dtype(
            device=load_device, model_params=-1, weight_dtype=weight_dtype,
        )
        manual_cast_dtype = comfy.model_management.unet_manual_cast(torch_dtype, load_device)
        operations = comfy.ops.pick_operations(torch_dtype, manual_cast_dtype, load_device=load_device, disable_fast_fp8=True)

        model = SAM3DBody(dtype=torch_dtype, operations=operations)
        model.load_state_dict(sd, strict=False)

        model.eval()
        model.backbone_dtype = torch_dtype
        model._sam3d_image_size = model.image_size

        model._sam3d_canonical_colors = compute_canonical_colors(model)
        model._sam3d_hand_vert_mask = compute_hand_vert_mask(model)

        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=load_device,
            offload_device=comfy.model_management.unet_offload_device(),
            size=comfy.model_management.module_size(model),
        )
        return io.NodeOutput(patcher)

# Predict

def _per_frame_bboxes_from_detections(bboxes, B: int):
    # BoundingBox payload (RT-DETR etc.): dict | list[dict] | list[list[dict]].
    if isinstance(bboxes, dict):
        norm = [[bboxes]]
    elif not bboxes:
        return None
    elif isinstance(bboxes[0], dict):
        norm = [bboxes]  # flat list → same detections every frame
    else:
        norm = list(bboxes)
    if len(norm) == 1:
        norm = norm * B
    norm = (norm + [[]] * B)[:B]
    out = []
    for frame in norm:
        if frame:
            boxes = torch.tensor(
                [[d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]] for d in frame],
                dtype=torch.float32,
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        out.append(boxes)
    return out


class SAM3DBody_Predict(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBody_Predict",
            display_name="Run SAM3D Body Prediction",
            category="image/detection",
            inputs=[
                SAM3DBodyModel.Input("sam3d_body_model"),
                io.Image.Input("image"),
                SAM3TrackData.Input(
                    "track_data", optional=True,
                    tooltip=("Tracking data from SAM3 Video Track, required for multi-person detection"),
                ),
                io.BoundingBox.Input(
                    "bboxes", optional=True, force_input=True,
                    tooltip=(
                        "Per-frame bounding boxes used for better detection. Can be used as an alternative to tracking data. "
                    ),
                ),
                io.Boolean.Input(
                    "run_hand_refinement", default=True,
                    tooltip="Improves hand pose at the cost of extra inference time and memory use"),
                io.Float.Input(
                    "fov",
                    tooltip=(
                        "Vertical FoV in degrees. Affects predicted depth and absolute scale. 0 = fall back to ~53° (16:9)."
                    ),
                ),
                io.Int.Input(
                    "chunk_size", #TODO: automate?
                    default=64, min=1, max=512, step=1, advanced=True,
                    tooltip=(
                        "Max frames to process as a batch. Larger values utilize more VRAM for faster inference."
                    ),
                ),
            ],
            outputs=[MHRPoseData.Output("mhr_pose_data")],
        )

    @classmethod
    def execute(cls, sam3d_body_model, image, sam3_track_data=None, bboxes=None, run_hand_refinement=True, fov=0.0, chunk_size=64) -> io.NodeOutput:
        comfy.model_management.load_model_gpu(sam3d_body_model)
        inner: SAM3DBody = sam3d_body_model.model

        B, H, W, _ = image.shape
        image_size = getattr(inner, "_sam3d_image_size", (512, 512))

        # Precedence: SAM3 track (masks + boxes) > detector boxes > full-frame fallback.
        per_frame_bboxes, per_frame_masks = (None, None)
        if sam3_track_data is not None:
            per_frame_bboxes, per_frame_masks = inputs_from_sam3_track(sam3_track_data, B, H, W)
        if per_frame_bboxes is None and bboxes:
            per_frame_bboxes = _per_frame_bboxes_from_detections(bboxes, B)
            per_frame_masks = None
        if per_frame_bboxes is None:
            # No track or detector boxes — single-person full-frame fallback.
            full_frame_bbox = torch.tensor([[0.0, 0.0, float(W), float(H)]], dtype=torch.float32)
            per_frame_bboxes = [full_frame_bbox.clone() for _ in range(B)]
            per_frame_masks = None
        inference_type = "full" if run_hand_refinement else "body"
        # fov > 0 sets intrinsics; else None falls back to prepare_batch's diagonal default.
        cam_int = cam_int_from_fov(int(H), int(W), float(fov))

        frames_rgb: List[Optional[torch.Tensor]] = []
        for f in range(B):
            if per_frame_bboxes[f].shape[0] == 0:
                frames_rgb.append(None)
            else:
                frames_rgb.append(image_to_uint8(image[f]))

        # Batched path requires uniform non-zero K across all frames.
        bbox_counts = {per_frame_bboxes[f].shape[0] for f in range(B) if frames_rgb[f] is not None}
        can_batch = (
            len(bbox_counts) == 1
            and 0 not in bbox_counts
            and all(frames_rgb[f] is not None for f in range(B))
        )

        frames_out: List[List[Dict[str, Any]]] = []
        pbar = comfy.utils.ProgressBar(B)

        if can_batch and B > 0:
            frames_out = run_batched_frames(
                inner, frames_rgb, per_frame_bboxes, per_frame_masks,
                image_size, inference_type,
                cam_int=cam_int,
                pbar=pbar,
                crops_per_chunk=int(chunk_size),
            )
        else:
            # Mixed K per frame — call the batched path once per frame.
            for f in range(B):
                if frames_rgb[f] is None or per_frame_bboxes[f].shape[0] == 0:
                    frames_out.append([])
                    pbar.update(1)
                    continue
                mask_f = [per_frame_masks[f]] if per_frame_masks is not None else None
                chunk = run_batched_single_chunk(
                    inner, [frames_rgb[f]], [per_frame_bboxes[f]], mask_f,
                    image_size, inference_type,
                    K=int(per_frame_bboxes[f].shape[0]),
                    cam_int=cam_int,
                )
                frames_out.append(chunk[0])
                pbar.update(1)

        mhr_pose_data = {
            "frames": frames_out,
            "faces": inner.head_pose.faces.cpu().numpy(),
            "image_size": (int(H), int(W)),
            "canonical_colors": getattr(inner, "_sam3d_canonical_colors", None),
            "hand_vert_mask": getattr(inner, "_sam3d_hand_vert_mask", None),
        }
        return io.NodeOutput(mhr_pose_data)


class SAM3DBody_FaceExpression(io.ComfyNode):
    """Drive MHR face blendshapes from the core MediaPipe Face Landmarker.

    Detects per-frame faces, IoU-matches each to a tracked person, maps the 52
    ARKit blendshapes onto MHR's 72-axis `expr_params`, and re-runs MHR forward
    so pred_vertices/pred_keypoints reflect the new expression.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBody_FaceExpression",
            description="Drive MHR face blendshapes from the core MediaPipe Face Landmarker.",
            display_name="Face Expression to SAM3D Body", #TODO: better name?
            category="image/detection",
            inputs=[
                SAM3DBodyModel.Input("sam3d_body_model"),
                MHRPoseData.Input("mhr_pose_data"),
                io.Image.Input("image"),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=4.0, step=0.05,
                    tooltip="Global multiplier on all blendshapes. >1 exaggerates.",
                ),
                io.Float.Input(
                    "mouth_strength", default=1.0, min=0.0, max=4.0, step=0.05,
                    tooltip="Multiplier on mouth/jaw shapes. MediaPipe's jawOpen saturates near 1.0.",
                    advanced=True,
                ),
                io.Float.Input(
                    "eye_strength", default=2.0, min=0.0, max=4.0, step=0.05,
                    tooltip="Multiplier on eye shapes. MediaPipe rarely exceeds 0.5; 2-3x often needed.",
                    advanced=True,
                ),
                io.Float.Input(
                    "brow_strength", default=2.0, min=0.0, max=4.0, step=0.05,
                    tooltip="Multiplier on brow/cheek/sneer shapes. MediaPipe outputs ~0.1-0.3; 2-3x.",
                    advanced=True,
                ),
                io.Float.Input(
                    "input_threshold", default=0.02, min=0.0, max=0.5, step=0.01,
                    tooltip=(
                        "Deadzone on MediaPipe's raw output (below = zero, above = linear remap). "
                    ),
                    advanced=True,
                ),
                io.Int.Input(
                    "blendshape_smooth_window", default=7, min=1, max=31, step=2,
                    tooltip=(
                        "Gaussian window on MediaPipe's per-frame signal before MHR mapping. "
                        "MediaPipe's raw output swings 30-70% frame-to-frame on static faces. "
                        "1 = disabled. Use odd values."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                MHRPoseData.Output("mhr_pose_data"),
            ],
        )

    @classmethod
    def execute(cls, mhr_pose_data, sam3d_body_model, image,
                strength=1.0, mouth_strength=1.0, eye_strength=2.0, brow_strength=2.0,
                input_threshold=0.02, blendshape_smooth_window=7) -> io.NodeOutput:

        comfy.model_management.load_model_gpu(sam3d_body_model)
        inner: SAM3DBody = sam3d_body_model.model

        frames = mhr_pose_data["frames"]
        B = len(frames)
        if B == 0:
            return io.NodeOutput(mhr_pose_data)

        img_np = (image * 255.0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
        new_frames: List[List[Dict[str, Any]]] = [[dict(p) for p in f] for f in frames]

        max_persons = max((len(f) for f in new_frames), default=0)
        per_person_coefs: List[List[Optional[Dict[str, float]]]] = [
            [None] * B for _ in range(max_persons)
        ]
        pbar = comfy.utils.ProgressBar(B)
        n_total_frames_with_persons = 0

        crop_factor = 1.2

        # Auto-pick full-frame vs per-person crops. BlazeFace full-range needs
        # ≥32px face in its 192px input; below that we escalate to per-person crops
        H_img0, W_img0 = img_np.shape[1], img_np.shape[2]
        min_bbox_px = 32.0 * max(H_img0, W_img0) / (192.0 * 0.20)  # Face height ≈ 20% of body-bbox height.
        use_per_person_crops = any(
            (p["bbox"][3] - p["bbox"][1]) < min_bbox_px
            for persons in new_frames for p in persons
        )

        for fi in tqdm(range(B), desc="SAM3D face expression detect"):
            persons = new_frames[fi]
            img_fi = img_np[min(fi, img_np.shape[0] - 1)]

            if not persons:
                pbar.update(1)
                continue
            n_total_frames_with_persons += 1

            person_bboxes = [np.asarray(p["bbox"], dtype=np.float32) for p in persons]
            H_img, W_img = img_fi.shape[:2]

            if use_per_person_crops:
                # One MP call per person on a tight head crop — recovers small/
                # distant faces that the full-frame 192px BlazeFace would miss.
                for pid, pb in enumerate(person_bboxes):
                    if pid >= max_persons:
                        continue
                    cr = fx.head_crop_from_keypoints(
                        persons[pid].get("pred_keypoints_2d"), crop_factor, W_img, H_img,
                    )
                    if cr is None:
                        cr = fx.head_region_crop(pb, crop_factor, W_img, H_img)
                    faces = fx.detect_faces_in_crop(inner, img_fi, cr, num_faces=1)
                    if not faces:
                        continue
                    # Pick face closest to person bbox center when a neighbor leaks in.
                    pcx, pcy = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])
                    best = min(
                        faces,
                        key=lambda f: (0.5 * (f["bbox_xyxy"][0] + f["bbox_xyxy"][2]) - pcx) ** 2
                                      + (0.5 * (f["bbox_xyxy"][1] + f["bbox_xyxy"][3]) - pcy) ** 2,
                    )
                    per_person_coefs[pid][fi] = best["blendshapes"]
            else:
                faces = inner.face_landmarker.detect_batch([img_fi], num_faces=max(1, len(persons)))[0]
                if faces:
                    face_bboxes = [f["bbox_xyxy"] for f in faces]
                    assignment = fx.assign_faces_to_persons(face_bboxes, person_bboxes)
                    for pid, face_idx in enumerate(assignment):
                        if face_idx is None or pid >= max_persons:
                            continue
                        per_person_coefs[pid][fi] = faces[face_idx]["blendshapes"]

            pbar.update(1)

        # Baseline subtraction. MP has subject-specific rest bias, without subtraction, strength
        # multipliers bake that into every frame.
        # Per-clip needs ~30 frames or it would zero out the expression.
        BASELINE_MIN_FRAMES = 30
        if n_total_frames_with_persons >= BASELINE_MIN_FRAMES:
            for pid in range(max_persons):
                per_person_coefs[pid] = fx.subtract_per_clip_baseline(
                    per_person_coefs[pid], percentile=5.0,
                )
        else:
            logging.warning(
                f"[SAM 3D Body FaceExpression] per-clip baseline subtraction "
                f"needs ~{BASELINE_MIN_FRAMES}+ frames with detections; "
                f"got {n_total_frames_with_persons}. Skipping subtraction."
            )

        # Smooth raw signal AFTER baseline subtraction but BEFORE gap fill
        # MP's per-frame noise gets averaged out at the source.
        bs_win = int(blendshape_smooth_window)
        if bs_win > 1:
            for pid in range(max_persons):
                per_person_coefs[pid] = fx.smooth_blendshape_series(
                    per_person_coefs[pid], window=bs_win,
                )

        for pid in range(max_persons):
            per_person_coefs[pid] = fx.fill_detection_gaps(
                per_person_coefs[pid], method="interpolate", max_gap=12,
            )

        n_written = 0
        for fi in range(B):
            for pid, p in enumerate(new_frames[fi]):
                if pid >= max_persons:
                    continue
                coefs = per_person_coefs[pid][fi]
                if coefs is None:
                    continue
                p["expr_params"] = fx.arkit_to_expr_params(
                    coefs,
                    strength=float(strength),
                    mouth_strength=float(mouth_strength),
                    eye_strength=float(eye_strength),
                    brow_strength=float(brow_strength),
                    input_threshold=float(input_threshold),
                ).astype(np.float32)
                n_written += 1

        if n_written > 0:
            fx.regenerate_mesh_from_params(inner, new_frames)

        new_pose = dict(mhr_pose_data)
        new_pose["frames"] = new_frames

        return io.NodeOutput(new_pose)


class SAM3DBody_Smooth(io.ComfyNode):
    """Reduce frame-to-frame jitter via vertex-space temporal averaging.
    Backs off on mesh-geometry keys when the subject rotates fast (averaging
    across a spin flattens the mesh); camera-space keys still get full
    smoothing.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBody_Smooth",
            description="Reduce frame-to-frame jitter via temporal smoothing.",
            display_name="Smooth SAM3D Body Pose Data",
            category="image/detection",
            inputs=[
                MHRPoseData.Input("mhr_pose_data"),
                io.Float.Input(
                    "strength",
                    default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Smoothing strength. 0 = raw, 1 = smoothed.",
                ),
                io.Combo.Input(
                    "method",
                    options=["gaussian", "savgol"],
                    default="gaussian", advanced=True,
                    tooltip=(
                        "gaussian: symmetric weighted average, best general-purpose smoother./n"
                        "savgol: sliding polynomial fit, preserves sharp peaks."
                    ),
                ),
                io.Int.Input(
                    "window",
                    default=7, min=1, max=51, step=2, advanced=True,
                    tooltip="Temporal window in frames (odd values).",
                ),
                io.Float.Input(
                    "rotation_threshold_deg",
                    default=30.0, min=0.0, max=90.0, step=1.0, advanced=True,
                    tooltip=(
                        "Disables smoothing for this root rotation rate (degree/frame) to preserve fast spins. "
                        "30° suits most content, low values might disable smoothing on ordinary jitter and "
                        "silently impacts quality. 0 = disable."
                    ),
                ),
            ],
            outputs=[MHRPoseData.Output("mhr_pose_data")],
        )

    @classmethod
    def execute(cls, mhr_pose_data, method, window, strength, rotation_threshold_deg) -> io.NodeOutput:
        if strength <= 0.0 or window <= 1:
            return io.NodeOutput(mhr_pose_data)

        frames = mhr_pose_data["frames"]
        B = len(frames)
        if B < 2:
            return io.NodeOutput(mhr_pose_data)
        max_p = max((len(f) for f in frames), default=0)
        if max_p == 0:
            return io.NodeOutput(mhr_pose_data)

        # Geometry keys rotate with the subject, so linear averaging during
        # fast spins flattens the mesh — these get per-frame adaptive strength.
        keys_geom = {
            "pred_vertices", "pred_keypoints_3d", "pred_joint_coords",
            "pred_global_rots", "mhr_model_params", "body_pose_params",
            "global_rot", "pred_pose_raw",
        }
        # Camera / appearance / 2D keys are safe to smooth linearly.
        keys_cam = {
            "pred_cam_t", "pred_keypoints_2d", "focal_length",
            "shape_params", "scale_params", "hand_pose_params", "expr_params",
        }
        all_keys = sorted(keys_geom | keys_cam)

        kernel = _smoothing_kernel(method, window)
        smoothed = [list(f) for f in frames]

        base_blend = float(strength)
        rot_thresh = float(np.deg2rad(max(0.0, rotation_threshold_deg)))

        for pid in range(max_p):
            valid = np.array([pid < len(f) for f in frames], dtype=bool)
            if valid.sum() < 2:
                continue

            # Adaptive blend per frame from `global_rot` (euler ZYX);
            geom_blend = np.full(B, base_blend, dtype=np.float32)
            if rot_thresh > 0.0:
                root_rotmats = []
                valid_root = []
                for fi in range(B):
                    if not valid[fi]:
                        root_rotmats.append(np.eye(3, dtype=np.float32))
                        valid_root.append(False)
                        continue
                    gr = frames[fi][pid].get("global_rot")
                    if gr is None:
                        root_rotmats.append(np.eye(3, dtype=np.float32))
                        valid_root.append(False)
                        continue
                    eul = np.asarray(gr, dtype=np.float32).reshape(3)
                    # ZYX convention: R = Rz @ Ry @ Rx
                    cz, sz = np.cos(eul[0]), np.sin(eul[0])
                    cy, sy = np.cos(eul[1]), np.sin(eul[1])
                    cx, sx = np.cos(eul[2]), np.sin(eul[2])
                    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
                    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
                    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
                    root_rotmats.append(Rz @ Ry @ Rx)
                    valid_root.append(True)
                ang = np.zeros(B, dtype=np.float32)
                for fi in range(1, B):
                    if valid_root[fi] and valid_root[fi - 1]:
                        R_delta = root_rotmats[fi] @ root_rotmats[fi - 1].T
                        cos_a = float(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
                        ang[fi] = abs(np.arccos(cos_a))
                # Peak-smear over ±window/2 — neighbors of a rotated frame
                # must back off too, or the temporal average pulls the rotated
                # pose in and flattens the mesh anyway.
                half = max(1, window // 2)
                ang_smooth = np.zeros_like(ang)
                for fi in range(B):
                    lo = max(0, fi - half)
                    hi = min(B, fi + half + 1)
                    ang_smooth[fi] = ang[lo:hi].max() if hi > lo else 0.0
                # base_blend at no rotation, 0 at threshold.
                ratio = np.clip(ang_smooth / rot_thresh, 0.0, 1.0)
                geom_blend = base_blend * (1.0 - ratio)

            for key in all_keys:
                ref = None
                for fi in range(B):
                    if valid[fi] and key in frames[fi][pid] and frames[fi][pid][key] is not None:
                        ref = np.asarray(frames[fi][pid][key])
                        break
                if ref is None or ref.dtype.kind not in "fiu":
                    continue
                stacked = np.zeros((B,) + ref.shape, dtype=np.float32)
                for fi in range(B):
                    if valid[fi] and key in frames[fi][pid] and frames[fi][pid][key] is not None:
                        stacked[fi] = np.asarray(frames[fi][pid][key], dtype=np.float32)
                    else:
                        stacked[fi] = stacked[fi - 1] if fi > 0 else 0.0
                filtered = _apply_temporal_filter(stacked, kernel)

                if key in keys_geom:
                    b = geom_blend
                    while b.ndim < stacked.ndim:
                        b = b[..., None]
                    out = (1.0 - b) * stacked + b * filtered
                else:
                    out = (1.0 - base_blend) * stacked + base_blend * filtered

                for fi in range(B):
                    if valid[fi]:
                        smoothed[fi][pid] = dict(smoothed[fi][pid])
                        smoothed[fi][pid][key] = out[fi].astype(ref.dtype)

        new_pose = dict(mhr_pose_data)
        new_pose["frames"] = smoothed
        return io.NodeOutput(new_pose)


def _smoothing_kernel(method: str, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if method == "savgol":
        order = 3 if window >= 5 else min(window - 1, 1)
        return savgol_coeffs(window, order).astype(np.float32)
    # gaussian (default)
    sigma = max(1.0, window / 5.0)
    x = np.arange(window) - (window - 1) / 2.0
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _apply_temporal_filter(stacked: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """stacked: (B, *feature_shape). Returns same shape, smoothed over axis 0."""
    B = stacked.shape[0]
    w = len(kernel)
    pad = w // 2
    flat = stacked.reshape(B, -1)                                    # (B, K)
    padded = np.concatenate(
        [np.repeat(flat[:1], pad, axis=0), flat, np.repeat(flat[-1:], pad, axis=0)],
        axis=0,
    )                                                                 # (B + 2*pad, K)
    out = np.zeros_like(flat)
    for i, k in enumerate(kernel):
        out += k * padded[i : i + B]
    return out.reshape(stacked.shape)


# Render

def rainbow_tilt_inputs():
    """Shared rainbow-shader tilt inputs (used by Render and ToGLB schemas)."""
    return [
        io.Float.Input(
            "rainbow_tilt_z", default=-35.0, min=-90.0, max=90.0, step=0.5,
            tooltip="Rotate rainbow jet axis around Z (forward). Differentiates left/right.",
        ),
        io.Float.Input(
            "rainbow_tilt_x", default=0.0, min=-90.0, max=90.0, step=0.5,
            tooltip="Rotate rainbow jet axis around X (right). Differentiates front/back.",
        ),
    ]


def _render_mesh_mode_inputs():
    return [
        io.DynamicCombo.Input(
            "shader",
            options=[
                io.DynamicCombo.Option("default", []),
                io.DynamicCombo.Option("normals", []),
                io.DynamicCombo.Option("rainbow", rainbow_tilt_inputs()),
                io.DynamicCombo.Option("rainbow_face_normal", rainbow_tilt_inputs()),
                io.DynamicCombo.Option("rainbow_face_semantic", rainbow_tilt_inputs()),
                io.DynamicCombo.Option("depth", []),
            ],
            tooltip=(
                "Preset shader. 'normals' = current surface normal in camera "
                "space (OpenGL Y+ normal-map convention: +X→R, +Y→G, +Z→B). "
                "'rainbow' = RealisDance style body-Y jet; the 'rainbow_face_*' "
                "variants override face verts with normal/per-region colors; "
                "'depth' = linear gray."
            ),
        ),
        io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),
        io.Float.Input(
            "person_palette_falloff",
            default=0.6, min=0.1, max=1.0, step=0.05,
            tooltip=(
                "Per-person desaturation toward white: track k gets a "
                "(1 - falloff^k) pastel mix (SCAIL 'softer second person'). 1.0 = off."
            ),
        ),
        io.Combo.Input(
            "region",
            options=["full_body", "hands_only"],
            default="full_body",
            tooltip=(
                "'hands_only' filters faces via the precomputed `hand_vert_mask` "
                "(LBS weights against canonical hand KPs) — isolates the hand "
                "mesh for debugging. Falls back to full mesh if the mask is missing."
            ),
        ),
    ]


def _render_capsules_mode_inputs():
    return [
        io.Float.Input(
            "radius_m", default=0.022, min=0.005, max=0.2, step=0.001,
            tooltip="Capsule radius in meters (SCAIL reference: ~0.022 m).",
        ),
        io.Combo.Input(
            "hand_style",
            options=["disabled", "dwpose", "openpose"],
            default="dwpose",
            tooltip=(
                "Composite 2D OpenPose hands on top of the 3D capsule body "
                "(matches SCAIL — no 3D hand capsules). 'disabled' = no hands. "
                "'dwpose' = solid-blue hand dots; 'openpose' = rainbow dots. "
                "Sticks stay rainbow per-finger either way."
            ),
        ),
        io.Combo.Input(
            "face_style",
            options=["disabled", "full", "eyes_mouth"],
            default="disabled",
            tooltip=(
                "'full' = all face landmarks (sapiens-238 if present, else "
                "rig-fallback ~30). 'eyes_mouth' = rig-fallback subset (~12 "
                "dots: eyes + outer lips only). 'disabled' = no face dots."
            ),
        ),
        io.Float.Input(
            "person_palette_falloff", default=0.6, min=0.1, max=1.0, step=0.05,
            tooltip=(
                "Per-person desaturation: track k blends toward white by "
                "1 - falloff^k. Track 0 stays vivid; 1.0 disables falloff."
            ),
        ),
    ]


def _render_openpose3d_mode_inputs():
    return [
        io.Float.Input(
            "radius_m", default=0.015, min=0.004, max=0.1, step=0.001,
            tooltip="Limb capsule radius in meters (thin = stick-like).",
        ),
        io.Boolean.Input(
            "include_hands", default=True,
            tooltip="Draw 21+21 hand keypoints as 3D capsules.",
        ),
        io.Float.Input(
            "person_palette_falloff", default=0.6, min=0.1, max=1.0, step=0.05,
            tooltip=(
                "Per-person desaturation: track k blends toward white by "
                "1 - falloff^k. Track 0 stays vivid; 1.0 disables falloff."
            ),
        ),
    ]


def _render_openpose_mode_inputs():
    return [
        io.Int.Input(
            "marker_radius_px", default=4, min=1, max=32, step=1,
            tooltip="Body keypoint dot radius (px).",
        ),
        io.Int.Input(
            "stick_width_px", default=4, min=1, max=32, step=1,
            tooltip="Body limb ellipse half-width (px). DWPose default = 4.",
        ),
        io.Float.Input(
            "limb_alpha", default=0.6, min=0.0, max=1.0, step=0.05,
            tooltip="Per-limb alpha. DWPose default = 0.6.",
        ),
        io.Combo.Input(
            "face_style",
            options=["disabled", "full", "eyes_mouth"],
            default="disabled",
            tooltip=(
                "'full' = all face landmarks (sapiens-238 if present, else "
                "rig-fallback ~30). 'eyes_mouth' = rig-fallback subset (~12 "
                "dots: eyes + outer lips only). 'disabled' = no face dots."
            ),
        ),
        io.Combo.Input(
            "hand_style",
            options=["disabled", "dwpose", "openpose"],
            default="disabled",
            tooltip=(
                "Draw 21+21 hand keypoints + sticks. 'disabled' = no hands. "
                "'dwpose' = solid-blue dots; 'openpose' = rainbow dots."
            ),
        ),
        io.Float.Input(
            "person_palette_falloff", default=0.6, min=0.1, max=1.0, step=0.05,
            tooltip=(
                "Per-person desaturation: track k blends toward white by "
                "1 - falloff^k. Track 0 stays vivid; 1.0 disables falloff."
            ),
        ),
    ]


def _scale_pose_data(mhr_pose_data: Dict[str, Any], new_H: int, new_W: int) -> Dict[str, Any]:
    # 2D coords must match the body's letterbox transform (uniform scale +
    # center offset), else face/hand overlays drift off the body.
    old_H, old_W = mhr_pose_data["image_size"]
    if new_H == old_H and new_W == old_W:
        return mhr_pose_data
    rW = new_W / old_W
    rH = new_H / old_H
    r_focal = min(rW, rH)
    offset_x = (new_W - r_focal * old_W) * 0.5
    offset_y = (new_H - r_focal * old_H) * 0.5
    new_frames: List[List[Dict[str, Any]]] = []
    for frame in mhr_pose_data["frames"]:
        scaled = []
        for p in frame:
            p = dict(p)
            f = p.get("focal_length")
            if f is not None:
                p["focal_length"] = np.asarray(f, dtype=np.float32) * r_focal
            for k in ("pred_keypoints_2d", "pred_face_keypoints_2d"):
                v = p.get(k)
                if v is not None:
                    arr = np.asarray(v, dtype=np.float32).copy()
                    arr[..., 0] = arr[..., 0] * r_focal + offset_x
                    arr[..., 1] = arr[..., 1] * r_focal + offset_y
                    p[k] = arr
            bb = p.get("bbox")
            if bb is not None:
                bb = np.asarray(bb, dtype=np.float32).copy()
                bb[..., [0, 2]] = bb[..., [0, 2]] * r_focal + offset_x
                bb[..., [1, 3]] = bb[..., [1, 3]] * r_focal + offset_y
                p["bbox"] = bb
            scaled.append(p)
        new_frames.append(scaled)
    out = dict(mhr_pose_data)
    out["image_size"] = (new_H, new_W)
    out["frames"] = new_frames
    return out


class SAM3DBody_Render(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBody_Render",
            display_name="Render SAM3D Body",
            category="image/detection",
            inputs=[
                MHRPoseData.Input("mhr_pose_data"),
                io.Image.Input(
                    "background",
                    optional=True,
                    tooltip="Per-frame background. Omitted = black canvas.",
                ),
                io.Int.Input(
                    "width", default=0, min=0, max=16384, step=8,
                    tooltip=(
                        "Output width in pixels. 0 = use pose data's native "
                        "image_size. If only one of width/height is set, the "
                        "other is derived preserving the original aspect."
                    ),
                ),
                io.Int.Input(
                    "height", default=0, min=0, max=16384, step=8,
                    tooltip=(
                        "Output height in pixels. 0 = use pose data's native "
                        "image_size. If only one of width/height is set, the "
                        "other is derived preserving the original aspect."
                    ),
                ),
                io.Load3DCamera.Input(
                    "camera_info", optional=True,
                    tooltip=(
                        "Free 6DOF camera override. When wired, the pose is re-projected through this camera "
                        "(position/target/zoom) instead of the predicted one. "
                    ),
                ),
                io.Float.Input(
                    "fov", default=0.0, min=0.0, max=170.0, step=0.5, advanced=True,
                    tooltip=(
                        "Override the vertical FoV of the camera_info. Ignored when camera_info is empty. 0 = keep the FoV of the camera_info."
                    ),
                ),
                io.DynamicCombo.Input(
                    "render_style",
                    options=[
                        io.DynamicCombo.Option("mesh", _render_mesh_mode_inputs()),
                        io.DynamicCombo.Option("silhouette", []),
                        io.DynamicCombo.Option("openpose_2d", _render_openpose_mode_inputs()),
                        io.DynamicCombo.Option("openpose_3d", _render_openpose3d_mode_inputs()),
                        io.DynamicCombo.Option("scail", _render_capsules_mode_inputs()),
                    ],
                    tooltip=(
                        "'mesh' = 3D MHR mesh rasterized through the camera. "
                        "'silhouette' = binary mask of the mesh. "
                        "'openpose_2d' = flat 2D skeleton "
                        "'openpose_3d' = openpose skeleton as flat-shaded 3D model "
                        "'scail' = SCAIL 3D capsules "
                    ),
                ),
            ],
            outputs=[io.Image.Output("image")],
        )


    @classmethod
    def execute(cls, mhr_pose_data, background=None, width=0, height=0, camera_info=None, fov=0.0, render_style=None) -> io.NodeOutput:
        render_style = render_style or {"render_style": "mesh"}
        mode_key = render_style.get("render_style", "mesh")

        native_H, native_W = mhr_pose_data["image_size"]
        new_W, new_H = int(width), int(height)
        if new_W == 0 and new_H == 0:
            H, W = native_H, native_W
            px_scale = 1.0
        else:
            if new_W == 0:
                new_W = max(1, round(native_W * new_H / native_H))
            elif new_H == 0:
                new_H = max(1, round(native_H * new_W / native_W))
            mhr_pose_data = _scale_pose_data(mhr_pose_data, new_H, new_W)
            H, W = new_H, new_W
            px_scale = min(new_W / native_W, new_H / native_H)

        if camera_info is not None:
            mhr_pose_data = apply_camera_override(mhr_pose_data, camera_info, H, W, fov_deg=float(fov))

        B = len(mhr_pose_data["frames"])
        if B == 0:
            return io.NodeOutput(torch.zeros(1, H, W, 3, dtype=torch.float32))

        out_device = comfy.model_management.intermediate_device()
        bg_t = None if background is None else background.to(device=out_device, dtype=torch.float32)

        if bg_t is not None and tuple(bg_t.shape[1:3]) != (H, W): # Match the background to the render resolution
            bg_t = comfy.utils.common_upscale(bg_t.movedim(-1, 1), W, H, "bilinear", "disabled").movedim(1, -1)

        if mode_key == "silhouette":
            composite = "silhouette"
        elif bg_t is not None:
            composite = "over"
        else:
            composite = "mesh_only"

        if mode_key == "openpose_2d":
            marker_radius_px = max(1, int(round(render_style.get("marker_radius_px", 4) * px_scale)))
            stick_width_px = max(1, int(round(render_style.get("stick_width_px", 4) * px_scale)))
            limb_alpha = float(render_style.get("limb_alpha", 0.6))
            face_style = str(render_style.get("face_style", "disabled"))
            hand_style = str(render_style.get("hand_style", "disabled"))
            include_hands = hand_style != "disabled"
            hand_color_style = hand_style if include_hands else "dwpose"
            person_palette_falloff = float(render_style.get("person_palette_falloff", 0.6))
        elif mode_key == "openpose_3d":
            op3d_radius_m = float(render_style.get("radius_m", 0.015))
            op3d_include_hands = bool(render_style.get("include_hands", True))
            person_palette_falloff = float(render_style.get("person_palette_falloff", 0.6))
        elif mode_key == "scail":
            cap_radius_m = float(render_style.get("radius_m", 0.030))
            cap_hand_style = str(render_style.get("hand_style", "disabled"))
            cap_include_hands = cap_hand_style != "disabled"
            cap_hand_color_style = cap_hand_style if cap_include_hands else "dwpose"
            cap_face_style = str(render_style.get("face_style", "disabled"))
            person_palette_falloff = float(render_style.get("person_palette_falloff", 0.6))
        elif mode_key == "mesh":
            shader_dict = render_style.get("shader") or {}
            shader_key = shader_dict.get("shader", "default")
            rainbow_tilt_x = float(shader_dict.get("rainbow_tilt_x", 0.0))
            rainbow_tilt_z = float(shader_dict.get("rainbow_tilt_z", -35.0))
            opacity = float(render_style.get("opacity", 1.0))
            person_palette_falloff = float(render_style.get("person_palette_falloff", 0.6))
            region = str(render_style.get("region", "full_body"))

            if region == "hands_only":
                hand_mask = mhr_pose_data["hand_vert_mask"]
                faces_full = np.asarray(mhr_pose_data["faces"])
                keep = hand_mask[faces_full].all(axis=1)
                mhr_pose_data = dict(mhr_pose_data)
                mhr_pose_data["faces"] = np.ascontiguousarray(
                    faces_full[keep], dtype=faces_full.dtype,
                )
        else:  # silhouette — no shader/opacity controls, mask is binary
            shader_key = "default"
            rainbow_tilt_x = 0.0
            rainbow_tilt_z = -35.0
            opacity = 1.0
            person_palette_falloff = 0.6

        frames_out = []
        pbar = comfy.utils.ProgressBar(B)
        desc = (
            "SAM3D openpose-2D render" if mode_key == "openpose_2d"
            else "SAM3D openpose-3D render" if mode_key == "openpose_3d"
            else "SAM3D SCAIL-3D render" if mode_key == "scail"
            else "SAM3D silhouette" if mode_key == "silhouette"
            else "SAM3D render"
        )
        for f in tqdm(range(B), desc=desc):
            bg_f = None
            if bg_t is not None:
                bg_f = bg_t[min(f, bg_t.shape[0] - 1)]
            if mode_key == "openpose_2d":
                img = render_pose_data_openpose(
                    mhr_pose_data, frame_idx=f, W=W, H=H,
                    background=bg_f,
                    composite=composite,
                    marker_radius_px=marker_radius_px,
                    stick_width_px=stick_width_px,
                    limb_alpha=limb_alpha,
                    include_hands=include_hands,
                    face_style=face_style,
                    hand_color_style=hand_color_style,
                    person_brightness_falloff=person_palette_falloff,
                )
            elif mode_key == "openpose_3d":
                img = render_pose_data_capsules(
                    mhr_pose_data, frame_idx=f, W=W, H=H,
                    background=bg_f,
                    composite=composite,
                    radius_m=op3d_radius_m,
                    include_hands=op3d_include_hands,
                    palette="openpose",
                    flat_shade=True,
                    person_brightness_falloff=person_palette_falloff,
                )
            elif mode_key == "scail":
                # SCAIL renders body as 3D capsules + 2D openpose hands on top
                img = render_pose_data_capsules(
                    mhr_pose_data, frame_idx=f, W=W, H=H,
                    background=bg_f,
                    composite=composite,
                    radius_m=cap_radius_m,
                    include_hands=False,
                    palette="scail",
                    person_brightness_falloff=person_palette_falloff,
                )
                if cap_include_hands or cap_face_style != "disabled":
                    scail_overlay_px = max(1, int(round(4 * px_scale)))
                    scail_face_px = max(1, int(round(1 * px_scale)))
                    img = render_pose_data_openpose(
                        mhr_pose_data, frame_idx=f, W=W, H=H,
                        background=img,
                        composite="over",
                        include_body=False,
                        include_hands=cap_include_hands,
                        face_style=cap_face_style,
                        marker_radius_px=scail_overlay_px,
                        stick_width_px=scail_overlay_px,
                        face_marker_radius_px=scail_face_px,
                        hand_color_style=cap_hand_color_style,
                        person_brightness_falloff=person_palette_falloff,
                    )
            else:
                img = render_pose_data(
                    mhr_pose_data, frame_idx=f, W=W, H=H,
                    background=bg_f, composite=composite, opacity=opacity,
                    shader_preset=shader_key,
                    rainbow_tilt_x_deg=rainbow_tilt_x,
                    rainbow_tilt_z_deg=rainbow_tilt_z,
                    person_brightness_falloff=person_palette_falloff,
                )
            frames_out.append(img)
            pbar.update(1)

        out_image = torch.stack(frames_out, dim=0)
        if out_image.device != out_device:
            out_image = out_image.to(out_device)
        return io.NodeOutput(out_image)


class SAM3DBodyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        return [
            SAM3DBody_Loader,
            SAM3DBody_Predict,
            SAM3DBody_FaceExpression,
            SAM3DBody_Smooth,
            SAM3DBody_Render,
        ]


async def comfy_entrypoint() -> SAM3DBodyExtension:
    return SAM3DBodyExtension()
