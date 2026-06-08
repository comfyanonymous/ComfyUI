"""SCAIL-2 preprocessing nodes that turn SAM3 video tracks into the conditioning
bundle the SCAIL-2 model consumes."""

from typing_extensions import override

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io


SAM3TrackData = io.Custom("SAM3_TRACK_DATA")


# Model was trained on these exact colors; deviating degrades multi-identity quality.
DEFAULT_PALETTE = [
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (1.0, 1.0, 0.0),  # Yellow
]


def _unpack(track_data):
    from comfy.ldm.sam3.tracker import unpack_masks
    packed = track_data["packed_masks"]
    if packed is None or packed.shape[1] == 0:
        return None
    return unpack_masks(packed)


def _first_frame_cx_area(masks_bool):
    first = masks_bool[0].float()
    H, W = first.shape[-2], first.shape[-1]
    n_pixels = H * W
    grid_x = torch.arange(W, device=first.device, dtype=first.dtype).view(1, W)
    area = first.sum(dim=(-1, -2)).clamp_(min=1)
    cx = (first * grid_x).sum(dim=(-1, -2)) / area
    return (cx / W).tolist(), (area / n_pixels).tolist()


def _sort_tracks(track_data, sort_by):
    masks_bool = _unpack(track_data)
    if masks_bool is None:
        return []
    cx, area = _first_frame_cx_area(masks_bool)
    if sort_by == "x":
        return sorted(range(len(cx)), key=lambda i: cx[i])
    return sorted(range(len(area)), key=lambda i: -area[i])  # "area"


def _subset_track_data(track_data, obj_indices):
    out = dict(track_data)
    packed = track_data["packed_masks"]
    if packed is None or not obj_indices:
        out["packed_masks"] = None
        if "scores" in out:
            out["scores"] = []
        return out
    out["packed_masks"] = packed[:, obj_indices].contiguous()
    scores = track_data.get("scores")
    if scores is not None:
        out["scores"] = [scores[i] for i in obj_indices if i < len(scores)]
    return out


def _bg_to_rgb(background):
    if background.startswith("white"):
        return (1.0, 1.0, 1.0)
    return (0.0, 0.0, 0.0)


def _render_colored_masks(track_data, background="black"):
    from comfy.ldm.sam3.tracker import unpack_masks
    packed = track_data["packed_masks"]
    H, W = track_data["orig_size"]
    device = comfy.model_management.intermediate_device()
    bg_rgb = _bg_to_rgb(background)
    if packed is None or packed.shape[1] == 0:
        T = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        out = torch.empty(T, H, W, 3, device=device)
        out[..., 0], out[..., 1], out[..., 2] = bg_rgb[0], bg_rgb[1], bg_rgb[2]
        return out
    T, N_obj = packed.shape[0], packed.shape[1]
    colors = torch.tensor(
        [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(N_obj)],
        device=device, dtype=torch.float32,
    )
    masks_full = unpack_masks(packed.to(device)).float()
    Hm, Wm = masks_full.shape[-2], masks_full.shape[-1]
    masks_full = F.interpolate(
        masks_full.view(T * N_obj, 1, Hm, Wm), size=(H, W), mode="nearest"
    ).view(T, N_obj, H, W) > 0.5
    any_mask = masks_full.any(dim=1)
    obj_idx_map = masks_full.to(torch.uint8).argmax(dim=1)
    color_overlay = colors[obj_idx_map]
    bg_tensor = torch.tensor(bg_rgb, device=device, dtype=color_overlay.dtype).view(1, 1, 1, 3)
    return torch.where(any_mask.unsqueeze(-1), color_overlay, bg_tensor.expand_as(color_overlay))


class SCAIL2ColoredMask(io.ComfyNode):
    """Render SAM3 tracks for the driving video and (optionally) the reference
    image into the two colored masks WanSCAILToVideo consumes. Shared `sort_by`
    across both outputs guarantees identity K maps to the same color on both
    sides, so multi-person workflows stay consistent without a separate
    alignment node. ref_mask is always rendered black-bg (model convention);
    mask_video bg follows the mode you'll set on WanSCAILToVideo."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2ColoredMask",
            display_name="SCAIL-2 Colored Mask",
            category="conditioning/video_models/scail",
            inputs=[
                SAM3TrackData.Input("driving_track_data"),
                SAM3TrackData.Input("ref_track_data", optional=True,
                                    tooltip="SAM3 track of the reference image. Optional — wire it for the ref_mask_image output."),
                io.String.Input("object_indices", default="",
                                tooltip="Comma-separated object indices to include (e.g. '0,2,3'). Applied to both sides. Empty = all."),
                io.Combo.Input("sort_by", options=["none", "x", "area"],
                               tooltip="Applied to both sides identically so index K = same logical slot. x = left-to-right by first-frame centroid; area = descending mask area; none = SAM3's order."),
                io.Boolean.Input("replacement_mode", default=False,
                                 tooltip="False = mask_video has black bg (Animation Mode). True = white bg (Replacement Mode). WanSCAILToVideo auto-detects mode from the wired mask_video's bg color, so this is the single source of truth. ref_mask_image is always black-bg regardless."),
            ],
            outputs=[
                io.Image.Output("driving_mask_video"),
                io.Image.Output("ref_mask_image"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, driving_track_data, object_indices, sort_by, replacement_mode, ref_track_data=None):
        def _prep(td):
            if sort_by != "none":
                td = _subset_track_data(td, _sort_tracks(td, sort_by))
            if object_indices.strip():
                indices = [int(i.strip()) for i in object_indices.split(",") if i.strip().isdigit()]
                packed = td.get("packed_masks")
                n_obj = packed.shape[1] if packed is not None else 0
                indices = [i for i in indices if 0 <= i < n_obj]
                td = _subset_track_data(td, indices)
            return td

        drv = _prep(driving_track_data)
        mask_video = _render_colored_masks(drv, "white" if replacement_mode else "black")

        if ref_track_data is not None:
            ref = _prep(ref_track_data)
            ref_mask_image = _render_colored_masks(ref, "black")
        else:
            H, W = drv["orig_size"]
            ref_mask_image = torch.zeros(1, H, W, 3, device=comfy.model_management.intermediate_device())

        return io.NodeOutput(mask_video, ref_mask_image)


class SCAIL2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SCAIL2ColoredMask,
        ]


async def comfy_entrypoint() -> SCAIL2Extension:
    return SCAIL2Extension()
