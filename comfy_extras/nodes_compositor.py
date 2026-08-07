import hashlib
import json
import math

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import ComfyExtension, io, UI
from comfy_extras.compositor_blend import (
    blend_composite,
    linear_to_srgb,
    placed_bounds,
    resolve_mode,
    srgb_to_linear,
)
from comfy_extras.color_util import hex_to_rgb
from comfy_extras.nodes_bounding_boxes import boxes_from_input
from nodes import MAX_RESOLUTION
from typing_extensions import override


def expand_batch_frames(tensor: torch.Tensor) -> list[torch.Tensor]:
    return [tensor[index : index + 1] for index in range(tensor.shape[0])]


def frame_alpha(
    tensor: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor | None:
    alpha = tensor[:1, :, :, 3] if tensor.shape[-1] == 4 else None
    if mask is None:
        return alpha
    h, w = tensor.shape[1], tensor.shape[2]
    m = mask[:1].to(dtype=torch.float32)
    if m.shape[1] != h or m.shape[2] != w:
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=(h, w), mode="bilinear"
        ).squeeze(1)
    inv = torch.clamp(1.0 - m, 0.0, 1.0)
    return inv if alpha is None else alpha * inv


def frame_alphas(
    tensors: list[torch.Tensor], masks: list[torch.Tensor]
) -> list[torch.Tensor | None]:
    return [
        frame_alpha(tensor, masks[index] if index < len(masks) else None)
        for index, tensor in enumerate(tensors)
    ]


def layer_preview_tensor(
    tensor: torch.Tensor, alpha: torch.Tensor | None
) -> torch.Tensor:
    rgb = tensor[:1, :, :, :3]
    if alpha is None:
        return rgb
    return torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1)


def canvas_size(tensors: list[torch.Tensor]) -> tuple[int, int]:
    return (
        max(tensor.shape[2] for tensor in tensors),
        max(tensor.shape[1] for tensor in tensors),
    )


def input_fingerprints(
    tensors: list[torch.Tensor], alphas: list[torch.Tensor | None]
) -> list[str]:
    fingerprints = []
    for tensor, alpha in zip(tensors, alphas):
        frame = tensor[0, :, :, :3].detach().cpu().numpy()
        frame8 = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
        digest = hashlib.sha256()
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(frame8.tobytes())
        if alpha is not None:
            alpha8 = np.clip(
                np.rint(alpha[0].detach().cpu().numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            digest.update(alpha8.tobytes())
        fingerprints.append(digest.hexdigest()[:16])
    return fingerprints


def _bbox_entries(bboxes) -> list:
    if bboxes is None:
        return []
    if isinstance(bboxes, str):
        text = bboxes.strip()
        if not text:
            return []
        try:
            bboxes = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"bboxes string input is not valid JSON: {exc}") from exc
    if isinstance(bboxes, dict):
        return [bboxes]
    if not isinstance(bboxes, list):
        raise ValueError(
            "bboxes input must be bounding boxes, elements, or a JSON string, "
            f"got {type(bboxes).__name__}"
        )
    if bboxes and isinstance(bboxes[0], list):
        return bboxes[0]
    return bboxes


def layout_bboxes(bboxes, width: int, height: int) -> list:
    slots = []
    for entry in _bbox_entries(bboxes):
        try:
            boxes = boxes_from_input(entry, width, height)
        except ValueError:
            boxes = []
        slots.append(boxes[0] if boxes else None)
    return slots


def bbox_layer_name(box: dict) -> str | None:
    meta = box.get("metadata")
    if not isinstance(meta, dict):
        return None
    for key in ("name", "desc"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _bbox_int(box: dict, key: str) -> int:
    value = box.get(key, 0)
    return int(round(value)) if isinstance(value, (int, float)) else 0


def bbox_ui_entries(slots: list, count: int) -> list:
    if not slots:
        return []
    entries = []
    for index in range(count):
        box = slots[index] if index < len(slots) else None
        if box is None:
            entries.append(None)
            continue
        entries.append({
            "x": _bbox_int(box, "x"),
            "y": _bbox_int(box, "y"),
            "width": _bbox_int(box, "width"),
            "height": _bbox_int(box, "height"),
            "name": bbox_layer_name(box),
        })
    return entries


def state_from_bboxes(tensors: list[torch.Tensor], slots: list) -> dict:
    layers = []
    for index in range(len(tensors)):
        box = slots[index] if index < len(slots) else None
        if box is None:
            layers.append(None)
        else:
            layers.append({
                "transform": {
                    "x": box.get("x", 0),
                    "y": box.get("y", 0),
                    "w": box.get("width", 0),
                    "h": box.get("height", 0),
                    "rotation": 0,
                }
            })
    return {
        "canvas": canvas_size(tensors),
        "layers": layers,
        "inputs": None,
        "background": {"color": "#ffffff", "opacity": 1.0, "visible": False},
    }


_HEX_DIGITS = set("0123456789abcdef")


def _normalize_hex_color(value) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("#"):
            digits = text[1:]
            if len(digits) == 3 and set(digits) <= _HEX_DIGITS:
                digits = "".join(ch * 2 for ch in digits)
            if len(digits) == 6 and set(digits) <= _HEX_DIGITS:
                return "#" + digits
    return "#ffffff"


def _parse_background(entry) -> dict | None:
    if not isinstance(entry, dict):
        return None
    return {
        "color": _normalize_hex_color(entry.get("color")),
        "opacity": min(max(_number(entry, "opacity", 1.0), 0.0), 1.0),
        "visible": bool(entry.get("visible", True)),
    }


def _parse_order(value, layer_count: int) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return None
    if sorted(value) != list(range(layer_count)):
        return None
    return value


def layer_state_provided(raw) -> bool:
    if isinstance(raw, dict):
        return bool(raw)
    if isinstance(raw, str):
        return raw not in ("", "{}")
    return False


def parse_layer_state(raw) -> dict | None:
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    state = raw
    canvas = state.get("canvas")
    layers = state.get("layers")
    if not isinstance(canvas, dict) or not isinstance(layers, list) or not layers:
        return None
    try:
        w = int(round(float(canvas.get("w"))))
        h = int(round(float(canvas.get("h"))))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    inputs = state.get("inputs")
    if not isinstance(inputs, list) or not all(
        isinstance(entry, str) for entry in inputs
    ):
        inputs = None
    return {
        "canvas": (w, h),
        "layers": layers,
        "inputs": inputs,
        "background": _parse_background(state.get("background")),
        "order": _parse_order(state.get("order"), len(layers)),
    }


def _number(source: dict, key: str, default: float) -> float:
    value = source.get(key, default)
    return float(value) if isinstance(value, (int, float)) else float(default)


def _layer_params(entry, natural_w: int, natural_h: int) -> dict:
    if not isinstance(entry, dict):
        entry = {}
    transform = entry.get("transform")
    if not isinstance(transform, dict):
        transform = {}
    blend = entry.get("blend")
    return {
        "visible": bool(entry.get("visible", True)),
        # The layer state is untrusted input: it round-trips through the saved
        # workflow and can be posted directly to /prompt. An out-of-range opacity
        # would otherwise reach blend_composite as a raw coverage multiplier and
        # produce negative or greater-than-white RGB. _parse_background already
        # clamps the same field.
        "opacity": min(max(_number(entry, "opacity", 1.0), 0.0), 1.0),
        "blend": blend if isinstance(blend, str) else "normal",
        "x": _number(transform, "x", 0.0),
        "y": _number(transform, "y", 0.0),
        "w": _number(transform, "w", natural_w),
        "h": _number(transform, "h", natural_h),
        "rotation": _number(transform, "rotation", 0.0),
        "flip_h": bool(entry.get("flipH", False)),
        "flip_v": bool(entry.get("flipV", False)),
    }


def _prepare_layer_bitmap(
    tensor: torch.Tensor, params: dict, alpha: torch.Tensor | None
) -> Image.Image:
    frame = tensor[0, :, :, :3].detach().cpu().numpy()
    rgb8 = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
    if alpha is None:
        img = Image.fromarray(rgb8, "RGB").convert("RGBA")
    else:
        alpha8 = np.clip(
            np.rint(alpha[0].detach().cpu().numpy() * 255.0), 0, 255
        ).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb8, alpha8]), "RGBA")
    if params["flip_h"]:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if params["flip_v"]:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    target = (max(1, round(params["w"])), max(1, round(params["h"])))
    if img.size != target:
        img = img.resize(target, Image.Resampling.LANCZOS)
    if params["rotation"] != 0:
        img = img.rotate(
            -math.degrees(params["rotation"]),
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )
    return img


def _place_in_bounds(img: Image.Image, bw: int, bh: int) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rgba = np.concatenate([srgb_to_linear(arr[..., :3]), arr[..., 3:4]], axis=-1)
    aw, ah = img.size
    buf = np.zeros((bh, bw, 4), dtype=np.float32)
    ox = (bw - aw) // 2
    oy = (bh - ah) // 2
    dx0, dy0 = max(ox, 0), max(oy, 0)
    dx1, dy1 = min(ox + aw, bw), min(oy + ah, bh)
    if dx0 < dx1 and dy0 < dy1:
        buf[dy0:dy1, dx0:dx1] = rgba[dy0 - oy : dy1 - oy, dx0 - ox : dx1 - ox]
    return buf


def _fill_background(canvas: np.ndarray, background: dict) -> np.ndarray:
    layer = np.empty(canvas.shape, dtype=np.float32)
    layer[..., :3] = srgb_to_linear(
        np.array(hex_to_rgb(background["color"]), dtype=np.float32) / 255.0
    )
    layer[..., 3] = 1.0
    return blend_composite(
        resolve_mode("normal"), canvas, layer, background["opacity"]
    )


def composite_from_state(
    tensors: list[torch.Tensor],
    state: dict,
    alphas: list[torch.Tensor | None],
) -> torch.Tensor:
    cw, ch = state["canvas"]
    if cw > MAX_RESOLUTION or ch > MAX_RESOLUTION:
        raise ValueError(
            f"Compositor canvas {cw}x{ch} exceeds the maximum supported size of "
            f"{MAX_RESOLUTION}x{MAX_RESOLUTION}"
        )
    canvas = np.zeros((ch, cw, 4), dtype=np.float32)
    background = state.get("background")
    if background is not None and background["visible"] and background["opacity"] > 0:
        canvas = _fill_background(canvas, background)
    layers = state["layers"]
    order = state.get("order") or range(len(tensors))
    for index in order:
        if index < 0 or index >= len(tensors):
            continue
        tensor = tensors[index]
        entry = layers[index] if index < len(layers) else None
        params = _layer_params(entry, tensor.shape[2], tensor.shape[1])
        if not params["visible"]:
            continue
        img = _prepare_layer_bitmap(
            tensor, params, alphas[index] if index < len(alphas) else None
        )
        bx, by, bw, bh = placed_bounds(
            params["x"], params["y"], params["w"], params["h"], params["rotation"]
        )
        buf = _place_in_bounds(img, bw, bh)
        x0, y0 = max(bx, 0), max(by, 0)
        x1, y1 = min(bx + bw, cw), min(by + bh, ch)
        if x0 >= x1 or y0 >= y1:
            continue
        region = buf[y0 - by : y1 - by, x0 - bx : x1 - bx]
        mode = resolve_mode(params["blend"])
        canvas[y0:y1, x0:x1] = blend_composite(
            mode, canvas[y0:y1, x0:x1], region, params["opacity"]
        )
    rgb = linear_to_srgb(np.clip(canvas[..., :3], 0.0, 1.0))
    alpha = np.clip(canvas[..., 3:4], 0.0, 1.0)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return torch.from_numpy(rgba.astype(np.float32)).unsqueeze(0)


OPAQUE_EPSILON = 1e-3


def composite_outputs(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if out.shape[-1] != 4:
        return out, torch.zeros(out.shape[:3], dtype=torch.float32)
    alpha = out[..., 3]
    if bool((alpha >= 1.0 - OPAQUE_EPSILON).all()):
        return out[..., :3], torch.zeros_like(alpha)
    return out, torch.clamp(1.0 - alpha, 0.0, 1.0)


class ImageCompositor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageCompositor",
            display_name="Create Layered Image",
            category="image",
            is_experimental=True,
            is_output_node=True,
            has_intermediate_output=True,
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Layers to composite. Each batch frame becomes a layer; the first frame is the back layer and each following frame is stacked above the previous one. Batch multiple images upstream to composite them.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional transparency masks, paired with image frames by batch index (the first mask applies to the first frame). Masked areas (value 1) become transparent, multiplying with any alpha channel the image already carries.",
                ),
                io.MultiType.Input(
                    "bboxes",
                    [io.BoundingBox, io.Array, io.String],
                    optional=True,
                    tooltip="Optional bounding boxes to initialize the layout, index-aligned with the image frames (bboxes[0] places the first frame). Frames without a bounding box keep their natural size at the origin. A saved composition that matches the current set of inputs takes priority.",
                ),
                io.Compositor.Input(
                    "compositor",
                    tooltip="Layered composition saved by the compositor editor.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="Composited image. Carries an alpha channel when the composite has transparent areas (e.g. hidden background), otherwise plain RGB."
                ),
                io.Mask.Output(
                    tooltip="Transparency of the composite (1 = fully transparent). All zeros when the composite is opaque."
                ),
            ],
        )

    @classmethod
    def execute(cls, image: io.Image.Type, mask: io.Mask.Type = None, compositor: io.Compositor.Type = None, bboxes: io.MultiType.Type = None) -> io.NodeOutput:
        tensors = expand_batch_frames(image)
        mask_frames = expand_batch_frames(mask) if mask is not None else []
        alphas = frame_alphas(tensors, mask_frames)

        layer_refs = []
        for tensor, alpha in zip(tensors, alphas):
            layer_refs.extend(
                UI.PreviewImage(layer_preview_tensor(tensor, alpha), cls=cls).values
            )

        fp = input_fingerprints(tensors, alphas)
        raw_state = compositor
        state = parse_layer_state(raw_state)
        replay = bool(state is not None and tensors and state["inputs"] == fp)
        slots = layout_bboxes(bboxes, *canvas_size(tensors)) if tensors else []
        if replay:
            out = composite_from_state(tensors, state, alphas)
        elif tensors:
            out = composite_from_state(
                tensors, state_from_bboxes(tensors, slots), alphas
            )
        else:
            out = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        state_stale = layer_state_provided(raw_state) and not replay
        out, mask = composite_outputs(out)

        ui_dict = UI.PreviewImage(out, cls=cls).as_dict()
        ui_dict["compositor_layers"] = layer_refs
        ui_dict["compositor_inputs"] = fp
        ui_dict["compositor_bboxes"] = bbox_ui_entries(slots, len(tensors))
        if state_stale:
            ui_dict["compositor_state_stale"] = [True]
        return io.NodeOutput(out, mask, ui=ui_dict)


class CompositorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageCompositor]


async def comfy_entrypoint() -> CompositorExtension:
    return CompositorExtension()
