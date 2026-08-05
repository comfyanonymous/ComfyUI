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
from typing_extensions import override


def sort_autogrow_images(images: dict) -> list[torch.Tensor]:
    images = images or {}
    tensors = []
    for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
        image = images[name]
        if image is None:
            continue
        tensors.append(image)
    return tensors


def expand_batch_frames(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    frames = []
    for tensor in tensors:
        for index in range(tensor.shape[0]):
            frames.append(tensor[index : index + 1])
    return frames


def stack_images(tensors: list[torch.Tensor]) -> torch.Tensor:
    canvas = tensors[0][:1, :, :, :3].clone()
    h, w = canvas.shape[1], canvas.shape[2]
    for layer in tensors[1:]:
        layer = layer[:1, :, :, :3]
        lh = min(layer.shape[1], h)
        lw = min(layer.shape[2], w)
        canvas[:, :lh, :lw, :] = layer[:, :lh, :lw, :]
    return canvas


def input_fingerprints(tensors: list[torch.Tensor]) -> list[str]:
    fingerprints = []
    for tensor in tensors:
        frame = tensor[0].detach().cpu().numpy()
        frame8 = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
        digest = hashlib.sha256()
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(frame8.tobytes())
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
        "canvas": (tensors[0].shape[2], tensors[0].shape[1]),
        "layers": layers,
        "inputs": None,
        "background": {"color": "#ffffff", "opacity": 1.0, "visible": True},
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
        "opacity": _number(entry, "opacity", 1.0),
        "blend": blend if isinstance(blend, str) else "normal",
        "x": _number(transform, "x", 0.0),
        "y": _number(transform, "y", 0.0),
        "w": _number(transform, "w", natural_w),
        "h": _number(transform, "h", natural_h),
        "rotation": _number(transform, "rotation", 0.0),
        "flip_h": bool(entry.get("flipH", False)),
        "flip_v": bool(entry.get("flipV", False)),
    }


def _prepare_layer_bitmap(tensor: torch.Tensor, params: dict) -> Image.Image:
    frame = tensor[0, :, :, :3].detach().cpu().numpy()
    rgb8 = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb8, "RGB").convert("RGBA")
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


def composite_from_state(tensors: list[torch.Tensor], state: dict) -> torch.Tensor:
    cw, ch = state["canvas"]
    canvas = np.zeros((ch, cw, 4), dtype=np.float32)
    background = state.get("background")
    if background is not None and background["visible"] and background["opacity"] > 0:
        canvas = _fill_background(canvas, background)
    layers = state["layers"]
    for index, tensor in enumerate(tensors):
        entry = layers[index] if index < len(layers) else None
        params = _layer_params(entry, tensor.shape[2], tensor.shape[1])
        if not params["visible"]:
            continue
        img = _prepare_layer_bitmap(tensor, params)
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
    rgb = rgb * np.clip(canvas[..., 3:4], 0.0, 1.0)
    return torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)


class ImageCompositor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageCompositor",
            display_name="Image Compositor",
            category="image",
            is_output_node=True,
            has_intermediate_output=True,
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("image"),
                        prefix="image_",
                        min=1,
                        max=50,
                    ),
                    tooltip="Layers to composite. The first input is the bottom layer; each subsequent input is stacked above the previous one.",
                ),
                io.MultiType.Input(
                    "bboxes",
                    [io.BoundingBox, io.Array, io.String],
                    optional=True,
                    tooltip="Optional initial layout: bounding boxes, elements, or a JSON string, index-aligned with the image inputs (bboxes[0] places image_0). Inputs without a box keep their natural size at the origin. A saved compositor recipe that matches the current inputs takes priority.",
                ),
                io.Compositor.Input(
                    "compositor",
                    tooltip="Layer recipe saved by the compositor editor, replayed over the current inputs",
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type = None, compositor: io.Compositor.Type = None, bboxes: io.MultiType.Type = None) -> io.NodeOutput:
        tensors = expand_batch_frames(sort_autogrow_images(images))

        layer_refs = []
        for tensor in tensors:
            layer_refs.extend(UI.PreviewImage(tensor, cls=cls).values)

        fp = input_fingerprints(tensors)
        raw_state = compositor
        state = parse_layer_state(raw_state)
        replay = bool(state is not None and tensors and state["inputs"] == fp)
        slots = (
            layout_bboxes(bboxes, tensors[0].shape[2], tensors[0].shape[1])
            if tensors
            else []
        )
        if replay:
            out = composite_from_state(tensors, state)
        elif tensors and any(slot is not None for slot in slots):
            out = composite_from_state(tensors, state_from_bboxes(tensors, slots))
        elif tensors:
            out = stack_images(tensors)
        else:
            out = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        state_stale = layer_state_provided(raw_state) and not replay

        ui_dict = UI.PreviewImage(out, cls=cls).as_dict()
        ui_dict["compositor_layers"] = layer_refs
        ui_dict["compositor_inputs"] = fp
        ui_dict["compositor_bboxes"] = bbox_ui_entries(slots, len(tensors))
        if state_stale:
            ui_dict["compositor_state_stale"] = [True]
        return io.NodeOutput(out, ui=ui_dict)


class CompositorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageCompositor]


async def comfy_entrypoint() -> CompositorExtension:
    return CompositorExtension()
