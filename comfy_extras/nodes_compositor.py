import hashlib
import json
import math

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import ComfyExtension, io, UI
from comfy_extras.compositor_blend import (
    _LAYER_MODES,
    blend_composite,
    linear_to_srgb,
    placed_bounds,
    resolve_mode,
    srgb_to_linear,
)
from comfy_extras.color_util import hex_to_rgb
from nodes import MAX_RESOLUTION
from typing_extensions import override


MAX_LAYERS = 50


def document_items(doc) -> list[dict]:
    if not isinstance(doc, dict):
        return []
    items = [
        item
        for item in (doc.get("layers") or [])
        if isinstance(item, dict) and isinstance(item.get("image"), torch.Tensor)
    ]
    return sorted(items, key=lambda item: _int(item.get("z_index"), 0))


def document_canvas(doc) -> tuple[int, int] | None:
    if not isinstance(doc, dict):
        return None
    canvas = doc.get("canvas")
    if not isinstance(canvas, (tuple, list)) or len(canvas) != 2:
        return None
    w, h = _int(canvas[0], 0), _int(canvas[1], 0)
    return (w, h) if w > 0 and h > 0 else None


def _int(value, default: int) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _item_mask_frame(mask, index: int) -> torch.Tensor | None:
    if not isinstance(mask, torch.Tensor):
        return None
    if mask.shape[0] == 1:
        return mask[:1]
    if index < mask.shape[0]:
        return mask[index : index + 1]
    return None


def expand_item_frames(items: list[dict]) -> list[dict]:
    frames = []
    for item in items:
        image = item["image"]
        for index in range(image.shape[0]):
            frames.append({
                "tensor": image[index : index + 1],
                "mask": _item_mask_frame(item.get("mask"), index),
                "name": item.get("name") if isinstance(item.get("name"), str) else None,
                "x": _int(item.get("x"), 0),
                "y": _int(item.get("y"), 0),
                "opacity": item.get("opacity", 1.0),
                "blend": item.get("blend_mode", "normal"),
                "visible": item.get("visible", True),
                "flip_h": bool(item.get("flip_h", False)),
                "flip_v": bool(item.get("flip_v", False)),
            })
    if len(frames) > MAX_LAYERS:
        raise ValueError(
            f"Compositor supports at most {MAX_LAYERS} layers, got {len(frames)}"
        )
    return frames


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


def layer_preview_tensor(
    tensor: torch.Tensor, alpha: torch.Tensor | None
) -> torch.Tensor:
    rgb = tensor[:1, :, :, :3]
    if alpha is None:
        return rgb
    return torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1)


def canvas_extent(frames: list[dict]) -> tuple[int, int]:
    return (
        max(frame["x"] + frame["tensor"].shape[2] for frame in frames),
        max(frame["y"] + frame["tensor"].shape[1] for frame in frames),
    )


def input_fingerprints(
    frames: list[dict], alphas: list[torch.Tensor | None]
) -> list[str]:
    fingerprints = []
    for frame, alpha in zip(frames, alphas):
        tensor = frame["tensor"]
        rgb = tensor[0, :, :, :3].detach().cpu().numpy()
        rgb8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        digest = hashlib.sha256()
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(rgb8.tobytes())
        if alpha is not None:
            alpha8 = np.clip(
                np.rint(alpha[0].detach().cpu().numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            digest.update(alpha8.tobytes())
        digest.update(
            repr((
                frame["x"],
                frame["y"],
                frame["opacity"],
                frame["blend"],
                bool(frame["visible"]),
                frame["flip_h"],
                frame["flip_v"],
            )).encode()
        )
        fingerprints.append(digest.hexdigest()[:16])
    return fingerprints


def state_from_items(frames: list[dict], canvas: tuple[int, int]) -> dict:
    layers = []
    for frame in frames:
        layers.append({
            "name": frame["name"],
            "visible": bool(frame["visible"]),
            "opacity": frame["opacity"],
            "blend": frame["blend"],
            "flipH": frame["flip_h"],
            "flipV": frame["flip_v"],
            "transform": {
                "x": frame["x"],
                "y": frame["y"],
                "w": frame["tensor"].shape[2],
                "h": frame["tensor"].shape[1],
                "rotation": 0,
            },
        })
    return {
        "canvas": canvas,
        "layers": layers,
        "inputs": None,
        "background": {"color": "#ffffff", "opacity": 1.0, "visible": False},
    }


def layer_ui_entries(frames: list[dict]) -> list:
    entries = []
    for frame in frames:
        entries.append({
            "x": frame["x"],
            "y": frame["y"],
            "width": int(frame["tensor"].shape[2]),
            "height": int(frame["tensor"].shape[1]),
            "name": frame["name"],
            "visible": bool(frame["visible"]),
            "opacity": frame["opacity"] if isinstance(frame["opacity"], (int, float)) else 1.0,
            "blend": frame["blend"] if isinstance(frame["blend"], str) else "normal",
            "flipH": frame["flip_h"],
            "flipV": frame["flip_v"],
        })
    return entries


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
    version = state.get("version")
    if version is not None and version != 1:
        return None
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
                io.Layers.Input(
                    "layers",
                    tooltip="Layer stack to composite; build it with Add Layer. Items are stacked by z_index, batch frames inside an item expand to consecutive layers, and item placement, opacity, and blend mode define the initial composition. Without an explicit document canvas the size is a best-effort maximum extent of the placed layers. A saved composition that matches the current inputs takes priority.",
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
    def execute(cls, layers: io.Layers.Type, compositor: io.Compositor.Type = None) -> io.NodeOutput:
        frames = expand_item_frames(document_items(layers))
        tensors = [frame["tensor"] for frame in frames]
        alphas = [frame_alpha(frame["tensor"], frame["mask"]) for frame in frames]

        layer_refs = []
        for tensor, alpha in zip(tensors, alphas):
            layer_refs.extend(
                UI.PreviewImage(layer_preview_tensor(tensor, alpha), cls=cls).values
            )

        fp = input_fingerprints(frames, alphas)
        raw_state = compositor
        state = parse_layer_state(raw_state)
        replay = bool(state is not None and tensors and state["inputs"] == fp)
        if replay:
            out = composite_from_state(tensors, state, alphas)
        elif tensors:
            canvas = document_canvas(layers) or canvas_extent(frames)
            out = composite_from_state(
                tensors, state_from_items(frames, canvas), alphas
            )
        else:
            out = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        state_stale = layer_state_provided(raw_state) and not replay
        out, mask = composite_outputs(out)

        ui_dict = UI.PreviewImage(out, cls=cls).as_dict()
        ui_dict["compositor_layers"] = layer_refs
        ui_dict["compositor_inputs"] = fp
        ui_dict["compositor_bboxes"] = layer_ui_entries(frames)
        if state_stale:
            ui_dict["compositor_state_stale"] = [True]
        return io.NodeOutput(out, mask, ui=ui_dict)


class AddLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AddLayer",
            display_name="Add Layer",
            category="image",
            is_experimental=True,
            inputs=[
                io.Layers.Input(
                    "layers",
                    optional=True,
                    tooltip="Layer stack to append to. Leave unconnected to start a new stack.",
                ),
                io.Image.Input(
                    "image",
                    tooltip="Layer content at its native size. A batch expands to consecutive layers.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Transparency mask for this layer. Masked areas (value 1) become transparent, multiplying with any alpha channel the image already carries.",
                ),
                io.String.Input(
                    "name",
                    optional=True,
                    default="",
                    tooltip="Layer name shown in the compositor editor.",
                ),
                io.Int.Input(
                    "x",
                    optional=True,
                    default=0,
                    min=-MAX_RESOLUTION,
                    max=MAX_RESOLUTION,
                    tooltip="Initial horizontal placement on the canvas.",
                ),
                io.Int.Input(
                    "y",
                    optional=True,
                    default=0,
                    min=-MAX_RESOLUTION,
                    max=MAX_RESOLUTION,
                    tooltip="Initial vertical placement on the canvas.",
                ),
                io.Float.Input(
                    "opacity",
                    optional=True,
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Initial layer opacity.",
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=list(_LAYER_MODES),
                    default="normal",
                    optional=True,
                    tooltip="Initial blend mode.",
                ),
                io.Int.Input(
                    "z_index",
                    optional=True,
                    default=0,
                    min=-1000,
                    max=1000,
                    tooltip="Stacking override. Layers are stable-sorted by z_index; equal values keep their list order.",
                ),
                io.Boolean.Input(
                    "flip_h",
                    optional=True,
                    default=False,
                    tooltip="Flip the layer horizontally.",
                ),
                io.Boolean.Input(
                    "flip_v",
                    optional=True,
                    default=False,
                    tooltip="Flip the layer vertically.",
                ),
            ],
            outputs=[
                io.Layers.Output(tooltip="The layer stack with this layer appended."),
            ],
        )

    @classmethod
    def execute(cls, image: io.Image.Type, layers: io.Layers.Type = None, mask: io.Mask.Type = None, name: str = "", x: int = 0, y: int = 0, opacity: float = 1.0, blend_mode: str = "normal", z_index: int = 0, flip_h: bool = False, flip_v: bool = False) -> io.NodeOutput:
        item: dict = {
            "image": image,
            "type": "raster",
            "x": int(x),
            "y": int(y),
            "z_index": int(z_index),
        }
        if mask is not None:
            item["mask"] = mask
        if name:
            item["name"] = name
        if opacity != 1.0:
            item["opacity"] = float(opacity)
        if blend_mode != "normal":
            item["blend_mode"] = blend_mode
        if flip_h:
            item["flip_h"] = True
        if flip_v:
            item["flip_v"] = True
        previous = layers if isinstance(layers, dict) else None
        document: dict = {
            "version": 1,
            "layers": [*(previous.get("layers") or []), item] if previous else [item],
        }
        previous_canvas = document_canvas(previous)
        if previous_canvas:
            document["canvas"] = previous_canvas
        return io.NodeOutput(document)


class CompositorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageCompositor, AddLayer]


async def comfy_entrypoint() -> CompositorExtension:
    return CompositorExtension()
