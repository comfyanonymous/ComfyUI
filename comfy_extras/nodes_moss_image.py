import json
import math
import re

import torch
from typing_extensions import override

import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io


class MOSSImageLayoutPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageLayoutPrompt",
            display_name="MOSS-Image Layout Prompt",
            category="model/conditioning/moss_image",
            description="Builds a MOSS-Image layout prompt from the prompt and the boxes from Create Bounding Boxes.",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, tooltip="Description of the whole image."),
                io.Array.Input("element", tooltip="Prompt elements from the node Create Bounding Boxes. The description is the box's caption, text boxes also render their text."),
                io.Int.Input("width", default=1024, min=16, max=16384, step=16, tooltip="Width of the generated image; the boxes are placed in its pixels."),
                io.Int.Input("height", default=1024, min=16, max=16384, step=16, tooltip="Height of the generated image; the boxes are placed in its pixels."),
                io.String.Input("character_boxes", default="", tooltip="Numbers of the object boxes to label as characters instead of objects, as tagged in the Create Bounding Boxes preview, e.g. \"1, 3\"."),
            ],
            outputs=[io.String.Output(display_name="prompt")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, prompt, element, width, height, character_boxes="") -> io.NodeOutput:
        characters = {int(i) for i in re.findall(r"\d+", character_boxes)}
        items = []
        for i, el in enumerate(element if isinstance(element, list) else []):
            label = "text" if el.get("type") == "text" else ("character" if i + 1 in characters else "object")
            item = {"label": label, "caption": el.get("desc", "")}
            if item["label"] == "text":
                item["text"] = el.get("text", "")
            if el.get("color_palette"):
                item["palette"] = el["color_palette"]
            # 0-1000 [y0, x0, y1, x1] -> pixel edges enclosing the box, origin at the canvas centre
            y0, x0, y1, x1 = el["bbox"]
            item["bbox"] = [math.floor(round(y0 * height / 1000, 9)) - height // 2, math.floor(round(x0 * width / 1000, 9)) - width // 2,
                            math.ceil(round(y1 * height / 1000, 9)) - height // 2, math.ceil(round(x1 * width / 1000, 9)) - width // 2]
            items.append(item)
        if not items:
            return io.NodeOutput(prompt)
        layout = {"summary": {"caption": prompt.strip()}, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
        return io.NodeOutput(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))


class MOSSImageControl(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageControl",
            display_name="MOSS-Image Color/Depth Control",
            category="model/conditioning/moss_image",
            description="Adds the color region and depth maps for a MOSS-Image color/depth control adapter loaded with a LoRA loader.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("color_map", optional=True, tooltip="One flat color per region, at the output size. The colors are only region IDs."),
                io.Image.Input("depth_map", optional=True, tooltip="Depth map at the output size, near is white."),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Scales the maps' contribution to the image tokens. 0.0 is the same as no maps, 1.0 is the trained strength."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, tooltip="Sampling percent at which the control starts."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, tooltip="Sampling percent at which the control ends. Ending early keeps the layout and depth while the rest of the steps loosen the shapes."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Conditioning.Output(display_name="negative")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, color_map=None, depth_map=None, strength=1.0, start_percent=0.0, end_percent=1.0) -> io.NodeOutput:
        latents = [vae.encode(m[:, :, :, :3]) if m is not None else None for m in (color_map, depth_map)]
        ref = next((latent for latent in latents if latent is not None), None)
        if ref is None:
            return io.NodeOutput(positive, negative)
        batch = max(latent.shape[0] for latent in latents if latent is not None)
        # [color | depth]; a missing map is a zero latent
        latents = [torch.zeros_like(ref) if latent is None else latent if latent.shape[-2:] == ref.shape[-2:]
                   else comfy.utils.common_upscale(latent, ref.shape[-1], ref.shape[-2], "bilinear", "center") for latent in latents]
        # the embedder is linear and zeros are no control, so scaling the latent scales the control's contribution
        control = torch.cat([comfy.utils.resize_to_batch_size(latent, batch) for latent in latents], dim=1) * strength
        values = {"concat_latent_image": control}
        return io.NodeOutput(node_helpers.conditioning_set_values_with_timestep_range(positive, values, start_percent, end_percent),
                             node_helpers.conditioning_set_values_with_timestep_range(negative, values, start_percent, end_percent))


class MOSSImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MOSSImageLayoutPrompt,
            MOSSImageControl,
        ]


async def comfy_entrypoint() -> MOSSImageExtension:
    return MOSSImageExtension()
