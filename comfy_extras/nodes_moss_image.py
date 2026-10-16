import json
import math
import re

from typing_extensions import override
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


class MOSSImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MOSSImageLayoutPrompt,
        ]


async def comfy_entrypoint() -> MOSSImageExtension:
    return MOSSImageExtension()
