import numpy as np
import torch
from PIL import Image as PILImage, ImageColor, ImageDraw, ImageFont
from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

LINE_SPACING = 1.2
BANNER_OPACITY = 0.45


class TextOverlay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TextOverlay",
            display_name="Text Overlay",
            category="image/text",
            description="Overlay text along the top of an image or batch of images.",
            search_aliases=["text", "label", "caption", "subtitle", "watermark", "title", "addlabel", "overlay"],
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input("text", multiline=True, default=""),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, text, font_size_percent=5.0, text_color="white", outline=True, background=False, background_color="auto", margin_percent=1.0) -> IO.NodeOutput:
        if text.strip() == "":
            return IO.NodeOutput(image)

        try:
            fill_color = ImageColor.getrgb(text_color)[:3]
        except ValueError:
            fill_color = (255, 255, 255)
        if background_color.lower() == "auto":
            luminance = 0.299 * fill_color[0] + 0.587 * fill_color[1] + 0.114 * fill_color[2]
            contrast_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
        else:
            contrast_color = ImageColor.getrgb(background_color)[:3]

        frames = [cls.render_text_on_frame(frame, text, font_size_percent, margin_percent, fill_color, contrast_color, outline, background)
                  for frame in image]
        return IO.NodeOutput(torch.stack(frames, dim=0))

    @classmethod
    def render_text_on_frame(cls, frame, text, font_size_percent, margin_percent, fill_color, contrast_color, outline, background):
        pil_image = PILImage.fromarray((frame.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8), mode="RGB")
        width, height = pil_image.width, pil_image.height

        margin = int(round(margin_percent / 100.0 * min(width, height)))
        max_width = max(1, width - 2 * margin)
        max_height = max(1, height - 2 * margin)

        # Font scales with resolution, then shrinks to fit the height.
        size = max(1, int(round(font_size_percent / 100.0 * height)))
        floor = min(size, max(10, int(round(0.02 * height))))
        while True:
            font = ImageFont.load_default(size=size)
            lines = cls.wrap_text(text, font, max_width)
            line_height = size * LINE_SPACING
            if line_height * len(lines) <= max_height or size <= floor:
                break
            size = max(floor, int(size * 0.9))

        if background:
            banner_bottom = 2 * margin + line_height * len(lines)
            overlay = PILImage.new("RGBA", pil_image.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).rectangle([0, 0, width, banner_bottom], fill=(*contrast_color, int(round(BANNER_OPACITY * 255))))
            pil_image = PILImage.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(pil_image)
        stroke = max(1, int(round(size / 24))) if outline else 0
        for index, line in enumerate(lines):
            draw.text((margin, margin + index * line_height), line, font=font,
                      fill=fill_color, stroke_width=stroke, stroke_fill=contrast_color)

        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)

    @staticmethod
    def wrap_text(text, font, max_width):
        lines = []
        for raw_line in text.split("\n"):
            words = raw_line.split()
            if not words:
                lines.append("")
                continue
            current = ""
            # Break the line into words and split words that are too long
            for word in words:
                while font.getlength(word) > max_width and len(word) > 1:
                    cut = 1
                    while cut < len(word) and font.getlength(word[:cut + 1]) <= max_width:
                        cut += 1
                    if current:
                        lines.append(current)
                        current = ""
                    lines.append(word[:cut])
                    word = word[cut:]
                candidate = word if not current else current + " " + word
                if not current or font.getlength(candidate) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
        return lines


class TextOverlayExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [TextOverlay]


async def comfy_entrypoint() -> TextOverlayExtension:
    return TextOverlayExtension()
