import numpy as np
import torch
from PIL import Image as PILImage, ImageColor, ImageDraw, ImageFont
from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO


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
                IO.Float.Input("font_size_percent", default=5.0, min=0.5, max=50.0, step=0.5, tooltip="Font size as a percentage of the image height.", advanced=True),
                #IO.Combo.Input("text_color", options=["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"], default="white"),
                #IO.Combo.Input("outline_color", options=["auto", "none", "black", "white", "red", "green", "blue", "yellow"], default="auto"),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, text, font_size_percent=5.0, text_color="white", outline_color="auto", margin_percent=1.0,
                line_spacing=1.2, background_opacity=0.0, min_font_percent=2.0, outline_thickness_factor=0.04) -> IO.NodeOutput:
        if text.strip() == "":
            return IO.NodeOutput(image)

        text = text.replace("\\n", "\n").replace("\\t", "\t")

        try:
            text_color = ImageColor.getrgb(text_color)[:3]
        except ValueError:
            text_color = (255, 255, 255)

        luminance = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
        contrast_color = (0, 0, 0) if luminance > 40 else (255, 255, 255)
        choice = outline_color.lower()
        if choice == "none":
            outline_color = None
        elif choice == "auto":
            outline_color = contrast_color
        else:
            outline_color = ImageColor.getrgb(outline_color)[:3]
        background_color = contrast_color if outline_color is None else outline_color

        frames = [cls.render_text_on_frame(frame, text, font_size_percent, margin_percent, text_color, outline_color, background_color,
                                           line_spacing, background_opacity, min_font_percent, outline_thickness_factor)
                  for frame in image]
        return IO.NodeOutput(torch.stack(frames, dim=0))

    @classmethod
    def render_text_on_frame(cls, frame, text, font_size_percent, margin_percent, text_color, outline_color, background_color,
                             line_spacing=1.2, background_opacity=0.0, min_font_percent=2.0, outline_thickness_factor=0.04, min_font_px=10):
        pil_image = PILImage.fromarray((frame.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8), mode="RGB")
        width, height = pil_image.width, pil_image.height

        margin = int(round(margin_percent / 100.0 * min(width, height)))
        max_width = max(1, width - 2 * margin)
        max_height = max(1, height - 2 * margin)

        # Font scales with resolution, then shrinks to fit the height.
        size = max(1, int(round(font_size_percent / 100.0 * height)))
        floor = min(size, max(min_font_px, int(round(min_font_percent / 100.0 * height))))
        while True:
            font = ImageFont.load_default(size=size)
            lines = cls.wrap_text(text, font, max_width)
            line_height = size * line_spacing
            if line_height * len(lines) <= max_height or size <= floor:
                break
            size = max(floor, int(size * 0.9))

        if background_opacity > 0:
            background_bottom = 2 * margin + line_height * len(lines)
            overlay = PILImage.new("RGBA", pil_image.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).rectangle([0, 0, width, background_bottom], fill=(*background_color, int(round(background_opacity * 255))))
            pil_image = PILImage.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(pil_image)
        stroke = max(1, int(round(size * outline_thickness_factor))) if outline_color is not None else 0
        for index, line in enumerate(lines):
            draw.text((margin, margin + index * line_height), line, font=font,
                      fill=text_color, stroke_width=stroke, stroke_fill=outline_color)

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
