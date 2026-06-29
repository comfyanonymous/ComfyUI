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
            description="Overlay text on an image or batch of images.",
            search_aliases=["text", "label", "caption", "subtitle", "watermark", "title", "addlabel", "overlay"],
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input("text", multiline=True, default=""),
                IO.Combo.Input("position", options=["top", "bottom"], default="top"),
                IO.Combo.Input("align", options=["left", "center", "right"], default="left"),
                IO.Float.Input("font_size_percent", default=5.0, min=0.5, max=50.0, step=0.5, tooltip="Font size as a percentage of the image height.", advanced=True),
                #IO.Combo.Input("text_color", options=["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"], default="white", tooltip="Color of the text.", advanced=True),
                #IO.Combo.Input("outline_color", options=["auto", "none", "black", "white", "red", "green", "blue", "yellow"], default="auto", tooltip="Color of the text outline.", advanced=True),
                #IO.Float.Input("background_opacity", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Opacity of the background behind the text (0 = transparent, 1 = solid).", advanced=True),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, text, position="top", align="left", font_size_percent=5.0, text_color="white", outline_color="auto",
                line_spacing=1.2, background_opacity=0.0) -> IO.NodeOutput:
        if text.strip() == "":
            return IO.NodeOutput(image)

        text = text.replace("\\n", "\n").replace("\\t", "\t")

        try:
            text_rgb = ImageColor.getrgb(text_color)[:3]
        except ValueError:
            text_rgb = (255, 255, 255)

        # "auto" outline color: set the outline color based on text color to maintain contrast
        luminance = 0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]
        contrast_rgb = (0, 0, 0) if luminance > 40 else (255, 255, 255)
        outline_choice = outline_color.lower()

        if outline_choice == "none":
            outline_rgb = None
        elif outline_choice == "auto":
            outline_rgb = contrast_rgb
        else:
            outline_rgb = ImageColor.getrgb(outline_color)[:3]

        background_rgb = contrast_rgb if outline_rgb is None else outline_rgb

        # Render the overlay once and composite it across all frames in the batch
        height = int(image.shape[1])
        width = int(image.shape[2])
        overlay_rgb, overlay_alpha = cls.render_overlay_text(width, height, text, position, align, font_size_percent,
                                                             text_rgb, outline_rgb, background_rgb, line_spacing, background_opacity)
        overlay_rgb = overlay_rgb.to(device=image.device, dtype=image.dtype)
        overlay_alpha = overlay_alpha.to(device=image.device, dtype=image.dtype)

        result = image * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
        return IO.NodeOutput(result)

    @classmethod
    def render_overlay_text(cls, width, height, text, position="top", align="left", font_size_percent=5.0,
                            text_rgb=(255, 255, 255), outline_rgb=None, background_rgb=None, line_spacing=1.2, background_opacity=0.0,
                            margin_percent=1.0, min_font_percent=2.0, min_font_pixels=10, outline_thickness_factor=0.04):
        # Draw onto a transparent layer so the result can be alpha-composited over any frame.
        layer = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        margin = int(round(margin_percent / 100.0 * min(width, height)))
        max_width = max(1, width - 2 * margin)
        max_height = max(1, height - 2 * margin)

        # Font scales with resolution, then shrinks to fit the height.
        size = max(1, int(round(font_size_percent / 100.0 * height)))
        floor = min(size, max(min_font_pixels, int(round(min_font_percent / 100.0 * height))))

        while True:
            font = ImageFont.load_default(size=size)
            stroke = max(1, int(round(size * outline_thickness_factor))) if outline_rgb is not None else 0
            block = "\n".join(cls.wrap_text(text, font, max_width))
            # convert line spacing to pixel spacing
            single = draw.textbbox((0, 0), "Ay", font=font, stroke_width=stroke)
            double = draw.multiline_textbbox((0, 0), "Ay\nAy", font=font, spacing=0, stroke_width=stroke)
            natural_advance = (double[3] - double[1]) - (single[3] - single[1])
            pixel_spacing = int(round(size * line_spacing - natural_advance))
            box = draw.multiline_textbbox((0, 0), block, font=font, spacing=pixel_spacing, stroke_width=stroke)
            block_height = box[3] - box[1]

            if block_height <= max_height or size <= floor:
                break

            size = max(floor, int(size * 0.9))

        if background_opacity > 0:
            band = block_height + 2 * margin
            rect = [0, height - band, width, height] if position == "bottom" else [0, 0, width, band]
            draw.rectangle(rect, fill=(*background_rgb, int(round(background_opacity * 255))))

        anchor_h, x = {"left": ("l", margin), "center": ("m", width / 2), "right": ("r", width - margin)}[align]

        if position == "bottom":
            anchor_v, y = "d", height - margin
        else:
            anchor_v, y = "a", margin

        draw.multiline_text((x, y), block, font=font, fill=text_rgb, anchor=anchor_h + anchor_v,
                            align=align, spacing=pixel_spacing, stroke_width=stroke, stroke_fill=outline_rgb)

        overlay = np.array(layer).astype(np.float32) / 255.0
        overlay_rgb = torch.from_numpy(overlay[:, :, :3])
        overlay_alpha = torch.from_numpy(overlay[:, :, 3:4])
        return overlay_rgb, overlay_alpha

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
