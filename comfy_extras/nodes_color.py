from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class ColorToRGBInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ColorToRGBInt",
            display_name="Color Picker",
            category="utilities",
            description="Return a color RGB integer value and hexadecimal representation.",
            inputs=[
                io.Color.Input("color"),
            ],
            outputs=[
                io.Int.Output(display_name="rgb_int"),
                io.Color.Output(display_name="hex")
            ],
        )

    @classmethod
    def execute(cls, color: str) -> io.NodeOutput:
        # expect format #RRGGBB
        if len(color) != 7 or color[0] != "#":
            raise ValueError("Color must be in format #RRGGBB")
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        rgb_int = r * 256 * 256 + g * 256 + b
        return io.NodeOutput(rgb_int, color)


class ColorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColorToRGBInt]


async def comfy_entrypoint() -> ColorExtension:
    return ColorExtension()
