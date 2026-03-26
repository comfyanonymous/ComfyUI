from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from comfy_api.input import CurveInput
from typing_extensions import override


class CurveEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CurveEditor",
            display_name="Curve Editor",
            category="utils",
            inputs=[
                io.Curve.Input("curve"),
                io.Histogram.Input("histogram", optional=True),
            ],
            outputs=[
                io.Curve.Output("curve"),
            ],
        )

    @classmethod
    def execute(cls, curve, histogram=None) -> io.NodeOutput:
        result = CurveInput.from_raw(curve)

        ui = {}
        if histogram is not None:
            ui["histogram"] = histogram if isinstance(histogram, list) else list(histogram)

        return io.NodeOutput(result, ui=ui) if ui else io.NodeOutput(result)


class CurveExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [CurveEditor]


async def comfy_entrypoint():
    return CurveExtension()
