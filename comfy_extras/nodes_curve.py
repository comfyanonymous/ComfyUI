from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from comfy_api.input import CurveInput, MonotoneCubicCurve, LinearCurve
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
            ],
            outputs=[
                io.Curve.Output("curve"),
            ],
        )

    @classmethod
    def execute(cls, curve) -> io.NodeOutput:
        if isinstance(curve, CurveInput):
            return io.NodeOutput(curve)
        raw_points = curve["points"] if isinstance(curve, dict) else curve
        points = [(float(x), float(y)) for x, y in raw_points]
        interpolation = curve.get("interpolation", "monotone_cubic") if isinstance(curve, dict) else "monotone_cubic"
        if interpolation == "linear":
            return io.NodeOutput(LinearCurve(points))
        return io.NodeOutput(MonotoneCubicCurve(points))


class CurveExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [CurveEditor]


async def comfy_entrypoint():
    return CurveExtension()
