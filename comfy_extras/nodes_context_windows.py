from __future__ import annotations
from comfy_api.latest import ComfyExtension, io
import comfy.context_windows
import nodes


class ContextWindowsManualNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ContextWindowsManual",
            display_name="Context Windows (Manual)",
            category="context",
            description="Manually set context windows.",
            inputs=[
                io.Model.Input("model", tooltip="The model to apply context windows to during sampling."),
                io.Int.Input("context_length", min=1, default=16, tooltip="The length of the context window."),
                io.Int.Input("context_overlap", min=0, default=4, tooltip="The overlap of the context window."),
                io.Combo.Input("context_schedule", options=[
                    comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                    comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                    comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                    comfy.context_windows.ContextSchedules.BATCHED,
                    ], tooltip="The stride of the context window."),
                io.Int.Input("context_stride", min=1, default=1, tooltip="The stride of the context window; only applicable to uniform schedules."),
                io.Boolean.Input("closed_loop", default=False, tooltip="Whether to close the context window loop; only applicable to looped schedules."),
                io.Combo.Input("fuse_method", options=comfy.context_windows.ContextFuseMethods.LIST_STATIC, default=comfy.context_windows.ContextFuseMethods.PYRAMID, tooltip="The method to use to fuse the context windows."),
                io.Int.Input("dim", min=0, max=5, default=0, tooltip="The dimension to apply the context windows to."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with context windows applied during sampling."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: io.Model.Type, context_length: int, context_overlap: int, context_schedule: str, context_stride: int, closed_loop: bool, fuse_method: str, dim: int) -> io.Model:
        model = model.clone()
        model.model_options["context_handler"] = comfy.context_windows.IndexListContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=dim)
        # make memory usage calculation only take into account the context window latents
        comfy.context_windows.create_prepare_sampling_wrapper(model)
        return io.NodeOutput(model)

class WanContextWindowsManualNode(ContextWindowsManualNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "WanContextWindowsManual"
        schema.display_name = "WAN Context Windows (Manual)"
        schema.description = "Manually set context windows for WAN-like models (dim=2)."
        schema.inputs = [
            io.Model.Input("model", tooltip="The model to apply context windows to during sampling."),
                io.Int.Input("context_length", min=1, max=nodes.MAX_RESOLUTION, step=4, default=81, tooltip="The length of the context window."),
                io.Int.Input("context_overlap", min=0, default=30, tooltip="The overlap of the context window."),
                io.Combo.Input("context_schedule", options=[
                    comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                    comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                    comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                    comfy.context_windows.ContextSchedules.BATCHED,
                    ], tooltip="The stride of the context window."),
                io.Int.Input("context_stride", min=1, default=1, tooltip="The stride of the context window; only applicable to uniform schedules."),
                io.Boolean.Input("closed_loop", default=False, tooltip="Whether to close the context window loop; only applicable to looped schedules."),
                io.Combo.Input("fuse_method", options=comfy.context_windows.ContextFuseMethods.LIST_STATIC, default=comfy.context_windows.ContextFuseMethods.PYRAMID, tooltip="The method to use to fuse the context windows."),
        ]
        return schema

    @classmethod
    def execute(cls, model: io.Model.Type, context_length: int, context_overlap: int, context_schedule: str, context_stride: int, closed_loop: bool, fuse_method: str) -> io.Model:
        context_length = max(((context_length - 1) // 4) + 1, 1)  # at least length 1
        context_overlap = max(((context_overlap - 1) // 4) + 1, 0)  # at least overlap 0
        return super().execute(model, context_length, context_overlap, context_schedule, context_stride, closed_loop, fuse_method, dim=2)


class ContextWindowsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ContextWindowsManualNode,
            WanContextWindowsManualNode,
        ]

def comfy_entrypoint():
    return ContextWindowsExtension()
