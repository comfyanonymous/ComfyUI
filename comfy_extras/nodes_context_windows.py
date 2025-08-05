from __future__ import annotations
from comfy_api.latest import ComfyExtension, io
import comfy.context_windows


class ContextWindowsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ContexWindowsTest",
            display_name="Context Windows Test",
            category="context",
            description="Test node for context windows",
            inputs=[
                io.Model.Input("model", tooltip="The model to apply context windows to during sampling."),
                io.Int.Input("context_length", min=1, default=1, tooltip="The length of the context window."),
                io.Int.Input("context_overlap", min=0, default=0, tooltip="The overlap of the context window."),
                io.Combo.Input("context_schedule", options=[
                    comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                    comfy.context_windows.ContextSchedules.BATCHED,
                    ], tooltip="The stride of the context window."),
                io.Combo.Input("fuse_method", options=comfy.context_windows.ContextFuseMethod.LIST_STATIC,default=comfy.context_windows.ContextFuseMethod.PYRAMID, tooltip="The method to use to fuse the context windows."),
                io.Int.Input("dim", min=0, max=1, default=0, tooltip="The dimension to apply the context windows to."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with context windows applied during sampling."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: io.Model.Type, context_length: int, context_overlap: int, context_schedule: str, fuse_method: str, dim: int) -> io.Model:
        model = model.clone()
        model.model_options["context_handler"] = comfy.context_windows.IndexListContextHandler(
            context_schedule=context_schedule,
            fuse_method=fuse_method,
            context_length=context_length,
            context_overlap=context_overlap,
            dim=dim)
        return io.NodeOutput(model)


class ContextWindowsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ContextWindowsNode,
        ]

def comfy_entrypoint():
    return ContextWindowsExtension()
