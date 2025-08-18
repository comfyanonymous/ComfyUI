from comfy_api.latest import io, ComfyExtension

class MemoryReserveNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AddMemoryToReserve",
            display_name="Reserve Additional Memory",
            description="Adds additional expected memory usage for the model, in gigabytes.",
            category="advanced/debug/model",
            inputs=[
                io.Model.Input("model", tooltip="The model to add memory reserve to."),
                io.Float.Input("memory_reserve_gb", min=0.0, default=0.0, max=2048.0, step=0.1, tooltip="The additional expected memory usage for the model, in gigabytes."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with the additional memory reserve."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, memory_reserve_gb: float) -> io.NodeOutput:
        model = model.clone()
        model.add_model_memory_reserve(memory_reserve_gb)
        return io.NodeOutput(model)

class MemoryReserveExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MemoryReserveNode,
        ]

def comfy_entrypoint():
    return MemoryReserveExtension()
