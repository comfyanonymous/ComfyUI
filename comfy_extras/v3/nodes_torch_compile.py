from __future__ import annotations

from comfy_api.torch_helpers import set_torch_compile_wrapper
from comfy_api.latest import io


class TorchCompileModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TorchCompileModel_V3",
            category="_for_testing",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("backend", options=["inductor", "cudagraphs"]),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, backend):
        m = model.clone()
        set_torch_compile_wrapper(model=m, backend=backend)
        return io.NodeOutput(m)


NODES_LIST = [
    TorchCompileModel,
]
