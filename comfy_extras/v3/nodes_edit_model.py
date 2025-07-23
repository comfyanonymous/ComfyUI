from __future__ import annotations

import node_helpers
from comfy_api.v3 import io


class ReferenceLatent(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReferenceLatent_V3",
            category="advanced/conditioning/edit_models",
            description="This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, conditioning, latent=None):
        if latent is not None:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": [latent["samples"]]}, append=True
            )
        return io.NodeOutput(conditioning)


NODES_LIST = [
    ReferenceLatent,
]
