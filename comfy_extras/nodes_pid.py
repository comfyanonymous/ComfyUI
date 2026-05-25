"""PiD (Pixel Diffusion Decoder) node"""

from typing_extensions import override

import node_helpers
import comfy.latent_formats
from comfy_api.latest import ComfyExtension, io


_LATENT_FORMAT_CLASSES = {
    "flux": comfy.latent_formats.Flux,
    "sd3": comfy.latent_formats.SD3,
    "flux2": comfy.latent_formats.Flux2,
}


class PiDConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PiDConditioning",
            display_name="PiD Conditioning",
            category="advanced/conditioning",
            description=(
                "Attaches an LDM latent (Flux/SD3/Flux2/Z-Image) and a degrade_sigma scalar "
                "to a CONDITIONING for PiD decoding. Latent is renormalized into PiD space "
                "via the chosen latent_format. Z-Image uses 'flux'."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent", tooltip="LDM latent (from VAEEncode or a KSampler)."),
                io.Combo.Input(
                    "latent_format",
                    options=list(_LATENT_FORMAT_CLASSES.keys()),
                    default="flux",
                ),
                io.Float.Input(
                    "degrade_sigma", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="0 = clean latent. Increase to denoise corrupted LDM outputs.",
                ),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, positive, latent, latent_format: str, degrade_sigma: float) -> io.NodeOutput:
        lq_latent = _LATENT_FORMAT_CLASSES[latent_format]().process_in(latent["samples"])
        return io.NodeOutput(node_helpers.conditioning_set_values(
            positive, {"lq_latent": lq_latent, "degrade_sigma": float(degrade_sigma)},
        ))


class PiDExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PiDConditioning]


async def comfy_entrypoint() -> PiDExtension:
    return PiDExtension()
