"""PiD (Pixel Diffusion Decoder) node"""

import torch
from typing_extensions import override

import node_helpers
import comfy.latent_formats
from comfy_api.latest import ComfyExtension, io


class PiDConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PiDConditioning",
            display_name="PiD Conditioning",
            category="advanced/conditioning",
            description=(
                "Attaches a latent and a degrade_sigma scalar to a CONDITIONING for PiD decoding/upscaling"
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent", tooltip="latent (from VAEEncode or a KSampler)."),
                io.Combo.Input("latent_format", options=["flux", "sd3"], default="flux",
                               tooltip="Flux1 and Flux2 latents auto-detected from channel dim, sd3 has to be selected manually."),
                io.Float.Input(
                    "degrade_sigma", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="0 = clean latent. Increase to denoise corrupted latent outputs.",
                ),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, positive, latent, latent_format: str, degrade_sigma: float) -> io.NodeOutput:
        samples = latent["samples"]
        if latent_format == "flux":
            fmt_cls = comfy.latent_formats.Flux2 if samples.shape[1] == 128 else comfy.latent_formats.Flux
        else:
            fmt_cls = comfy.latent_formats.SD3
        lq_latent = fmt_cls().process_in(samples)
        sigma_t = torch.tensor([float(degrade_sigma)], dtype=torch.float32)
        return io.NodeOutput(node_helpers.conditioning_set_values(
            positive, {"lq_latent": lq_latent, "degrade_sigma": sigma_t},
        ))


class PiDExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PiDConditioning]


async def comfy_entrypoint() -> PiDExtension:
    return PiDExtension()
