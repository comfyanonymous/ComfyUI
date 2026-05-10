from typing_extensions import override

import torch

import comfy.model_management
import node_helpers
from comfy_api.latest import ComfyExtension, io

from comfy.ldm.hidream_o1.utils import find_closest_resolution


class EmptyHiDreamO1LatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyHiDreamO1LatentImage",
            display_name="Empty HiDream-O1 Latent Image",
            category="latent/image",
            description=(
                "Empty pixel-space latent for HiDream-O1-Image. When "
                "snap_to_predefined is on, dimensions are matched (by aspect "
                "ratio) to the upstream HiDream-O1 PREDEFINED_RESOLUTIONS list."
            ),
            inputs=[
                io.Int.Input(id="width", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="height", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="batch_size", default=1, min=1, max=64),
                io.Boolean.Input(
                    id="snap_to_predefined",
                    default=True,
                    tooltip=(
                        "Snap (W, H) to the closest aspect ratio in HiDream-O1's "
                        "PREDEFINED_RESOLUTIONS table for best parity with the "
                        "upstream CLI. Disable for arbitrary 32-aligned sizes."
                    ),
                ),
            ],
            outputs=[io.Latent().Output()],
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1,
                snap_to_predefined: bool = True) -> io.NodeOutput:
        if snap_to_predefined: #TODO: better way to handle this
            sw, sh = find_closest_resolution(width, height)
            width, height = sw, sh
        width = (width // 32) * 32
        height = (height // 32) * 32
        latent = torch.zeros(
            (batch_size, 3, height, width),
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": latent})


class HiDreamO1ReferenceImages(io.ComfyNode):
    """Attach reference images to both positive and negative conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HiDreamO1ReferenceImages",
            display_name="HiDream-O1 Reference Images",
            category="conditioning/image",
            description=(
                "Attach 1-10 reference images to conditioning, one for edit instruction"
                "or multiple for subject-driven personalization."
            ),
            inputs=[
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="negative"),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 11)],
                        min=1,
                    ),
                    tooltip=(
                        "Reference images. K=1 -> instruction edit; "
                        "K=2..10 -> subject-driven personalization."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, *, positive, negative, images: io.Autogrow.Type) -> io.NodeOutput:
        refs = [images[f"image_{i}"] for i in range(1, 11) if f"image_{i}" in images]
        positive = node_helpers.conditioning_set_values(positive, {"hidream_o1_ref_images": refs})
        negative = node_helpers.conditioning_set_values(negative, {"hidream_o1_ref_images": refs})
        return io.NodeOutput(positive, negative)


class HiDreamO1Sampling(io.ComfyNode):
    """Adjust HiDream-O1's flow-match sigma shift and noise scale together."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HiDreamO1Sampling",
            display_name="HiDream-O1 Sampling",
            category="advanced/model",
            description=(
                "Patch HiDream-O1's sigma shift and noise scaling factor. "
                "Base model defaults: shift=3.0, s_noise=8.0. "
                "Dev/flash sampler defaults: shift=1.0, s_noise=7.5."
            ),
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(
                    id="shift", default=3.0, min=0.0, max=100.0, step=0.01,
                    tooltip="Flow-match sigma shift. Defaults: 3.0 for base, 1.0 for dev.",
                ),
                io.Float.Input(
                    id="s_noise", default=8.0, min=0.0, max=64.0, step=0.1,
                    tooltip=("HiDream-O1 noise scale (CONST_SCALED_NOISE). Defaults: 8.0 for base, 7.5 for dev/flash."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift: float, s_noise: float) -> io.NodeOutput:
        import comfy.model_sampling
        m = model.clone()

        class _HiDreamO1SamplingPatched(
            comfy.model_sampling.ModelSamplingDiscreteFlow,
            comfy.model_sampling.CONST_SCALED_NOISE,
        ):
            pass

        ms = _HiDreamO1SamplingPatched(m.model.model_config)
        ms.set_parameters(shift=float(shift), multiplier=1000)
        ms._s_noise = float(s_noise)
        m.add_object_patch("model_sampling", ms)
        return io.NodeOutput(m)


class SamplerEulerFlashFlowmatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplerEulerFlashFlowmatch",
            display_name="Sampler Euler Flash Flowmatch",
            category="sampling/custom_sampling/samplers",
            description=("HiDream-O1 dev/flash sampler with tunable per-step noise"),
            inputs=[
                io.Float.Input(
                    id="s_noise_start", default=7.5, min=0.0, max=64.0, step=0.1,
                    tooltip="Per-step noise scale at the first sampling step.",
                ),
                io.Float.Input(
                    id="s_noise_end", default=7.5, min=0.0, max=64.0, step=0.1,
                    tooltip=(
                        "Per-step noise scale at the last step. Default: 7.5 for dev/flash. "
                        "Differ from s_noise_start to linearly ramp noise across steps."
                    ),
                ),
                io.Float.Input(
                    id="noise_clip_std", default=2.5, min=0.0, max=10.0, step=0.1,
                    tooltip=("Clamp per-step noise to +/- N*std. 0 disables.")
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, *, s_noise_start: float, s_noise_end: float,
                noise_clip_std: float) -> io.NodeOutput:
        import comfy.samplers
        import comfy.k_diffusion.sampling
        sampler = comfy.samplers.KSAMPLER(
            comfy.k_diffusion.sampling.sample_euler_flash_flowmatch,
            extra_options={
                "s_noise": float(s_noise_start),
                "s_noise_end": float(s_noise_end),
                "noise_clip_std": float(noise_clip_std),
            },
        )
        return io.NodeOutput(sampler)


class HiDreamO1Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyHiDreamO1LatentImage,
            HiDreamO1ReferenceImages,
            HiDreamO1Sampling,
            SamplerEulerFlashFlowmatch,
        ]


async def comfy_entrypoint() -> HiDreamO1Extension:
    return HiDreamO1Extension()
