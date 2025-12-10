import node_helpers
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import torch
import math
import nodes

class CLIPTextEncodeFlux(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeFlux",
            category="advanced/conditioning/flux",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input("t5xxl", multiline=True, dynamic_prompts=True),
                io.Float.Input("guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_l, t5xxl, guidance) -> io.NodeOutput:
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}))

    encode = execute  # TODO: remove

class EmptyFlux2LatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyFlux2LatentImage",
            display_name="Empty Flux 2 Latent",
            category="latent",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 128, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent})

class FluxGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxGuidance",
            category="advanced/conditioning/flux",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input("guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, guidance) -> io.NodeOutput:
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


class FluxDisableGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxDisableGuidance",
            category="advanced/conditioning/flux",
            description="This node completely disables the guidance embed on Flux and Flux like models",
            inputs=[
                io.Conditioning.Input("conditioning"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning) -> io.NodeOutput:
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": None})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class FluxKontextImageScale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxKontextImageScale",
            category="advanced/conditioning/flux",
            description="This node resizes the image to one that is more optimal for flux kontext.",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        return io.NodeOutput(image)

    scale = execute  # TODO: remove


class FluxKontextMultiReferenceLatentMethod(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxKontextMultiReferenceLatentMethod",
            category="advanced/conditioning/flux",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Combo.Input(
                    "reference_latents_method",
                    options=["offset", "index", "uxo/uno"],
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, conditioning, reference_latents_method) -> io.NodeOutput:
        if "uxo" in reference_latents_method or "uso" in reference_latents_method:
            reference_latents_method = "uxo"
        c = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": reference_latents_method})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


def generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps


class Flux2Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux2Scheduler",
            category="sampling/custom_sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=4096),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[
                io.Sigmas.Output(),
            ],
        )

    @classmethod
    def execute(cls, steps, width, height) -> io.NodeOutput:
        seq_len = (width * height / (16 * 16))
        sigmas = get_schedule(steps, round(seq_len))
        return io.NodeOutput(sigmas)


class FluxExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeFlux,
            FluxGuidance,
            FluxDisableGuidance,
            FluxKontextImageScale,
            FluxKontextMultiReferenceLatentMethod,
            EmptyFlux2LatentImage,
            Flux2Scheduler,
        ]


async def comfy_entrypoint() -> FluxExtension:
    return FluxExtension()
