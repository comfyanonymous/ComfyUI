import math

from typing_extensions import override

import node_helpers
from comfy_api.latest import ComfyExtension, io

# sigma 0.9 at the default shift 1.15: a generation shows Qwen3-VL the images of its caption's text from there on
TEXT_IMAGES_START = math.exp(1.15) / (9 + math.exp(1.15))


def has_images(tokens):
    return any(isinstance(t[0], dict) for batches in tokens.values() for batch in batches for t in batch)


class TextEncodeFibo2(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeFibo2",
            category="model/conditioning/fibo",
            description="Encode a Fibo 2.0 prompt, a structured JSON caption or plain text. With images, the prompt is the edit instruction, or a JSON caption of the result with an edit_instruction key, and the images become reference latents of both outputs when a vae is connected.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True),
                io.String.Input("negative_prompt", multiline=True, advanced=True, tooltip="Empty is the null caption Fibo 2.0 was trained with for classifier-free guidance."),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Images to edit, referred to as <image_1>, <image_2> and so on in the instruction.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt="", vae=None, images: io.Autogrow.Type = None) -> io.NodeOutput:
        images = images or {}
        images = [images[name] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[name] is not None]
        images = [image[i:i + 1] for image in images for i in range(image.shape[0])]

        tokens = clip.tokenize(prompt, images=images)
        positive = clip.encode_from_tokens_scheduled(tokens)
        if len(images) == 0 and has_images(tokens):
            base = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, text_images=False))
            # ranges include both ends: the base stops just above sigma 0.9, which already has the text images
            positive = (node_helpers.conditioning_set_values(base, {"end_percent": TEXT_IMAGES_START - 1e-6})
                        + node_helpers.conditioning_set_values(positive, {"start_percent": TEXT_IMAGES_START}))
        # the negative is text only; an edit's reference latents condition both
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        if vae is not None and len(images) > 0:
            ref_latents = [vae.encode(image) for image in images]
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(positive, negative)


class Fibo2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeFibo2,
        ]


async def comfy_entrypoint() -> Fibo2Extension:
    return Fibo2Extension()
