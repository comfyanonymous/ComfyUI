import node_helpers
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


# fmt: off
BUCKETS_1024 = [
    (512, 1792), (512, 1856), (512, 1920), (512, 1984), (512, 2048),
    (576, 1600), (576, 1664), (576, 1728), (576, 1792),
    (640, 1472), (640, 1536), (640, 1600),
    (704, 1344), (704, 1408), (704, 1472),
    (768, 1216), (768, 1280), (768, 1344),
    (832, 1152), (832, 1216),
    (896, 1088), (896, 1152),
    (960, 1024), (960, 1088),
    (1024, 960), (1024, 1024),
    (1088, 896), (1088, 960),
    (1152, 832), (1152, 896),
    (1216, 768), (1216, 832),
    (1280, 768),
    (1344, 704), (1344, 768),
    (1408, 704),
    (1472, 640), (1472, 704),
    (1536, 640),
    (1600, 576), (1600, 640),
    (1664, 576),
    (1728, 576),
    (1792, 512), (1792, 576),
    (1856, 512),
    (1920, 512),
    (1984, 512),
    (2048, 512),
]
# fmt: on


def _find_best_bucket(height: int, width: int) -> tuple[int, int]:
    target_ratio = height / width
    return min(BUCKETS_1024, key=lambda hw: abs(hw[0] / hw[1] - target_ratio))


class TextEncodeJoyImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeJoyImageEdit",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
            ],
            outputs=[
                io.Conditioning.Output(),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae, image) -> io.NodeOutput:
        samples = image.movedim(-1, 1)
        src_h, src_w = samples.shape[2], samples.shape[3]
        bucket_h, bucket_w = _find_best_bucket(src_h, src_w)

        resized = comfy.utils.common_upscale(samples, bucket_w, bucket_h, "bilinear", "center")
        resized_image = resized.movedim(1, -1)[:, :, :, :3]

        tokens = clip.tokenize(prompt, images=[resized_image])
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latent = vae.encode(resized_image)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)

        return io.NodeOutput(conditioning, resized_image)


class JoyImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeJoyImageEdit,
        ]


async def comfy_entrypoint() -> JoyImageExtension:
    return JoyImageExtension()
