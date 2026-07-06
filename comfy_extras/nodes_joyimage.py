import node_helpers
import comfy.utils
import torch
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


class TextEncodeJoyImageEditPlus(io.ComfyNode):
    """JoyImageEdit multi-image (Plus) text-encode node.

    Accepts 1-6 optional reference images. Each supplied image is
    bucket-resized independently (same buckets/resize as the single-image
    node), VAE-encoded, and appended in order to
    ``conditioning["reference_latents"]`` (image1 → ref0, image2 → ref1, ...).
    All resized images are passed to the VL tower in one call; the tokenizer
    emits one ``<|vision_start|><|image_pad|><|vision_end|>`` block per image.
    """

    MAX_IMAGES = 6

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeJoyImageEditPlus",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae"),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
                io.Image.Input("image5", optional=True),
                io.Image.Input("image6", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae, image1=None, image2=None, image3=None,
                image4=None, image5=None, image6=None) -> io.NodeOutput:
        images = [image1, image2, image3, image4, image5, image6]
        supplied = [img for img in images if img is not None]
        if len(supplied) == 0:
            raise ValueError(
                "TextEncodeJoyImageEditPlus requires at least one reference image."
            )

        resized_images = []
        ref_latents = []
        for image in supplied:
            samples = image.movedim(-1, 1)
            src_h, src_w = samples.shape[2], samples.shape[3]
            bucket_h, bucket_w = _find_best_bucket(src_h, src_w)

            resized = comfy.utils.common_upscale(samples, bucket_w, bucket_h, "bilinear", "center")
            resized_image = resized.movedim(1, -1)[:, :, :, :3]
            resized_images.append(resized_image)
            ref_latents.append(vae.encode(resized_image))

        tokens = clip.tokenize(prompt, images=resized_images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": ref_latents}, append=True,
        )

        # The last reference sets the target resolution; return it for VAEEncode and the
        # matching negative encode.
        return io.NodeOutput(conditioning, resized_images[-1])


class JoyImageGuidanceRescale(io.ComfyNode):
    """CFG combine + per-token L2 norm rescale required by JoyImageEdit.

    Wire this onto the model before sampling: JoyImageEdit's diffusers pipeline
    rescales the combined noise prediction back to the conditional branch's norm
    (comb * ||cond|| / ||comb||), the same rescale CFGNorm's pre_cfg branch does.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyImageGuidanceRescale",
            category="model/patch",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        def guidance_rescale(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            comb = uncond + cond_scale * (cond - uncond)
            cond_norm = torch.norm(cond, dim=1, keepdim=True)
            comb_norm = torch.norm(comb, dim=1, keepdim=True)
            return comb * (cond_norm / comb_norm.clamp_min(1e-6))

        m = model.clone()
        m.set_model_sampler_cfg_function(guidance_rescale)
        return io.NodeOutput(m)


class JoyImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeJoyImageEdit,
            TextEncodeJoyImageEditPlus,
            JoyImageGuidanceRescale,
        ]


async def comfy_entrypoint() -> JoyImageExtension:
    return JoyImageExtension()
