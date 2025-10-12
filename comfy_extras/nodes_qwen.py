import node_helpers
import comfy.utils
import math
import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class TextEncodeQwenImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEdit",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image=None) -> io.NodeOutput:
        ref_latent = None
        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        return io.NodeOutput(conditioning)

class QwenImageInpaintConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImageInpaintConditioning",
            category="advanced/conditioning",
            description=(
                "Prepares conditioning and latents for Qwen Image Edit inpainting."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Boolean.Input(
                    "use_noise_mask",
                    default=True,
                    tooltip="When enabled, provide the resized mask as noise mask so sampling only affects the painted region.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        image,
        mask,
        use_noise_mask=True,
    ) -> io.NodeOutput:
        if image.ndim != 4:
            raise ValueError("Expected image tensor with shape [B, H, W, C].")

        image = image[:, :, :, :3]
        batch, height, width, _ = image.shape

        requested_width = float(width)
        requested_height = float(height)

        spacial_scale = vae.spacial_compression_encode()
        if isinstance(spacial_scale, tuple):
            spacial_scale = spacial_scale[-1]
        spacial_scale = int(spacial_scale)
        align_multiple = max(1, spacial_scale * 2)

        target_width = max(
            align_multiple,
            round(requested_width / align_multiple) * align_multiple,
        )
        target_height = max(
            align_multiple,
            round(requested_height / align_multiple) * align_multiple,
        )

        samples = image.movedim(-1, 1)
        resized = comfy.utils.common_upscale(samples, target_width, target_height, "area", "disabled")
        resized = resized.movedim(1, -1)

        mask_tensor = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(target_height, target_width), mode="bilinear")
        mask_tensor = mask_tensor.clamp(0.0, 1.0)
        mask_tensor = mask_tensor.to(resized.dtype)
        mask_tensor = comfy.utils.resize_to_batch_size(mask_tensor, batch)

        masked_pixels = resized.clone()
        keep_region = (1.0 - mask_tensor.round()).squeeze(1)
        masked_pixels[:, :, :, :3] = (masked_pixels[:, :, :, :3] - 0.5) * keep_region.unsqueeze(-1) + 0.5

        concat_latent = vae.encode(masked_pixels)
        orig_latent = vae.encode(resized)

        out_latent: dict[str, torch.Tensor] = {"samples": orig_latent}
        if use_noise_mask:
            out_latent["noise_mask"] = mask_tensor

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent, "concat_mask": mask_tensor}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent, "concat_mask": mask_tensor}
        )

        return io.NodeOutput(positive, negative, out_latent)

class TextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)


class QwenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeQwenImageEdit,
            QwenImageInpaintConditioning,
            TextEncodeQwenImageEditPlus,
        ]


async def comfy_entrypoint() -> QwenExtension:
    return QwenExtension()
