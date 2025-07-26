from __future__ import annotations

import torch

import comfy.model_management
import node_helpers
import nodes
from comfy_api.latest import io


class CLIPTextEncodeHunyuanDiT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeHunyuanDiT_V3",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("bert", multiline=True, dynamic_prompts=True),
                io.String.Input("mt5xl", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, bert, mt5xl):
        tokens = clip.tokenize(bert)
        tokens["mt5xl"] = clip.tokenize(mt5xl)["mt5xl"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class EmptyHunyuanLatentVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyHunyuanLatentVideo_V3",
            category="latent/video",
            inputs=[
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=25, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, length, batch_size):
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples":latent})


PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


class TextEncodeHunyuanVideo_ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeHunyuanVideo_ImageToVideo_V3",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.ClipVisionOutput.Input("clip_vision_output"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "image_interleave",
                    default=2,
                    min=1,
                    max=512,
                    tooltip="How much the image influences things vs the text prompt. Higher number means more influence from the text prompt.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_vision_output, prompt, image_interleave):
        tokens = clip.tokenize(
            prompt, llama_template=PROMPT_TEMPLATE_ENCODE_VIDEO_I2V,
            image_embeds=clip_vision_output.mm_projected,
            image_interleave=image_interleave,
        )
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class HunyuanImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanImageToVideo_V3",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=53, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Combo.Input("guidance_type", options=["v1 (concat)", "v2 (replace)", "custom"]),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, vae, width, height, length, batch_size, guidance_type, start_image=None):
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        out_latent = {}

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

            concat_latent_image = vae.encode(start_image)
            mask = torch.ones(
                (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                device=start_image.device,
                dtype=start_image.dtype,
            )
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            if guidance_type == "v1 (concat)":
                cond = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            elif guidance_type == "v2 (replace)":
                cond = {'guiding_frame_index': 0}
                latent[:, :, :concat_latent_image.shape[2]] = concat_latent_image
                out_latent["noise_mask"] = mask
            elif guidance_type == "custom":
                cond = {"ref_latent": concat_latent_image}

            positive = node_helpers.conditioning_set_values(positive, cond)

        out_latent["samples"] = latent
        return io.NodeOutput(positive, out_latent)


NODES_LIST: list[type[io.ComfyNode]] = [
    CLIPTextEncodeHunyuanDiT,
    EmptyHunyuanLatentVideo,
    HunyuanImageToVideo,
    TextEncodeHunyuanVideo_ImageToVideo,
]
