import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.clip_vision
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
import folder_paths
import json

class CLIPTextEncodeHunyuanDiT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeHunyuanDiT",
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
    def execute(cls, clip, bert, mt5xl) -> io.NodeOutput:
        tokens = clip.tokenize(bert)
        tokens["mt5xl"] = clip.tokenize(mt5xl)["mt5xl"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

    encode = execute  # TODO: remove


class EmptyHunyuanLatentVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyHunyuanLatentVideo",
            display_name="Empty HunyuanVideo 1.0 Latent",
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
    def execute(cls, width, height, length, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 8})

    generate = execute  # TODO: remove


class EmptyHunyuanVideo15Latent(EmptyHunyuanLatentVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "EmptyHunyuanVideo15Latent"
        schema.display_name = "Empty HunyuanVideo 1.5 Latent"
        return schema

    @classmethod
    def execute(cls, width, height, length, batch_size=1) -> io.NodeOutput:
        # Using scale factor of 16 instead of 8
        latent = torch.zeros([batch_size, 32, ((length - 1) // 4) + 1, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 16})


class HunyuanVideo15ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanVideo15ImageToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=33, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 32, ((length - 1) // 4) + 1, height // 16, width // 16], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            encoded = vae.encode(start_image[:, :, :, :3])
            concat_latent_image = torch.zeros((latent.shape[0], 32, latent.shape[2], latent.shape[3], latent.shape[4]), device=comfy.model_management.intermediate_device())
            concat_latent_image[:, :, :encoded.shape[2], :, :] = encoded

            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


class HunyuanVideo15SuperResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanVideo15SuperResolution",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Latent.Input("latent"),
                io.Float.Input("noise_augmentation", default=0.70, min=0.0, max=1.0, step=0.01, advanced=True),

            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, latent, noise_augmentation, vae=None, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        in_latent = latent["samples"]
        in_channels = in_latent.shape[1]
        cond_latent = torch.zeros([in_latent.shape[0], in_channels * 2 + 2, in_latent.shape[-3], in_latent.shape[-2], in_latent.shape[-1]], device=comfy.model_management.intermediate_device())
        cond_latent[:, in_channels + 1 : 2 * in_channels + 1] = in_latent
        cond_latent[:, 2 * in_channels + 1] = 1
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image.movedim(-1, 1), in_latent.shape[-1] * 16, in_latent.shape[-2] * 16, "bilinear", "center").movedim(1, -1)
            encoded = vae.encode(start_image[:, :, :, :3])
            cond_latent[:, :in_channels, :encoded.shape[2], :, :] = encoded
            cond_latent[:, in_channels + 1, 0] = 1

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": cond_latent, "noise_augmentation": noise_augmentation})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": cond_latent, "noise_augmentation": noise_augmentation})
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        return io.NodeOutput(positive, negative, latent)


class LatentUpscaleModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentUpscaleModelLoader",
            display_name="Load Latent Upscale Model",
            category="loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("latent_upscale_models")),
            ],
            outputs=[
                io.LatentUpscaleModel.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)

        if "blocks.0.block.0.conv.weight" in sd:
            config = {
                "in_channels": sd["in_conv.conv.weight"].shape[1],
                "out_channels": sd["out_conv.conv.weight"].shape[0],
                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
                "global_residual": False,
            }
            model_type = "720p"
            model = HunyuanVideo15SRModel(model_type, config)
            model.load_sd(sd)
        elif "up.0.block.0.conv1.conv.weight" in sd:
            sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
            config = {
                "z_channels": sd["conv_in.conv.weight"].shape[1],
                "out_channels": sd["conv_out.conv.weight"].shape[0],
                "block_out_channels": tuple(sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0] for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))),
            }
            model_type = "1080p"
            model = HunyuanVideo15SRModel(model_type, config)
            model.load_sd(sd)
        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
            config = json.loads(metadata["config"])
            model = LatentUpsampler.from_config(config).to(dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
            model.load_state_dict(sd)

        return io.NodeOutput(model)


class HunyuanVideo15LatentUpscaleWithModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanVideo15LatentUpscaleWithModel",
            display_name="Hunyuan Video 15 Latent Upscale With Model",
            category="latent",
            inputs=[
                io.LatentUpscaleModel.Input("model"),
                io.Latent.Input("samples"),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], default="bilinear"),
                io.Int.Input("width", default=1280, min=0, max=16384, step=8),
                io.Int.Input("height", default=720, min=0, max=16384, step=8),
                io.Combo.Input("crop", options=["disabled", "center"]),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, samples, upscale_method, width, height, crop) -> io.NodeOutput:
        if width == 0 and height == 0:
            return io.NodeOutput(samples)
        else:
            if width == 0:
                height = max(64, height)
                width = max(64, round(samples["samples"].shape[-1] * height / samples["samples"].shape[-2]))
            elif height == 0:
                width = max(64, width)
                height = max(64, round(samples["samples"].shape[-2] * width / samples["samples"].shape[-1]))
            else:
                width = max(64, width)
                height = max(64, height)
            s = comfy.utils.common_upscale(samples["samples"], width // 16, height // 16, upscale_method, crop)
            s = model.resample_latent(s)
            return io.NodeOutput({"samples": s.cpu().float()})


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
            node_id="TextEncodeHunyuanVideo_ImageToVideo",
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
                    advanced=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_vision_output, prompt, image_interleave) -> io.NodeOutput:
        tokens = clip.tokenize(prompt, llama_template=PROMPT_TEMPLATE_ENCODE_VIDEO_I2V, image_embeds=clip_vision_output.mm_projected, image_interleave=image_interleave)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

    encode = execute  # TODO: remove


class TextEncodeHunyuanVideo15Omni(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeHunyuanVideo15Omni",
            display_name="Text Encode HunyuanVideo 15 Omni",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("task", options=["t2v", "i2v", "interpolation", "reference2v", "editing", "tiv2v"], default="t2v"),
                io.Boolean.Input("use_visual_inputs", default=True, advanced=True),
                io.Int.Input("max_visual_inputs", default=8, min=1, max=64, advanced=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @staticmethod
    def _task_system_prompt(task: str) -> str:
        prompts = {
            "t2v": "Describe a high-quality target video from the user's request with concrete scene details, motion, camera behavior, and style.",
            "i2v": "Describe a target video that should stay consistent with the provided reference image while following the user's request.",
            "interpolation": "Describe a target video that smoothly transitions between the provided keyframe images while following the user's request.",
            "reference2v": "Describe a target video that composes the provided reference subjects into a coherent scene following the user's request.",
            "editing": "Describe an edited output video that follows the user's instruction while preserving relevant source video content.",
            "tiv2v": "Describe an edited output video using both the provided source video and reference image guidance according to the user's instruction.",
        }
        return prompts.get(task, prompts["t2v"])

    @classmethod
    def _build_template(cls, task: str, image_count: int) -> str:
        system_prompt = cls._task_system_prompt(task)
        visual_tokens = "<|vision_start|><|image_pad|><|vision_end|>\n" * image_count
        return (
            "<|im_start|>system\n"
            f"{system_prompt}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{visual_tokens}" + "{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _extract_image_embeds(clip_vision_output, max_visual_inputs: int):
        if clip_vision_output is None:
            return []
        mm_projected = getattr(clip_vision_output, "mm_projected", None)
        if mm_projected is None:
            return []
        if mm_projected.ndim == 2:
            return [mm_projected]
        count = min(mm_projected.shape[0], max_visual_inputs)
        return [mm_projected[i] for i in range(count)]

    @classmethod
    def execute(cls, clip, prompt, task, use_visual_inputs, max_visual_inputs, clip_vision_output=None) -> io.NodeOutput:
        image_embeds = cls._extract_image_embeds(clip_vision_output, max_visual_inputs) if use_visual_inputs else []
        template = cls._build_template(task, len(image_embeds))

        # HunyuanVideo 1.5 tokenizers use `images=...`; HunyuanVideo 1.0 uses `image_embeds=...`.
        try:
            tokens = clip.tokenize(prompt, llama_template=template, images=image_embeds)
        except TypeError:
            embeds = None
            if len(image_embeds) > 0:
                embeds = torch.stack(image_embeds, dim=0)
            tokens = clip.tokenize(prompt, llama_template=template, image_embeds=embeds, image_interleave=1)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

    encode = execute  # TODO: remove


class HunyuanClipVisionOutputConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanClipVisionOutputConcat",
            display_name="Hunyuan CLIP Vision Output Concat",
            category="conditioning/video_models",
            inputs=[
                io.ClipVisionOutput.Input("clip_vision_output_1"),
                io.ClipVisionOutput.Input("clip_vision_output_2", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output_3", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output_4", optional=True),
            ],
            outputs=[
                io.ClipVisionOutput.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip_vision_output_1, clip_vision_output_2=None, clip_vision_output_3=None, clip_vision_output_4=None) -> io.NodeOutput:
        outputs = [o for o in (clip_vision_output_1, clip_vision_output_2, clip_vision_output_3, clip_vision_output_4) if o is not None]
        merged = comfy.clip_vision.Output()
        tensor_attrs = ["last_hidden_state", "image_embeds", "penultimate_hidden_states", "all_hidden_states", "mm_projected"]
        for attr in tensor_attrs:
            values = [getattr(o, attr) for o in outputs if hasattr(o, attr)]
            if len(values) > 0 and torch.is_tensor(values[0]):
                setattr(merged, attr, torch.cat(values, dim=0))

        image_sizes = []
        for o in outputs:
            if hasattr(o, "image_sizes"):
                image_sizes.extend(getattr(o, "image_sizes"))
        if len(image_sizes) > 0:
            merged.image_sizes = image_sizes
        return io.NodeOutput(merged)


class HunyuanVideo15OmniConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanVideo15OmniConditioning",
            display_name="HunyuanVideo 15 Omni Conditioning",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Combo.Input("task", options=["t2v", "i2v", "interpolation", "reference2v", "editing", "tiv2v"], default="t2v"),
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("reference_images", optional=True, tooltip="For i2v/interpolation/reference2v/tiv2v."),
                io.Image.Input("condition_video", optional=True, tooltip="For editing/tiv2v."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @staticmethod
    def _latent_length(length: int) -> int:
        return ((length - 1) // 4) + 1

    @staticmethod
    def _upscale_frames(frames: torch.Tensor, width: int, height: int):
        return comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

    @classmethod
    def _encode_single_image(cls, vae, image: torch.Tensor, width: int, height: int):
        upscaled = cls._upscale_frames(image[:1], width, height)
        return vae.encode(upscaled[:, :, :, :3])

    @classmethod
    def _encode_video(cls, vae, video: torch.Tensor, width: int, height: int, length: int):
        upscaled = cls._upscale_frames(video[:length], width, height)
        return vae.encode(upscaled[:, :, :, :3])

    @staticmethod
    def _assign_frame(target: torch.Tensor, source: torch.Tensor, frame_idx: int):
        if frame_idx < 0 or frame_idx >= target.shape[2]:
            return
        target[:, :, frame_idx:frame_idx + 1] = source[:, :, :1]

    @classmethod
    def execute(cls, positive, negative, vae, task, width, height, length, batch_size, reference_images=None, condition_video=None, clip_vision_output=None) -> io.NodeOutput:
        latent_length = cls._latent_length(length)
        latent = torch.zeros([batch_size, 32, latent_length, height // 16, width // 16], device=comfy.model_management.intermediate_device())

        if task == "t2v":
            if clip_vision_output is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
            return io.NodeOutput(positive, negative, {"samples": latent})

        cond_latent = torch.zeros_like(latent[:1])
        omni_mask = torch.zeros((latent_length,), device=cond_latent.device, dtype=cond_latent.dtype)

        if task == "i2v":
            if reference_images is None or reference_images.shape[0] < 1:
                raise ValueError("Task i2v requires at least one reference image.")
            encoded = cls._encode_single_image(vae, reference_images, width, height)
            cls._assign_frame(cond_latent, encoded, 0)
            omni_mask[0] = 1.0

        elif task == "interpolation":
            if reference_images is None or reference_images.shape[0] < 2:
                raise ValueError("Task interpolation requires at least two reference images.")
            encoded_first = cls._encode_single_image(vae, reference_images[:1], width, height)
            encoded_last = cls._encode_single_image(vae, reference_images[-1:], width, height)
            cls._assign_frame(cond_latent, encoded_first, 0)
            cls._assign_frame(cond_latent, encoded_last, latent_length - 1)
            omni_mask[0] = 1.0
            omni_mask[-1] = 1.0

        elif task == "reference2v":
            if reference_images is None or reference_images.shape[0] < 1:
                raise ValueError("Task reference2v requires at least one reference image.")
            num_refs = min(reference_images.shape[0], max(1, latent_length - 1))
            for idx in range(num_refs):
                encoded = cls._encode_single_image(vae, reference_images[idx:idx + 1], width, height)
                frame_idx = min(idx + 1, latent_length - 1)
                cls._assign_frame(cond_latent, encoded, frame_idx)
                omni_mask[frame_idx] = 1.0

        elif task == "editing":
            if condition_video is None or condition_video.shape[0] < 1:
                raise ValueError("Task editing requires condition_video.")
            encoded = cls._encode_video(vae, condition_video, width, height, length)
            valid_frames = min(latent_length, encoded.shape[2])
            cond_latent[:, :, :valid_frames] = encoded[:, :, :valid_frames]
            omni_mask[:valid_frames] = 1.0

        elif task == "tiv2v":
            if condition_video is None or condition_video.shape[0] < 1:
                raise ValueError("Task tiv2v requires condition_video.")
            if reference_images is None or reference_images.shape[0] < 1:
                raise ValueError("Task tiv2v requires at least one reference image.")
            encoded_video = cls._encode_video(vae, condition_video, width, height, length)
            valid_frames = min(latent_length, encoded_video.shape[2])
            cond_latent[:, :, :valid_frames] = encoded_video[:, :, :valid_frames]
            omni_mask[:valid_frames] = 1.0

            encoded_ref = cls._encode_single_image(vae, reference_images[:1], width, height)
            ref_idx = 1 if latent_length > 1 else 0
            cond_latent[:, :, ref_idx:ref_idx + 1] += encoded_ref[:, :, :1]
            omni_mask[ref_idx] += 1.0

        cond_latent = comfy.utils.resize_to_batch_size(cond_latent, batch_size)
        # BaseModel/HunyuanVideo15 inverts concat_mask (mask = 1 - concat_mask), so pass the pre-inverted mask.
        concat_mask = (1.0 - omni_mask).view(1, 1, latent_length, 1, 1).expand(cond_latent.shape[0], 1, latent_length, cond_latent.shape[-2], cond_latent.shape[-1]).to(cond_latent.dtype)

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": cond_latent, "concat_mask": concat_mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": cond_latent, "concat_mask": concat_mask})
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        return io.NodeOutput(positive, negative, {"samples": latent})


class HunyuanImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanImageToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=53, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Combo.Input("guidance_type", options=["v1 (concat)", "v2 (replace)", "custom"], advanced=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, vae, width, height, length, batch_size, guidance_type, start_image=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            concat_latent_image = vae.encode(start_image)
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
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

    encode = execute  # TODO: remove


class EmptyHunyuanImageLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyHunyuanImageLatent",
            category="latent",
            inputs=[
                io.Int.Input("width", default=2048, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=2048, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 64, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples":latent})

    generate = execute  # TODO: remove


class HunyuanRefinerLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanRefinerLatent",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Float.Input("noise_augmentation", default=0.10, min=0.0, max=1.0, step=0.01, advanced=True),

            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, latent, noise_augmentation) -> io.NodeOutput:
        latent = latent["samples"]
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": latent, "noise_augmentation": noise_augmentation})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": latent, "noise_augmentation": noise_augmentation})
        out_latent = {}
        out_latent["samples"] = torch.zeros([latent.shape[0], 32, latent.shape[-3], latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, out_latent)


class HunyuanExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeHunyuanDiT,
            TextEncodeHunyuanVideo_ImageToVideo,
            TextEncodeHunyuanVideo15Omni,
            HunyuanClipVisionOutputConcat,
            EmptyHunyuanLatentVideo,
            EmptyHunyuanVideo15Latent,
            HunyuanVideo15ImageToVideo,
            HunyuanVideo15OmniConditioning,
            HunyuanVideo15SuperResolution,
            HunyuanVideo15LatentUpscaleWithModel,
            LatentUpscaleModelLoader,
            HunyuanImageToVideo,
            EmptyHunyuanImageLatent,
            HunyuanRefinerLatent,
        ]


async def comfy_entrypoint() -> HunyuanExtension:
    return HunyuanExtension()
