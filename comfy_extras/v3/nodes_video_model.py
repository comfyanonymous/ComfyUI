from __future__ import annotations

import torch

import comfy.sd
import comfy.utils
import comfy_extras.nodes_model_merging
import folder_paths
import node_helpers
import nodes
from comfy_api.latest import io


class ConditioningSetAreaPercentageVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConditioningSetAreaPercentageVideo_V3",
            category="conditioning",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input("width", default=1.0, min=0, max=1.0, step=0.01),
                io.Float.Input("height", default=1.0, min=0, max=1.0, step=0.01),
                io.Float.Input("temporal", default=1.0, min=0, max=1.0, step=0.01),
                io.Float.Input("x", default=0, min=0, max=1.0, step=0.01),
                io.Float.Input("y", default=0, min=0, max=1.0, step=0.01),
                io.Float.Input("z", default=0, min=0, max=1.0, step=0.01),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, width, height, temporal, x, y, z, strength):
        c = node_helpers.conditioning_set_values(
            conditioning,
            {
                "area": ("percentage", temporal, height, width, z, y, x),
                "strength": strength,
                "set_area_to_bounds": False
            ,}
        )
        return io.NodeOutput(c)


class ImageOnlyCheckpointLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageOnlyCheckpointLoader_V3",
            display_name="Image Only Checkpoint Loader (img2vid model) _V3",
            category="loaders/video_models",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
            ],
            outputs=[
                io.Model.Output(),
                io.ClipVision.Output(),
                io.Vae.Output(),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=False,
            output_clipvision=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(out[0], out[3], out[2])


class ImageOnlyCheckpointSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageOnlyCheckpointSave_V3",
            category="advanced/model_merging",
            inputs=[
                io.Model.Input("model"),
                io.ClipVision.Input("clip_vision"),
                io.Vae.Input("vae"),
                io.String.Input("filename_prefix", default="checkpoints/ComfyUI"),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, model, clip_vision, vae, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        comfy_extras.nodes_model_merging.save_checkpoint(
            model,
            clip_vision=clip_vision,
            vae=vae,
            filename_prefix=filename_prefix,
            output_dir=output_dir,
            prompt=cls.hidden.prompt,
            extra_pnginfo=cls.hidden.extra_pnginfo,
        )
        return io.NodeOutput()


class SVD_img2vid_Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SVD_img2vid_Conditioning_V3",
            category="conditioning/video_models",
            inputs=[
                io.ClipVision.Input("clip_vision"),
                io.Image.Input("init_image"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=576, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("video_frames", default=14, min=1, max=4096),
                io.Int.Input("motion_bucket_id", default=127, min=1, max=1023),
                io.Int.Input("fps", default=6, min=1, max=1024),
                io.Float.Input("augmentation_level", default=0.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(
            init_image.movedim(-1,1), width, height, "bilinear", "center"
        ).movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [
            [
                pooled,
                {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": t},
            ]
        ]
        negative = [
            [
                torch.zeros_like(pooled),
                {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": torch.zeros_like(t)},
            ]
        ]
        latent = torch.zeros([video_frames, 4, height // 8, width // 8])
        return io.NodeOutput(positive, negative, {"samples":latent})


class VideoLinearCFGGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoLinearCFGGuidance_V3",
            category="sampling/video_models",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("min_cfg", default=1.0, min=0.0, max=100.0, step=0.5, round=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, min_cfg):
        def linear_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]

            scale = torch.linspace(
                min_cfg, cond_scale, cond.shape[0], device=cond.device
            ).reshape((cond.shape[0], 1, 1, 1))
            return uncond + scale * (cond - uncond)

        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg)
        return io.NodeOutput(m)


class VideoTriangleCFGGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoTriangleCFGGuidance_V3",
            category="sampling/video_models",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("min_cfg", default=1.0, min=0.0, max=100.0, step=0.5, round=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, min_cfg):
        def linear_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            period = 1.0
            values = torch.linspace(0, 1, cond.shape[0], device=cond.device)
            values = 2 * (values / period - torch.floor(values / period + 0.5)).abs()
            scale = (values * (cond_scale - min_cfg) + min_cfg).reshape((cond.shape[0], 1, 1, 1))

            return uncond + scale * (cond - uncond)

        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg)
        return io.NodeOutput(m)


NODES_LIST = [
    ConditioningSetAreaPercentageVideo,
    ImageOnlyCheckpointLoader,
    ImageOnlyCheckpointSave,
    SVD_img2vid_Conditioning,
    VideoLinearCFGGuidance,
    VideoTriangleCFGGuidance,
]
