import comfy.sd
import comfy.utils
import comfy.model_base
import comfy.model_management
import comfy.model_sampling

import torch
import folder_paths
import json
import os

from comfy.cli_args import args
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override


class ModelMergeSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelMergeSimple",
            category="advanced/model_merging",
            inputs=[
                io.Model.Input("model1"),
                io.Model.Input("model2"),
                io.Float.Input("ratio", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model1, model2, ratio) -> io.NodeOutput:
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return io.NodeOutput(m)

    merge = execute  # TODO: remove


class ModelSubtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelMergeSubtract",
            category="advanced/model_merging",
            inputs=[
                io.Model.Input("model1"),
                io.Model.Input("model2"),
                io.Float.Input("multiplier", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model1, model2, multiplier) -> io.NodeOutput:
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, - multiplier, multiplier)
        return io.NodeOutput(m)

    merge = execute  # TODO: remove


class ModelAdd(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelMergeAdd",
            category="advanced/model_merging",
            inputs=[
                io.Model.Input("model1"),
                io.Model.Input("model2"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model1, model2) -> io.NodeOutput:
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0, 1.0)
        return io.NodeOutput(m)

    merge = execute  # TODO: remove


class CLIPMergeSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPMergeSimple",
            category="advanced/model_merging",
            inputs=[
                io.Clip.Input("clip1"),
                io.Clip.Input("clip2"),
                io.Float.Input("ratio", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip1, clip2, ratio) -> io.NodeOutput:
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return io.NodeOutput(m)

    merge = execute  # TODO: remove


class CLIPSubtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPMergeSubtract",
            search_aliases=["clip difference", "text encoder subtract"],
            category="advanced/model_merging",
            inputs=[
                io.Clip.Input("clip1"),
                io.Clip.Input("clip2"),
                io.Float.Input("multiplier", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip1, clip2, multiplier) -> io.NodeOutput:
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, - multiplier, multiplier)
        return io.NodeOutput(m)

    merge = execute  # TODO: remove


class CLIPAdd(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPMergeAdd",
            search_aliases=["combine clip"],
            category="advanced/model_merging",
            inputs=[
                io.Clip.Input("clip1"),
                io.Clip.Input("clip2"),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip1, clip2) -> io.NodeOutput:
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, 1.0, 1.0)
        return io.NodeOutput(m)

    merge = execute  # TODO: remove


class ModelMergeBlocks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelMergeBlocks",
            category="advanced/model_merging",
            inputs=[
                io.Model.Input("model1"),
                io.Model.Input("model2"),
                io.Float.Input("input", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("middle", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("out", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model1, model2, **kwargs) -> io.NodeOutput:
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        default_ratio = next(iter(kwargs.values()))

        for k in kp:
            ratio = default_ratio
            k_unet = k[len("diffusion_model."):]

            last_arg_size = 0
            for arg in kwargs:
                if k_unet.startswith(arg) and last_arg_size < len(arg):
                    ratio = kwargs[arg]
                    last_arg_size = len(arg)

            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return io.NodeOutput(m)

    merge = execute  # TODO: remove


def save_checkpoint(model, clip=None, vae=None, clip_vision=None, filename_prefix=None, output_dir=None, prompt=None, extra_pnginfo=None):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {}

    enable_modelspec = True
    if isinstance(model.model, comfy.model_base.SDXL):
        if isinstance(model.model, comfy.model_base.SDXL_instructpix2pix):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-edit"
        else:
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    elif isinstance(model.model, comfy.model_base.SDXLRefiner):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
    elif isinstance(model.model, comfy.model_base.SVD_img2vid):
        metadata["modelspec.architecture"] = "stable-video-diffusion-img2vid-v1"
    elif isinstance(model.model, comfy.model_base.SD3):
        metadata["modelspec.architecture"] = "stable-diffusion-v3-medium" #TODO: other SD3 variants
    else:
        enable_modelspec = False

    if enable_modelspec:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = "{} {}".format(filename, counter)

    #TODO:
    # "stable-diffusion-v1", "stable-diffusion-v1-inpainting", "stable-diffusion-v2-512",
    # "stable-diffusion-v2-768-v", "stable-diffusion-v2-unclip-l", "stable-diffusion-v2-unclip-h",
    # "v2-inpainting"

    extra_keys = {}
    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
        if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
            extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
            extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

    if model.model.model_type == comfy.model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"
        extra_keys["v_pred"] = torch.tensor([])
        if getattr(model_sampling, "zsnr", False):
            extra_keys["ztsnr"] = torch.tensor([])

    if not args.disable_metadata:
        metadata["prompt"] = prompt_info
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    output_checkpoint = f"{filename}_{counter:05}_.safetensors"
    output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

    comfy.sd.save_checkpoint(output_checkpoint, model, clip, vae, clip_vision, metadata=metadata, extra_keys=extra_keys)


class CheckpointSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CheckpointSave",
            display_name="Save Checkpoint",
            search_aliases=["save model", "export checkpoint", "merge save"],
            category="advanced/model_merging",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("filename_prefix", default="checkpoints/ComfyUI"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, clip, vae, filename_prefix) -> io.NodeOutput:
        save_checkpoint(model, clip=clip, vae=vae, filename_prefix=filename_prefix, output_dir=folder_paths.get_output_directory(), prompt=cls.hidden.prompt, extra_pnginfo=cls.hidden.extra_pnginfo)
        return io.NodeOutput()

    save = execute  # TODO: remove


class CLIPSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPSave",
            category="advanced/model_merging",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("filename_prefix", default="clip/ComfyUI"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, clip, filename_prefix) -> io.NodeOutput:
        prompt_info = ""
        if cls.hidden.prompt is not None:
            prompt_info = json.dumps(cls.hidden.prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata["format"] = "pt"
            metadata["prompt"] = prompt_info
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        comfy.model_management.load_models_gpu([clip.load_model()], force_patch_weights=True)
        clip_sd = clip.get_sd()

        output_dir = folder_paths.get_output_directory()
        for prefix in ["clip_l.", "clip_g.", "clip_h.", "t5xxl.", "pile_t5xl.", "mt5xl.", "umt5xxl.", "t5base.", "gemma2_2b.", "llama.", "hydit_clip.", ""]:
            k = list(filter(lambda a: a.startswith(prefix), clip_sd.keys()))
            current_clip_sd = {}
            for x in k:
                current_clip_sd[x] = clip_sd.pop(x)
            if len(current_clip_sd) == 0:
                continue

            p = prefix[:-1]
            replace_prefix = {}
            filename_prefix_ = filename_prefix
            if len(p) > 0:
                filename_prefix_ = "{}_{}".format(filename_prefix_, p)
                replace_prefix[prefix] = ""
            replace_prefix["transformer."] = ""

            full_output_folder, filename, counter, subfolder, filename_prefix_ = folder_paths.get_save_image_path(filename_prefix_, output_dir)

            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
            output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

            current_clip_sd = comfy.utils.state_dict_prefix_replace(current_clip_sd, replace_prefix)

            comfy.utils.save_torch_file(current_clip_sd, output_checkpoint, metadata=metadata)
        return io.NodeOutput()

    save = execute  # TODO: remove


class VAESave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAESave",
            category="advanced/model_merging",
            inputs=[
                io.Vae.Input("vae"),
                io.String.Input("filename_prefix", default="vae/ComfyUI_vae"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, vae, filename_prefix) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        prompt_info = ""
        if cls.hidden.prompt is not None:
            prompt_info = json.dumps(cls.hidden.prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata["prompt"] = prompt_info
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(vae.get_sd(), output_checkpoint, metadata=metadata)
        return io.NodeOutput()

    save = execute  # TODO: remove


class ModelSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelSave",
            search_aliases=["export model", "checkpoint save"],
            category="advanced/model_merging",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("filename_prefix", default="diffusion_models/ComfyUI"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, filename_prefix) -> io.NodeOutput:
        save_checkpoint(model, filename_prefix=filename_prefix, output_dir=folder_paths.get_output_directory(), prompt=cls.hidden.prompt, extra_pnginfo=cls.hidden.extra_pnginfo)
        return io.NodeOutput()

    save = execute  # TODO: remove


class ModelMergingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ModelMergeSimple,
            ModelMergeBlocks,
            ModelSubtract,
            ModelAdd,
            CheckpointSave,
            CLIPMergeSimple,
            CLIPSubtract,
            CLIPAdd,
            CLIPSave,
            VAESave,
            ModelSave,
        ]


async def comfy_entrypoint() -> ModelMergingExtension:
    return ModelMergingExtension()
