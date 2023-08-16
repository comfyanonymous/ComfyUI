import gc
import json
import os.path
import typing

import torch
import torchvision.transforms.functional as TF
from diffusers import DiffusionPipeline, IFPipeline, StableDiffusionUpscalePipeline, IFSuperResolutionPipeline
from diffusers.utils import is_accelerate_available, is_accelerate_version
from transformers import T5EncoderModel

from comfy.model_management import throw_exception_if_processing_interrupted, get_torch_device, cpu_state, CPUState
# todo: this relies on the setup-py cleanup fork
from comfy.utils import ProgressBar, get_project_root

# todo: find or download the models automatically by their config jsons instead of using well known names
_model_base_path = os.path.join(get_project_root(), "models", "deepfloyd")


def _find_files(directory: str, filename: str) -> typing.List[str]:
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file == filename]


# todo: ticket diffusers to correctly deal with an omitted unet
def _patched_enable_model_cpu_offload_ifpipeline(self: IFPipeline | IFSuperResolutionPipeline, gpu_id=0):
    r"""
    Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
    to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
    method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
    `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
    """
    if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
        from accelerate import cpu_offload_with_hook
    else:
        raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

    if cpu_state == CPUState.GPU:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = get_torch_device()

    if cpu_state == CPUState.CPU or cpu_state == CPUState.MPS:
        return

    if self.device.type != "cpu":
        self.to("cpu", silence_dtype_warnings=True)
        torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

    hook = None

    if self.text_encoder is not None:
        _, hook = cpu_offload_with_hook(self.text_encoder, device, prev_module_hook=hook)

        # Accelerate will move the next model to the device _before_ calling the offload hook of the
        # previous model. This will cause both models to be present on the device at the same time.
        # IF uses T5 for its text encoder which is really large. We can manually call the offload
        # hook for the text encoder to ensure it's moved to the cpu before the unet is moved to
        # the GPU.
        self.text_encoder_offload_hook = hook

    # todo: patch here
    if self.unet is not None:
        _, hook = cpu_offload_with_hook(self.unet, device, prev_module_hook=hook)

        # if the safety checker isn't called, `unet_offload_hook` will have to be called to manually offload the unet
        self.unet_offload_hook = hook

    if self.safety_checker is not None:
        _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

    # We'll offload the last model manually.
    self.final_offload_hook = hook


def _cpu_offload(self: DiffusionPipeline, gpu_id=0):
    # todo: use sequential for low vram, ordinary cpu offload for normal vram
    if isinstance(self, IFPipeline) or isinstance(self, IFSuperResolutionPipeline):
        _patched_enable_model_cpu_offload_ifpipeline(self, gpu_id)
    # todo: include sequential usage
    # elif isinstance(self, StableDiffusionUpscalePipeline):
    #     self.enable_sequential_cpu_offload(gpu_id)
    elif hasattr(self, 'enable_model_cpu_offload'):
        self.enable_model_cpu_offload(gpu_id)


class Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (Loader._MODELS, {"default": "I-M"}),
                "load_in_8bit": ([False, True], {"default": False}),
                "device": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "deepfloyd"
    FUNCTION = "process"
    RETURN_TYPES = ("IF_MODEL",)

    _MODELS = ["I-M", "I-L", "I-XL", "II-M", "II-L", "III", "t5"]

    # todo: correctly use load_in_8bit
    def process(self, model_name: str, load_in_8bit: bool, device: str):
        assert model_name in Loader._MODELS

        model_v: DiffusionPipeline
        model_path: str
        kwargs = {
            "variant": "fp16",
            "torch_dtype": torch.float16,
            "requires_safety_checker": False,
            "feature_extractor": None,
            "safety_checker": None,
            "watermarker": None,
            "load_in_8bit": load_in_8bit,
            # todo: fix diffusers when using device_map auto on multi-gpu setups, layers are not assigned to different devices correctly
            "device_map": None if device else "auto"
        }

        if model_name == "t5":
            # find any valid IF model
            model_path = next(os.path.dirname(file) for file in _find_files(_model_base_path, "model_index.json") if
                              any(x == T5EncoderModel.__name__ for x in
                                  json.load(open(file, 'r'))["text_encoder"]))
            # todo: this must use load_in_8bit correctly
            # kwargs["text_encoder"] = text_encoder
            kwargs["unet"] = None
        elif model_name == "III":
            model_path = f"{_model_base_path}/stable-diffusion-x4-upscaler"
            del kwargs["variant"]
        else:
            model_path = f"{_model_base_path}/IF-{model_name}-v1.0"
            kwargs["text_encoder"] = None

        model_v = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            **kwargs
        )

        if device:
            model_v = model_v.to(device)

        _cpu_offload(model_v, gpu_id=model_v.device.index)

        return (model_v,)


class Encoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IF_MODEL",),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "deepfloyd"
    FUNCTION = "process"
    MODEL = None
    RETURN_TYPES = ("POSITIVE", "NEGATIVE",)
    TEXT_ENCODER = None

    def process(self, model: IFPipeline, positive, negative):
        positive, negative = model.encode_prompt(
            prompt=positive,
            negative_prompt=negative,
        )

        return (positive, negative,)


class StageI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("POSITIVE",),
                "negative": ("NEGATIVE",),
                "model": ("IF_MODEL",),
                "width": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "height": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})
            },
        }

    CATEGORY = "deepfloyd"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, model: IFPipeline, positive, negative, width, height, batch_size, seed, steps, cfg):
        progress = ProgressBar(steps)

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        gc.collect()
        image = model(
            prompt_embeds=positive,
            negative_prompt_embeds=negative,
            width=width,
            height=height,
            generator=torch.manual_seed(seed),
            guidance_scale=cfg,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            callback=callback,
            output_type="pt",
        ).images

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().permute(0, 2, 3, 1)
        return (image,)


class StageII:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("POSITIVE",),
                "negative": ("NEGATIVE",),
                "model": ("IF_MODEL",),
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            },
        }

    CATEGORY = "deepfloyd"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES",)
    RETURN_TYPES = ("IMAGE",)

    def process(self, model, images, positive, negative, seed, steps, cfg):
        images = images.permute(0, 3, 1, 2)
        progress = ProgressBar(steps)
        batch_size, channels, height, width = images.shape
        max_dim = max(height, width)
        images = TF.center_crop(images, max_dim)
        model.unet.config.sample_size = max_dim * 4

        if batch_size > 1:
            positive = positive.repeat(batch_size, 1, 1)
            negative = negative.repeat(batch_size, 1, 1)

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        images = model(
            image=images,
            prompt_embeds=positive,
            negative_prompt_embeds=negative,
            generator=torch.manual_seed(seed),
            guidance_scale=cfg,
            num_inference_steps=steps,
            callback=callback,
            output_type="pt",
        ).images.cpu().float()

        images = TF.center_crop(images, [height * 4, width * 4])
        images = images.permute(0, 2, 3, 1)
        return (images,)


class StageIII:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IF_MODEL",),
                "image": ("IMAGE",),
                "tile": ([False, True], {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
                "noise": ("INT", {"default": 20, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "deepfloyd"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, model: StableDiffusionUpscalePipeline, image, tile, tile_size, noise, seed, steps, cfg, positive,
                negative):
        image = image.permute(0, 3, 1, 2)
        progress = ProgressBar(steps)
        batch_size = image.shape[0]

        if batch_size > 1:
            positive = [positive] * batch_size
            negative = [negative] * batch_size

        if tile:
            model.vae.config.sample_size = tile_size
            model.vae.enable_tiling()

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        image = model(
            image=image,
            prompt=positive,
            negative_prompt=negative,
            noise_level=noise,
            generator=torch.manual_seed(seed),
            guidance_scale=cfg,
            num_inference_steps=steps,
            callback=callback,
            output_type="pt",
        ).images.cpu().float().permute(0, 2, 3, 1)

        return (image,)
