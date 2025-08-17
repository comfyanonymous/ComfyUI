import numpy as np
import os
import json
import torch
import folder_paths
import comfy
import comfy.model_management
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from nodes import ConditioningSetMask, RepeatLatentBatch
from comfy_extras.nodes_mask import LatentCompositeMasked
from ..libs.log import log_node_info, log_node_warn
from ..libs.adv_encode import advanced_encode
from ..libs.utils import AlwaysEqualProxy
any_type = AlwaysEqualProxy("*")


class If:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "if": (any_type,),
                "else": (any_type,),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("?",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/🚫 Deprecated"
    DEPRECATED = True

    def execute(self, *args, **kwargs):
        return (kwargs['if'] if kwargs['any'] else kwargs['else'],)


class poseEditor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("STRING", {"default": ""})
        }}

    FUNCTION = "output_pose"
    CATEGORY = "EasyUse/🚫 Deprecated"
    DEPRECATED = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    def output_pose(self, image):
        return ()


class imageToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "channel": (['red', 'green', 'blue'],),
        }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    CATEGORY = "EasyUse/🚫 Deprecated"
    DEPRECATED = True

    def convert_to_single_channel(self, image, channel='red'):
        from PIL import Image
        # Convert to RGB mode to access individual channels
        image = image.convert('RGB')

        # Extract the desired channel and convert to greyscale
        if channel == 'red':
            channel_img = image.split()[0].convert('L')
        elif channel == 'green':
            channel_img = image.split()[1].convert('L')
        elif channel == 'blue':
            channel_img = image.split()[2].convert('L')
        else:
            raise ValueError(
                "Invalid channel option. Please choose 'red', 'green', or 'blue'.")

        # Convert the greyscale channel back to RGB mode
        channel_img = Image.merge(
            'RGB', (channel_img, channel_img, channel_img))

        return channel_img

    def convert(self, image, channel='red'):
        from ..libs.image import pil2tensor, tensor2pil
        image = self.convert_to_single_channel(tensor2pil(image), channel)
        image = pil2tensor(image)
        return (image.squeeze().mean(2),)

# 显示推理时间
class showSpentTime:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    FUNCTION = "notify"
    OUTPUT_NODE = True
    CATEGORY = "EasyUse/🚫 Deprecated"
    DEPRECATED = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    def notify(self, pipe, spent_time=None, unique_id=None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                spent_time = pipe['loader_settings']['spent_time'] if 'spent_time' in pipe['loader_settings'] else ''
                node["widgets_values"] = [spent_time]

        return {"ui": {"text": [spent_time]}, "result": {}}


# 潜空间sigma相乘
class latentNoisy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "steps": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 1, "max": 10000}),
            "source": (["CPU", "GPU"],),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        "optional": {
            "pipe": ("PIPE_LINE",),
            "optional_model": ("MODEL",),
            "optional_latent": ("LATENT",)
        }}

    RETURN_TYPES = ("PIPE_LINE", "LATENT", "FLOAT",)
    RETURN_NAMES = ("pipe", "latent", "sigma",)
    FUNCTION = "run"
    DEPRECATED = True

    CATEGORY = "EasyUse/🚫 Deprecated"

    def run(self, sampler_name, scheduler, steps, start_at_step, end_at_step, source, seed, pipe=None, optional_model=None, optional_latent=None):
        model = optional_model if optional_model is not None else pipe["model"]
        batch_size = pipe["loader_settings"]["batch_size"]
        empty_latent_height = pipe["loader_settings"]["empty_latent_height"]
        empty_latent_width = pipe["loader_settings"]["empty_latent_width"]

        if optional_latent is not None:
            samples = optional_latent
        else:
            torch.manual_seed(seed)
            if source == "CPU":
                device = "cpu"
            else:
                device = comfy.model_management.get_torch_device()
            noise = torch.randn((batch_size, 4, empty_latent_height // 8, empty_latent_width // 8), dtype=torch.float32,
                                device=device).cpu()

            samples = {"samples": noise}

        device = comfy.model_management.get_torch_device()
        end_at_step = min(steps, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        comfy.model_management.load_model_gpu(model)
        model_patcher = comfy.model_patcher.ModelPatcher(model.model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
        sampler = comfy.samplers.KSampler(model_patcher, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.cpu().numpy()

        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1 * sigma

        if pipe is None:
            pipe = {}
        new_pipe = {
            **pipe,
            "samples": samples_out
        }
        del pipe

        return (new_pipe, samples_out, sigma)

# Latent遮罩复合
class latentCompositeMaskedWithCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "text_combine": ("LIST",),
                "source_latent": ("LATENT",),
                "source_mask": ("MASK",),
                "destination_mask": ("MASK",),
                "text_combine_mode": (["add", "replace", "cover"], {"default": "add"}),
                "replace_text": ("STRING", {"default": ""})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    OUTPUT_IS_LIST = (False, False, True)

    RETURN_TYPES = ("PIPE_LINE", "LATENT", "CONDITIONING")
    RETURN_NAMES = ("pipe", "latent", "conditioning",)
    FUNCTION = "run"

    CATEGORY = "EasyUse/🚫 Deprecated"
    DEPRECATED = True

    def run(self, pipe, text_combine, source_latent, source_mask, destination_mask, text_combine_mode, replace_text, prompt=None, extra_pnginfo=None, my_unique_id=None):
        positive = None
        clip = pipe["clip"]
        destination_latent = pipe["samples"]

        conds = []

        for text in text_combine:
            if text_combine_mode == 'cover':
                positive = text
            elif text_combine_mode == 'replace' and replace_text != '':
                positive = pipe["loader_settings"]["positive"].replace(replace_text, text)
            else:
                positive = pipe["loader_settings"]["positive"] + ',' + text
            positive_token_normalization = pipe["loader_settings"]["positive_token_normalization"]
            positive_weight_interpretation = pipe["loader_settings"]["positive_weight_interpretation"]
            a1111_prompt_style = pipe["loader_settings"]["a1111_prompt_style"]
            positive_cond = pipe["positive"]

            log_node_warn("Positive encoding...")
            steps = pipe["loader_settings"]["steps"] if "steps" in pipe["loader_settings"] else 1
            positive_embeddings_final = advanced_encode(clip, positive,
                                         positive_token_normalization,
                                         positive_weight_interpretation, w_max=1.0,
                                         apply_to_pooled='enable', a1111_prompt_style=a1111_prompt_style, steps=steps)

            # source cond
            (cond_1,) = ConditioningSetMask().append(positive_cond, source_mask, "default", 1)
            (cond_2,) = ConditioningSetMask().append(positive_embeddings_final, destination_mask, "default", 1)
            positive_cond = cond_1 + cond_2

            conds.append(positive_cond)
        # latent composite masked
        (samples,) = LatentCompositeMasked().composite(destination_latent, source_latent, 0, 0, False)

        new_pipe = {
            **pipe,
            "samples": samples,
            "loader_settings": {
                **pipe["loader_settings"],
                "positive": positive,
            }
        }

        del pipe

        return (new_pipe, samples, conds)

# 噪声注入到潜空间
class injectNoiseToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 200.0, "step": 0.0001}),
            "normalize": ("BOOLEAN", {"default": False}),
            "average": ("BOOLEAN", {"default": False}),
        },
            "optional": {
                "pipe_to_noise": ("PIPE_LINE",),
                "image_to_latent": ("IMAGE",),
                "latent": ("LATENT",),
                "noise": ("LATENT",),
                "mask": ("MASK",),
                "mix_randn_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject"
    CATEGORY = "EasyUse/🚫 Deprecated"
    DEPRECATED = True


    def inject(self,strength, normalize, average, pipe_to_noise=None, noise=None, image_to_latent=None, latent=None, mix_randn_amount=0, mask=None, seed=None):

        vae = pipe_to_noise["vae"] if pipe_to_noise is not None else pipe_to_noise["vae"]
        batch_size = pipe_to_noise["loader_settings"]["batch_size"] if pipe_to_noise is not None and "batch_size" in pipe_to_noise["loader_settings"] else 1
        if noise is None and pipe_to_noise is not None:
            noise = pipe_to_noise["samples"]
        elif noise is None:
            raise Exception("InjectNoiseToLatent: No noise provided")

        if image_to_latent is not None and vae is not None:
            samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
            latents = RepeatLatentBatch().repeat(samples, batch_size)[0]
        elif latent is not None:
            latents = latent
        else:
            latents = {"samples": noise["samples"].clone()}

        samples = latents.copy()
        if latents["samples"].shape != noise["samples"].shape:
            raise ValueError("InjectNoiseToLatent: Latent and noise must have the same shape")
        if average:
            noised = (samples["samples"].clone() + noise["samples"].clone()) / 2
        else:
            noised = samples["samples"].clone() + noise["samples"].clone() * strength
        if normalize:
            noised = noised / noised.std()
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                                   size=(noised.shape[2], noised.shape[3]), mode="bilinear")
            mask = mask.expand((-1, noised.shape[1], -1, -1))
            if mask.shape[0] < noised.shape[0]:
                mask = mask.repeat((noised.shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:noised.shape[0]]
            noised = mask * noised + (1 - mask) * latents["samples"]
        if mix_randn_amount > 0:
            if seed is not None:
                torch.manual_seed(seed)
            rand_noise = torch.randn_like(noised)
            noised = ((1 - mix_randn_amount) * noised + mix_randn_amount *
                      rand_noise) / ((mix_randn_amount ** 2 + (1 - mix_randn_amount) ** 2) ** 0.5)
        samples["samples"] = noised
        return (samples,)


from ..libs.api.stability import stableAPI
class stableDiffusion3API:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),
                "model": (["sd3", "sd3-turbo"],),
                "aspect_ratio": (['16:9', '1:1', '21:9', '2:3', '3:2', '4:5', '5:4', '9:16', '9:21'],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "optional_image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "generate"
    OUTPUT_NODE = False

    CATEGORY = "EasyUse/🚫 Deprecated"
    DEPRECATED = True


    def generate(self, positive, negative, model, aspect_ratio, seed, denoise, optional_image=None, unique_id=None, extra_pnginfo=None):
        stableAPI.getAPIKeys()
        mode = 'text-to-image'
        if optional_image is not None:
            mode = 'image-to-image'
        output_image = stableAPI.generate_sd3_image(positive, negative, aspect_ratio, seed=seed, mode=mode, model=model, strength=denoise, image=optional_image)
        return (output_image,)


class saveImageLazy():
  def __init__(self):
    self.output_dir = folder_paths.get_output_directory()
    self.type = "output"
    self.compress_level = 4

  @classmethod
  def INPUT_TYPES(s):
    return {"required":
          {"images": ("IMAGE",),
           "filename_prefix": ("STRING", {"default": "ComfyUI"}),
           "save_metadata": ("BOOLEAN", {"default": True}),
           },
        "optional":{},
        "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
      }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("images",)
  OUTPUT_NODE = False
  FUNCTION = "save"

  DEPRECATED = True
  CATEGORY = "EasyUse/🚫 Deprecated"

  def save(self, images, filename_prefix, save_metadata, prompt=None, extra_pnginfo=None):
    extension = 'png'

    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
      filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

    results = list()
    for (batch_number, image) in enumerate(images):
      i = 255. * image.cpu().numpy()
      img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
      metadata = None

      filename_with_batch_num = filename.replace(
        "%batch_num%", str(batch_number))

      counter = 1

      if os.path.exists(full_output_folder) and os.listdir(full_output_folder):
        filtered_filenames = list(filter(
          lambda filename: filename.startswith(
            filename_with_batch_num + "_")
                           and filename[len(filename_with_batch_num) + 1:-4].isdigit(),
          os.listdir(full_output_folder)
        ))

        if filtered_filenames:
          max_counter = max(
            int(filename[len(filename_with_batch_num) + 1:-4])
            for filename in filtered_filenames
          )
          counter = max_counter + 1

      file = f"{filename_with_batch_num}_{counter:05}.{extension}"

      save_path = os.path.join(full_output_folder, file)

      if save_metadata:
        metadata = PngInfo()
        if prompt is not None:
          metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
          for x in extra_pnginfo:
            metadata.add_text(
              x, json.dumps(extra_pnginfo[x]))

      img.save(save_path, pnginfo=metadata)

      results.append({
        "filename": file,
        "subfolder": subfolder,
        "type": self.type
      })

    return {"ui": {"images": results} , "result": (images,)}

from .logic import saveText, showAnything

class showAnythingLazy(showAnything):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"anything": (any_type, {}), },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                           }}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('output',)
    INPUT_IS_LIST = True
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = (False,)
    DEPRECATED = True
    FUNCTION = "log_input"
    CATEGORY = "EasyUse/🚫 Deprecated"

class saveTextLazy(saveText):

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", 'image',)

    FUNCTION = "save_text"
    OUTPUT_NODE = False
    DEPRECATED = True
    CATEGORY = "EasyUse/🚫 Deprecated"

NODE_CLASS_MAPPINGS = {
    "easy if": If,
    "easy poseEditor": poseEditor,
    "easy imageToMask": imageToMask,
    "easy showSpentTime": showSpentTime,
    "easy latentNoisy": latentNoisy,
    "easy latentCompositeMaskedWithCond": latentCompositeMaskedWithCond,
    "easy injectNoiseToLatent": injectNoiseToLatent,
    "easy stableDiffusion3API": stableDiffusion3API,
    "easy saveImageLazy": saveImageLazy,
    "easy saveTextLazy": saveTextLazy,
    "easy showAnythingLazy": showAnythingLazy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy if": "If (🚫Deprecated)",
    "easy poseEditor": "PoseEditor (🚫Deprecated)",
    "easy imageToMask": "ImageToMask (🚫Deprecated)",
    "easy showSpentTime": "Show Spent Time (🚫Deprecated)",
    "easy latentNoisy": "LatentNoisy (🚫Deprecated)",
    "easy latentCompositeMaskedWithCond": "LatentCompositeMaskedWithCond (🚫Deprecated)",
    "easy injectNoiseToLatent": "InjectNoiseToLatent (🚫Deprecated)",
    "easy stableDiffusion3API": "StableDiffusion3API (🚫Deprecated)",
    "easy saveImageLazy": "SaveImageLazy (🚫Deprecated)",
    "easy saveTextLazy": "SaveTextLazy (🚫Deprecated)",
    "easy showAnythingLazy": "ShowAnythingLazy (🚫Deprecated)",
}