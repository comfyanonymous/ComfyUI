import torch

import os
import sys
import json
import hashlib
import copy

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

sys.path.append(os.path.join(sys.path[0], "comfy"))


import comfy.samplers
import comfy.sd

supported_ckpt_extensions = ['.ckpt']
supported_pt_extensions = ['.ckpt', '.pt']
try:
    import safetensors.torch
    supported_ckpt_extensions += ['.safetensors']
    supported_pt_extensions += ['.safetensors']
except:
    print("Could not import safetensors, safetensors support disabled.")

def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions, files)))

class CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text):
        return ([[clip.encode(text), {}]], )

class ConditioningCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_1": ("CONDITIONING", ), "conditioning_2": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"

    CATEGORY = "conditioning"

    def combine(self, conditioning_1, conditioning_2):
        return (conditioning_1 + conditioning_2, )

class ConditioningSetArea:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("INT", {"default": 64, "min": 64, "max": 4096, "step": 64}),
                              "height": ("INT", {"default": 64, "min": 64, "max": 4096, "step": 64}),
                              "x": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                              "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength, min_sigma=0.0, max_sigma=99.0):
        c = copy.deepcopy(conditioning)
        for t in c:
            t[1]['area'] = (height // 8, width // 8, y // 8, x // 8)
            t[1]['strength'] = strength
            t[1]['min_sigma'] = min_sigma
            t[1]['max_sigma'] = max_sigma
        return (c, )

class VAEDecode:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, samples):
        return (vae.decode(samples), )

class VAEEncode:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent"

    def encode(self, vae, pixels):
        x = (pixels.shape[1] // 64) * 64
        y = (pixels.shape[2] // 64) * 64
        if pixels.shape[1] != x or pixels.shape[2] != y:
            pixels = pixels[:,:x,:y,:]
        return (vae.encode(pixels), )

class CheckpointLoader:
    models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    config_dir = os.path.join(models_dir, "configs")
    ckpt_dir = os.path.join(models_dir, "checkpoints")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "config_name": (filter_files_extensions(os.listdir(s.config_dir), '.yaml'), ),
                              "ckpt_name": (filter_files_extensions(os.listdir(s.ckpt_dir), supported_ckpt_extensions), )}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        config_path = os.path.join(self.config_dir, config_name)
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        embedding_directory = os.path.join(self.models_dir, "embeddings")
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=embedding_directory)

class LoraLoader:
    models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    lora_dir = os.path.join(models_dir, "loras")
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (filter_files_extensions(os.listdir(s.lora_dir), supported_pt_extensions), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = os.path.join(self.lora_dir, lora_name)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_path, strength_model, strength_clip)
        return (model_lora, clip_lora)

class VAELoader:
    models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    vae_dir = os.path.join(models_dir, "vae")
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (filter_files_extensions(os.listdir(s.vae_dir), supported_pt_extensions), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "loaders"

    #TODO: scale factor?
    def load_vae(self, vae_name):
        vae_path = os.path.join(self.vae_dir, vae_name)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        return (vae,)

class EmptyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                              "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (latent, )

class LatentUpscale:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                              "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, samples, upscale_method, width, height, crop):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:,:,y:old_height-y,x:old_width-x]
        else:
            s = samples
        s = torch.nn.functional.interpolate(s, size=(height // 8, width // 8), mode=upscale_method)
        return (s,)

class LatentRotate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "rotation": (["none", "90 degrees", "180 degrees", "270 degrees"],),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "rotate"

    CATEGORY = "latent"

    def rotate(self, samples, rotation):
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        s = torch.rot90(samples, k=rotate_by, dims=[3, 2])
        return (s,)

class LatentFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "flip_method": (["x-axis: vertically", "y-axis: horizontally"],),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "flip"

    CATEGORY = "latent"

    def flip(self, samples, flip_method):
        if flip_method.startswith("x"):
            s = torch.flip(samples, dims=[2])
        elif flip_method.startswith("y"):
            s = torch.flip(samples, dims=[3])
        else:
            s = samples

        return (s,)

class LatentComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                              "x": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"

    CATEGORY = "latent"

    def composite(self, samples_to, samples_from, x, y, composite_method="normal"):
        x =  x // 8
        y = y // 8
        s = samples_to.clone()
        s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        return (s,)

def common_ksampler(device, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=torch.manual_seed(seed), device="cpu")

    try:
        real_model = model.patch_model()
        real_model.to(device)
        noise = noise.to(device)
        latent_image = latent_image.to(device)

        positive_copy = []
        negative_copy = []

        for p in positive:
            t = p[0]
            if t.shape[0] < noise.shape[0]:
                t = torch.cat([t] * noise.shape[0])
            t = t.to(device)
            positive_copy += [[t] + p[1:]]
        for n in negative:
            t = n[0]
            if t.shape[0] < noise.shape[0]:
                t = torch.cat([t] * noise.shape[0])
            t = t.to(device)
            negative_copy += [[t] + n[1:]]

        if sampler_name in comfy.samplers.KSampler.SAMPLERS:
            sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise)
        else:
            #other samplers
            pass

        samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise)
        samples = samples.cpu()
        real_model.cpu()
        model.unpatch_model()
    except Exception as e:
        real_model.cpu()
        model.unpatch_model()
        raise e

    return (samples, )

class KSampler:
    def __init__(self, device="cuda"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(self.device, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

class KSamplerAdvanced:
    def __init__(self, device="cuda"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler(self.device, model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

class SaveImage:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        try:
            counter = max(filter(lambda a: a[1][:-1] == filename_prefix and a[1][-1] == "_", map(map_filename, os.listdir(self.output_dir))))[0] + 1
        except ValueError:
            counter = 1
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(i.astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            img.save(f"output/{filename_prefix}_{counter:05}_.png", pnginfo=metadata, optimize=True)
            counter += 1

class LoadImage:
    input_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": (os.listdir(s.input_dir), )},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = os.path.join(self.input_dir, image)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image[None])[None,]
        return image

    @classmethod
    def IS_CHANGED(s, image):
        image_path = os.path.join(s.input_dir, image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()



NODE_CLASS_MAPPINGS = {
    "KSampler": KSampler,
    "CheckpointLoader": CheckpointLoader,
    "CLIPTextEncode": CLIPTextEncode,
    "VAEDecode": VAEDecode,
    "VAEEncode": VAEEncode,
    "VAELoader": VAELoader,
    "EmptyLatentImage": EmptyLatentImage,
    "LatentUpscale": LatentUpscale,
    "SaveImage": SaveImage,
    "LoadImage": LoadImage,
    "ConditioningCombine": ConditioningCombine,
    "ConditioningSetArea": ConditioningSetArea,
    "KSamplerAdvanced": KSamplerAdvanced,
    "LatentComposite": LatentComposite,
    "LatentRotate": LatentRotate,
    "LatentFlip": LatentFlip,
    "LoraLoader": LoraLoader,
}


