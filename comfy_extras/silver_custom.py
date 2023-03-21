import datetime

import torch

import os
import sys
import json
import hashlib
import copy
import traceback

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy_extras.clip_vision
import model_management
import importlib
import folder_paths


class Note:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ()
    FUNCTION = "Note"

    OUTPUT_NODE = False

    CATEGORY = "silver_custom"


class SaveImageList:
    def __init__(self):
        current_dir = os.path.abspath(os.getcwd())
        print(current_dir)
        self.output_dir = os.path.join(current_dir, "output")
        print(self.output_dir)
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images_list"

    OUTPUT_NODE = True

    CATEGORY = "silver_custom"

    def save_images_list(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(self.output_dir, subfolder)

        if os.path.commonpath((self.output_dir, os.path.realpath(full_output_folder))) != self.output_dir:
            print("Saving image outside the output folder is not allowed.")
            return {}

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                                 map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file = f"{filename}-{now}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, optimize=True)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return self.get_all_files()

    def get_all_files(self):
        results = []
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                subfolder = os.path.relpath(root, self.output_dir)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
        sorted_results = sorted(results, key=lambda x: x["filename"])
        return {"ui": {"images": sorted_results}}


def custom_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                    disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, in_seed=None):
    latent_image = latent["samples"]
    noise_mask = None
    device = model_management.get_torch_device()
    if in_seed is not None:
        seed = in_seed
    print(seed)
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                            generator=torch.manual_seed(seed), device="cpu")

    if "noise_mask" in latent:
        noise_mask = latent['noise_mask']
        noise_mask = torch.nn.functional.interpolate(noise_mask[None, None,], size=(noise.shape[2], noise.shape[3]),
                                                     mode="bilinear")
        noise_mask = noise_mask.round()
        noise_mask = torch.cat([noise_mask] * noise.shape[1], dim=1)
        noise_mask = torch.cat([noise_mask] * noise.shape[0])
        noise_mask = noise_mask.to(device)

    real_model = None
    model_management.load_model_gpu(model)
    real_model = model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = []
    negative_copy = []

    control_nets = []
    for p in positive:
        t = p[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        t = t.to(device)
        if 'control' in p[1]:
            control_nets += [p[1]['control']]
        positive_copy += [[t] + p[1:]]
    for n in negative:
        t = n[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        t = t.to(device)
        if 'control' in n[1]:
            control_nets += [n[1]['control']]
        negative_copy += [[t] + n[1:]]

    control_net_models = []
    for x in control_nets:
        control_net_models += x.get_control_models()
    model_management.load_controlnet_gpu(control_net_models)

    if sampler_name in comfy.samplers.KSampler.SAMPLERS:
        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=denoise)
    else:
        # other samplers
        pass

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask)
    samples = samples.cpu()
    for c in control_nets:
        c.cleanup()

    out = latent.copy()
    out["samples"] = samples
    return (out, seed,)


class CustomKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
                {
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent_image": ("LATENT",),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
            "optional":
                {
                    "in_seed": ()
                }
        }

    RETURN_TYPES = ("LATENT", "seed",)
    FUNCTION = "sample"

    CATEGORY = "silver_custom"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
               in_seed=None):
        return custom_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise, in_seed=in_seed)


NODE_CLASS_MAPPINGS = {
    "Note": Note,
    "SaveImageList": SaveImageList,
    "CustomKSampler": CustomKSampler,
}
