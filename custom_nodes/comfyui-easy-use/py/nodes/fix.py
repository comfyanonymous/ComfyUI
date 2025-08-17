import sys
import time
import comfy
import torch
import folder_paths

from comfy_extras.chainner_models import model_loading

from server import PromptServer
from nodes import MAX_RESOLUTION, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

from ..libs.utils import easySave, get_sd_version
from ..libs.sampler import easySampler
from .. import easyCache, sampler

class hiresFix:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                 "model_name": (folder_paths.get_filename_list("upscale_models"),),
                 "rescale_after_model": ([False, True], {"default": True}),
                 "rescale_method": (s.upscale_methods,),
                 "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect'],),
                 "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                 "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "crop": (s.crop_methods,),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                },
                "optional": {
                    "pipe": ("PIPE_LINE",),
                    "image": ("IMAGE",),
                    "vae": ("VAE",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                           },
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "LATENT", )
    RETURN_NAMES = ('pipe', 'image', "latent", )

    FUNCTION = "upscale"
    CATEGORY = "EasyUse/Fix"
    OUTPUT_NODE = True

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def upscale(self, model_name, rescale_after_model, rescale_method, rescale, percent, width, height,
                longer_side, crop, image_output, link_id, save_prefix, pipe=None, image=None, vae=None, prompt=None,
                extra_pnginfo=None, my_unique_id=None):

        new_pipe = {}
        if pipe is not None:
            image = image if image is not None else pipe["images"]
            vae = vae if vae is not None else pipe.get("vae")
        elif image is None or vae is None:
            raise ValueError("pipe or image or vae missing.")
        # Load Model
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        upscale_model = model_loading.load_state_dict(sd).eval()

        # Model upscale
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 128 + 64
        overlap = 8
        steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile,
                                                                    tile_y=tile, overlap=overlap)
        pbar = comfy.utils.ProgressBar(steps)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap,
                                    upscale_amount=upscale_model.scale, pbar=pbar)
        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

        # Post Model Rescale
        if rescale_after_model == True:
            samples = s.movedim(-1, 1)
            orig_height = samples.shape[2]
            orig_width = samples.shape[3]
            if rescale == "by percentage" and percent != 0:
                height = percent / 100 * orig_height
                width = percent / 100 * orig_width
                if (width > MAX_RESOLUTION):
                    width = MAX_RESOLUTION
                if (height > MAX_RESOLUTION):
                    height = MAX_RESOLUTION

                width = easySampler.enforce_mul_of_64(width)
                height = easySampler.enforce_mul_of_64(height)
            elif rescale == "to longer side - maintain aspect":
                longer_side = easySampler.enforce_mul_of_64(longer_side)
                if orig_width > orig_height:
                    width, height = longer_side, easySampler.enforce_mul_of_64(longer_side * orig_height / orig_width)
                else:
                    width, height = easySampler.enforce_mul_of_64(longer_side * orig_width / orig_height), longer_side

            s = comfy.utils.common_upscale(samples, width, height, rescale_method, crop)
            s = s.movedim(1, -1)

        # vae encode
        pixels = self.vae_encode_crop_pixels(s)
        t = vae.encode(pixels[:, :, :, :3])

        if pipe is not None:
            new_pipe = {
                "model": pipe['model'],
                "positive": pipe['positive'],
                "negative": pipe['negative'],
                "vae": vae,
                "clip": pipe['clip'],

                "samples": {"samples": t},
                "images": s,
                "seed": pipe['seed'],

                "loader_settings": {
                    **pipe["loader_settings"],
                }
            }
            del pipe
        else:
            new_pipe = {}

        results = easySave(s, save_prefix, image_output, prompt, extra_pnginfo)

        if image_output in ("Sender", "Sender&Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        if image_output in ("Hide", "Hide&Save"):
            return (new_pipe, s, {"samples": t},)

        return {"ui": {"images": results},
                "result": (new_pipe, s, {"samples": t},)}

# 预细节修复
class preDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "pipe": ("PIPE_LINE",),
             "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
             "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['align_your_steps'],),
             "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
             "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
             "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
             "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
             "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
        },
            "optional": {
                "bbox_segm_pipe": ("PIPE_LINE",),
                "sam_pipe": ("PIPE_LINE",),
                "optional_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, pipe, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint, drop_size, wildcard, cycle, bbox_segm_pipe=None, sam_pipe=None, optional_image=None):

        model = pipe["model"] if "model" in pipe else None
        if model is None:
            raise Exception(f"[ERROR] pipe['model'] is missing")
        clip = pipe["clip"] if"clip" in pipe else None
        if clip is None:
            raise Exception(f"[ERROR] pipe['clip'] is missing")
        vae = pipe["vae"] if "vae" in pipe else None
        if vae is None:
            raise Exception(f"[ERROR] pipe['vae'] is missing")
        if optional_image is not None:
            images = optional_image
        else:
            images = pipe["images"] if "images" in pipe else None
            if images is None:
                raise Exception(f"[ERROR] pipe['image'] is missing")
        positive = pipe["positive"] if "positive" in pipe else None
        if positive is None:
            raise Exception(f"[ERROR] pipe['positive'] is missing")
        negative = pipe["negative"] if "negative" in pipe else None
        if negative is None:
            raise Exception(f"[ERROR] pipe['negative'] is missing")
        bbox_segm_pipe = bbox_segm_pipe or (pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None)
        if bbox_segm_pipe is None:
            raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
        sam_pipe = sam_pipe or (pipe["sam_pipe"] if pipe and "sam_pipe" in pipe else None)
        if sam_pipe is None:
            raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")

        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}

        if(scheduler == 'align_your_steps'):
            model_version = get_sd_version(model)
            if model_version == 'sdxl':
                scheduler = 'AYS SDXL'
            elif model_version == 'svd':
                scheduler = 'AYS SVD'
            else:
                scheduler = 'AYS SD1'

        new_pipe = {
            "images": images,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": positive,
            "negative": negative,
            "seed": seed,

            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": loader_settings,

            "detail_fix_settings": {
                "guide_size": guide_size,
                "guide_size_for": guide_size_for,
                "max_size": max_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "noise_mask": noise_mask,
                "force_inpaint": force_inpaint,
                "drop_size": drop_size,
                "wildcard": wildcard,
                "cycle": cycle
            }
        }


        del bbox_segm_pipe
        del sam_pipe

        return (new_pipe,)

# 预遮罩细节修复
class preMaskDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "pipe": ("PIPE_LINE",),
             "mask": ("MASK",),

             "guide_size": ("FLOAT", {"default": 384, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
             "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "mask_mode": ("BOOLEAN", {"default": True, "label_on": "masked only", "label_off": "whole"}),

             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
             "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),

             "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
             "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
             "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
             "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
             "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
             "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
        },
            "optional": {
                # "patch": ("INPAINT_PATCH",),
                "optional_image": ("IMAGE",),
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, pipe, mask, guide_size, guide_size_for, max_size, mask_mode, seed, steps, cfg, sampler_name, scheduler, denoise, feather, crop_factor, drop_size,refiner_ratio, batch_size, cycle, optional_image=None, inpaint_model=False, noise_mask_feather=20):

        model = pipe["model"] if "model" in pipe else None
        if model is None:
            raise Exception(f"[ERROR] pipe['model'] is missing")
        clip = pipe["clip"] if"clip" in pipe else None
        if clip is None:
            raise Exception(f"[ERROR] pipe['clip'] is missing")
        vae = pipe["vae"] if "vae" in pipe else None
        if vae is None:
            raise Exception(f"[ERROR] pipe['vae'] is missing")
        if optional_image is not None:
            images = optional_image
        else:
            images = pipe["images"] if "images" in pipe else None
            if images is None:
                raise Exception(f"[ERROR] pipe['image'] is missing")
        positive = pipe["positive"] if "positive" in pipe else None
        if positive is None:
            raise Exception(f"[ERROR] pipe['positive'] is missing")
        negative = pipe["negative"] if "negative" in pipe else None
        if negative is None:
            raise Exception(f"[ERROR] pipe['negative'] is missing")
        latent = pipe["samples"] if "samples" in pipe else None
        if latent is None:
            raise Exception(f"[ERROR] pipe['samples'] is missing")

        if 'noise_mask' not in latent:
            if images is None:
                raise Exception("No Images found")
            if vae is None:
                raise Exception("No VAE found")
            x = (images.shape[1] // 8) * 8
            y = (images.shape[2] // 8) * 8
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                                   size=(images.shape[1], images.shape[2]), mode="bilinear")

            pixels = images.clone()
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

            mask_erosion = mask

            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] -= 0.5
                pixels[:, :, :, i] *= m
                pixels[:, :, :, i] += 0.5
            t = vae.encode(pixels)

            latent = {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}
        # when patch was linked
        # if patch is not None:
        #     worker = InpaintWorker(node_name="easy kSamplerInpainting")
        #     model, = worker.patch(model, latent, patch)

        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}

        new_pipe = {
            "images": images,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "mask": mask,

            "loader_settings": loader_settings,

            "detail_fix_settings": {
                "guide_size": guide_size,
                "guide_size_for": guide_size_for,
                "max_size": max_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "crop_factor": crop_factor,
                "drop_size": drop_size,
                "refiner_ratio": refiner_ratio,
                "batch_size": batch_size,
                "cycle": cycle
            },

            "mask_settings": {
                "mask_mode": mask_mode,
                "inpaint_model": inpaint_model,
                "noise_mask_feather": noise_mask_feather
            }
        }

        del pipe

        return (new_pipe,)

# 细节修复
class detailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipe": ("PIPE_LINE",),
            "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
            "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
            "save_prefix": ("STRING", {"default": "ComfyUI"}),
        },
            "optional": {
                "model": ("MODEL",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID", }
        }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pipe", "image", "cropped_refined", "cropped_enhanced_alpha")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"


    def doit(self, pipe, image_output, link_id, save_prefix, model=None, prompt=None, extra_pnginfo=None, my_unique_id=None):

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        model = model or (pipe["model"] if "model" in pipe else None)
        if model is None:
            raise Exception(f"[ERROR] model or pipe['model'] is missing")

        detail_fix_settings = pipe["detail_fix_settings"] if "detail_fix_settings" in pipe else None
        if detail_fix_settings is None:
            raise Exception(f"[ERROR] detail_fix_settings or pipe['detail_fix_settings'] is missing")

        mask = pipe["mask"] if "mask" in pipe else None

        image = pipe["images"]
        clip = pipe["clip"]
        vae = pipe["vae"]
        seed = pipe["seed"]
        positive = pipe["positive"]
        negative = pipe["negative"]
        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}
        guide_size = pipe["detail_fix_settings"]["guide_size"] if "guide_size" in pipe["detail_fix_settings"] else 256
        guide_size_for = pipe["detail_fix_settings"]["guide_size_for"] if "guide_size_for" in pipe[
            "detail_fix_settings"] else True
        max_size = pipe["detail_fix_settings"]["max_size"] if "max_size" in pipe["detail_fix_settings"] else 768
        steps = pipe["detail_fix_settings"]["steps"] if "steps" in pipe["detail_fix_settings"] else 20
        cfg = pipe["detail_fix_settings"]["cfg"] if "cfg" in pipe["detail_fix_settings"] else 1.0
        sampler_name = pipe["detail_fix_settings"]["sampler_name"] if "sampler_name" in pipe[
            "detail_fix_settings"] else None
        scheduler = pipe["detail_fix_settings"]["scheduler"] if "scheduler" in pipe["detail_fix_settings"] else None
        denoise = pipe["detail_fix_settings"]["denoise"] if "denoise" in pipe["detail_fix_settings"] else 0.5
        feather = pipe["detail_fix_settings"]["feather"] if "feather" in pipe["detail_fix_settings"] else 5
        crop_factor = pipe["detail_fix_settings"]["crop_factor"] if "crop_factor" in pipe["detail_fix_settings"] else 3.0
        drop_size = pipe["detail_fix_settings"]["drop_size"] if "drop_size" in pipe["detail_fix_settings"] else 10
        refiner_ratio = pipe["detail_fix_settings"]["refiner_ratio"] if "refiner_ratio" in pipe else 0.2
        batch_size = pipe["detail_fix_settings"]["batch_size"] if "batch_size" in pipe["detail_fix_settings"] else 1
        noise_mask = pipe["detail_fix_settings"]["noise_mask"] if "noise_mask" in pipe["detail_fix_settings"] else None
        force_inpaint = pipe["detail_fix_settings"]["force_inpaint"] if "force_inpaint" in pipe["detail_fix_settings"] else False
        wildcard = pipe["detail_fix_settings"]["wildcard"] if "wildcard" in pipe["detail_fix_settings"] else ""
        cycle = pipe["detail_fix_settings"]["cycle"] if "cycle" in pipe["detail_fix_settings"] else 1

        bbox_segm_pipe = pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None
        sam_pipe = pipe["sam_pipe"] if "sam_pipe" in pipe else None

        # 细节修复初始时间
        start_time = int(time.time() * 1000)
        if "mask_settings" in pipe:
            mask_mode = pipe['mask_settings']["mask_mode"] if "inpaint_model" in pipe['mask_settings'] else True
            inpaint_model = pipe['mask_settings']["inpaint_model"] if "inpaint_model" in pipe['mask_settings'] else False
            noise_mask_feather = pipe['mask_settings']["noise_mask_feather"] if "noise_mask_feather" in pipe['mask_settings'] else 20
            cls = ALL_NODE_CLASS_MAPPINGS["MaskDetailerPipe"]
            if "MaskDetailerPipe" not in ALL_NODE_CLASS_MAPPINGS:
                raise Exception(f"[ERROR] To use MaskDetailerPipe, you need to install 'Impact Pack'")
            basic_pipe = (model, clip, vae, positive, negative)
            result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, basic_pipe, refiner_basic_pipe_opt = cls().doit(image, mask, basic_pipe, guide_size, guide_size_for, max_size, mask_mode,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             feather, crop_factor, drop_size, refiner_ratio, batch_size, cycle=1,
             refiner_basic_pipe_opt=None, detailer_hook=None, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)
            result_mask = mask
            result_cnet_images = ()
        else:
            if bbox_segm_pipe is None:
                raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
            if sam_pipe is None:
                raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")
            bbox_detector_opt, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector_opt = bbox_segm_pipe
            sam_model_opt, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative = sam_pipe
            if "FaceDetailer" not in ALL_NODE_CLASS_MAPPINGS:
                raise Exception(f"[ERROR] To use FaceDetailer, you need to install 'Impact Pack'")
            cls = ALL_NODE_CLASS_MAPPINGS["FaceDetailer"]

            result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images = cls().doit(
                image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                scheduler,
                positive, negative, denoise, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector_opt, wildcard, cycle, sam_model_opt,
                segm_detector_opt,
                detailer_hook=None)

        # 细节修复结束时间
        end_time = int(time.time() * 1000)

        spent_time = 'Fix:' + str((end_time - start_time) / 1000) + '"'

        results = easySave(result_img, save_prefix, image_output, prompt, extra_pnginfo)
        sampler.update_value_by_id("results", my_unique_id, results)

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        new_pipe = {
            "samples": None,
            "images": result_img,
            "model": model,
            "clip": clip,
            "vae": vae,
            "seed": seed,
            "positive": positive,
            "negative": negative,
            "wildcard": wildcard,
            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": {
                **loader_settings,
                "spent_time": spent_time
            },
            "detail_fix_settings": detail_fix_settings
        }
        if "mask_settings" in pipe:
            new_pipe["mask_settings"] = pipe["mask_settings"]

        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del bbox_segm_pipe
        del sam_pipe
        del pipe

        if image_output in ("Hide", "Hide&Save"):
            return (new_pipe, result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, result_cnet_images)

        if image_output in ("Sender", "Sender&Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        return {"ui": {"images": results}, "result": (new_pipe, result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, result_cnet_images )}

class ultralyticsDetectorForDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {"required":
                    {"model_name": (bboxs + segms,),
                    "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                    "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                    }
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("bbox_segm_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, model_name, bbox_threshold, bbox_dilation, bbox_crop_factor):
        if 'UltralyticsDetectorProvider' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use UltralyticsDetectorProvider, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['UltralyticsDetectorProvider']
        bbox_detector, segm_detector = cls().doit(model_name)
        pipe = (bbox_detector, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector)
        return (pipe,)

class samLoaderForDetailerFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("sams"),),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],{"default": "AUTO"}),
                "sam_detection_hint": (
                ["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points",
                 "mask-point-bbox", "none"],),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
            }
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("sam_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, model_name, device_mode, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative):
        if 'SAMLoader' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use SAMLoader, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['SAMLoader']
        (sam_model,) = cls().load_model(model_name, device_mode)
        pipe = (sam_model, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative)
        return (pipe,)


NODE_CLASS_MAPPINGS = {
    "easy hiresFix": hiresFix,
    "easy preDetailerFix": preDetailerFix,
    "easy preMaskDetailerFix": preMaskDetailerFix,
    "easy ultralyticsDetectorPipe": ultralyticsDetectorForDetailerFix,
    "easy samLoaderPipe": samLoaderForDetailerFix,
    "easy detailerFix": detailerFix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy hiresFix": "HiresFix",
    "easy preDetailerFix": "PreDetailerFix",
    "easy preMaskDetailerFix": "preMaskDetailerFix",
    "easy ultralyticsDetectorPipe": "UltralyticsDetector (Pipe)",
    "easy samLoaderPipe": "SAMLoader (Pipe)",
    "easy detailerFix": "DetailerFix",
}