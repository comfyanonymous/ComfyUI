import re
import torch
import comfy
from comfy_extras.nodes_mask import GrowMask
from nodes import VAEEncodeForInpaint, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from ..libs.utils import get_local_filepath
from ..libs.log import log_node_info
from ..libs import cache as backend_cache
from ..config import *

# FooocusInpaint
class applyFooocusInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "head": (list(FOOOCUS_INPAINT_HEAD.keys()),),
                "patch": (list(FOOOCUS_INPAINT_PATCH.keys()),),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, model, latent, head, patch):
        from ..modules.fooocus import InpaintHead, InpaintWorker
        head_file = get_local_filepath(FOOOCUS_INPAINT_HEAD[head]["model_url"], INPAINT_DIR)
        inpaint_head_model = InpaintHead()
        sd = torch.load(head_file, map_location='cpu')
        inpaint_head_model.load_state_dict(sd)

        patch_file = get_local_filepath(FOOOCUS_INPAINT_PATCH[patch]["model_url"], INPAINT_DIR)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        patch = (inpaint_head_model, inpaint_lora)
        worker = InpaintWorker(node_name="easy kSamplerInpainting")
        cloned = model.clone()

        m, = worker.patch(cloned, latent, patch)
        return (m,)

# brushnet
from ..modules.brushnet import BrushNet
class applyBrushNet:

    def get_files_with_extension(folder='inpaint', extensions='.safetensors'):
        return [file for file in folder_paths.get_filename_list(folder) if file.endswith(extensions)]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "brushnet": (s.get_files_with_extension(),),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'], ),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, pipe, image, mask, brushnet, dtype, scale, start_at, end_at):

        model = pipe['model']
        vae = pipe['vae']
        positive = pipe['positive']
        negative = pipe['negative']
        cls = BrushNet()
        if brushnet in backend_cache.cache:
            log_node_info("easy brushnetApply", f"Using {brushnet} Cached")
            _, brushnet_model = backend_cache.cache[brushnet][1]
        else:
            brushnet_file = os.path.join(folder_paths.get_full_path("inpaint", brushnet))
            brushnet_model, = cls.load_brushnet_model(brushnet_file, dtype)
            backend_cache.update_cache(brushnet, 'brushnet', (False, brushnet_model))
        m, positive, negative, latent = cls.brushnet_model_update(model=model, vae=vae, image=image, mask=mask,
                                                           brushnet=brushnet_model, positive=positive,
                                                           negative=negative, scale=scale, start_at=start_at,
                                                           end_at=end_at)
        new_pipe = {
            **pipe,
            "model": m,
            "positive": positive,
            "negative": negative,
            "samples": latent,
        }
        del pipe
        return (new_pipe,)

# #powerpaint
class applyPowerPaint:
    def get_files_with_extension(folder='inpaint', extensions='.safetensors'):
        return [file for file in folder_paths.get_filename_list(folder) if file.endswith(extensions)]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "powerpaint_model": (s.get_files_with_extension(),),
                "powerpaint_clip": (s.get_files_with_extension(extensions='.bin'),),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'],),
                "fitting": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "save_memory": (['none', 'auto', 'max'],),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, pipe, image, mask, powerpaint_model, powerpaint_clip, dtype, fitting, function, scale, start_at, end_at, save_memory='none'):
        model = pipe['model']
        vae = pipe['vae']
        positive = pipe['positive']
        negative = pipe['negative']

        cls = BrushNet()
        # load powerpaint clip
        if powerpaint_clip in backend_cache.cache:
            log_node_info("easy powerpaintApply", f"Using {powerpaint_clip} Cached")
            _, ppclip = backend_cache.cache[powerpaint_clip][1]
        else:
            model_url = POWERPAINT_MODELS['base_fp16']['model_url']
            base_clip = get_local_filepath(model_url, os.path.join(folder_paths.models_dir, 'clip'))
            ppclip, = cls.load_powerpaint_clip(base_clip, os.path.join(folder_paths.get_full_path("inpaint", powerpaint_clip)))
            backend_cache.update_cache(powerpaint_clip, 'ppclip', (False, ppclip))
        # load powerpaint model
        if powerpaint_model in backend_cache.cache:
            log_node_info("easy powerpaintApply", f"Using {powerpaint_model} Cached")
            _, powerpaint = backend_cache.cache[powerpaint_model][1]
        else:
            powerpaint_file = os.path.join(folder_paths.get_full_path("inpaint", powerpaint_model))
            powerpaint, = cls.load_brushnet_model(powerpaint_file, dtype)
            backend_cache.update_cache(powerpaint_model, 'powerpaint', (False, powerpaint))
        m, positive, negative, latent = cls.powerpaint_model_update(model=model, vae=vae, image=image, mask=mask, powerpaint=powerpaint,
                                                           clip=ppclip, positive=positive,
                                                           negative=negative, fitting=fitting, function=function,
                                                           scale=scale, start_at=start_at, end_at=end_at, save_memory=save_memory)
        new_pipe = {
            **pipe,
            "model": m,
            "positive": positive,
            "negative": negative,
            "samples": latent,
        }
        del pipe
        return (new_pipe,)

from node_helpers import conditioning_set_values
class applyInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "inpaint_mode": (('normal', 'fooocus_inpaint', 'brushnet_random', 'brushnet_segmentation', 'powerpaint'),),
                "encode": (('none', 'vae_encode_inpaint', 'inpaint_model_conditioning', 'different_diffusion'), {"default": "none"}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'],),
                "fitting": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            },
            "optional":{
                "noise_mask": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def inpaint_model_conditioning(self, pipe, image, vae, mask, grow_mask_by, noise_mask=True):
        if grow_mask_by >0:
            mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
        positive, negative, = pipe['positive'], pipe['negative']

        pixels = image
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)

        pipe['positive'] = out[0]
        pipe['negative'] = out[1]
        pipe['samples'] = out_latent

        return pipe

    def get_brushnet_model(self, type, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if type == 'brushnet_random':
            brush_model = BRUSHNET_MODELS['random_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.random.mask.sdxl.*.(safetensors|bin)$'
            else:
                pattern = 'brushnet.random.mask.*.(safetensors|bin)$'
        elif type == 'brushnet_segmentation':
            brush_model = BRUSHNET_MODELS['segmentation_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.segmentation.mask.sdxl.*.(safetensors|bin)$'
            else:
                pattern = 'brushnet.segmentation.mask.*.(safetensors|bin)$'


        brushfile = [e for e in folder_paths.get_filename_list('inpaint') if re.search(pattern, e, re.IGNORECASE)]
        brushname = brushfile[0] if brushfile else None
        if not brushname:
            from urllib.parse import urlparse
            get_local_filepath(brush_model, INPAINT_DIR)
            parsed_url = urlparse(brush_model)
            brushname = os.path.basename(parsed_url.path)
        return brushname

    def get_powerpaint_model(self, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if model_type == 'sdxl':
            raise Exception("Powerpaint not supported for SDXL models")

        powerpaint_model = POWERPAINT_MODELS['v2.1']['model_url']
        powerpaint_clip = POWERPAINT_MODELS['v2.1']['clip_url']

        from urllib.parse import urlparse
        get_local_filepath(powerpaint_model, os.path.join(INPAINT_DIR, 'powerpaint'))
        model_parsed_url = urlparse(powerpaint_model)
        clip_parsed_url = urlparse(powerpaint_clip)
        model_name = os.path.join("powerpaint",os.path.basename(model_parsed_url.path))
        clip_name = os.path.join("powerpaint",os.path.basename(clip_parsed_url.path))
        return model_name, clip_name

    def apply(self, pipe, image, mask, inpaint_mode, encode, grow_mask_by, dtype, fitting, function, scale, start_at, end_at, noise_mask=True):
        new_pipe = {
            **pipe,
        }
        del pipe
        if inpaint_mode in ['brushnet_random', 'brushnet_segmentation']:
            brushnet = self.get_brushnet_model(inpaint_mode, new_pipe['model'])
            new_pipe, = applyBrushNet().apply(new_pipe, image, mask, brushnet, dtype, scale, start_at, end_at)
        elif inpaint_mode == 'powerpaint':
            powerpaint_model, powerpaint_clip = self.get_powerpaint_model(new_pipe['model'])
            new_pipe, = applyPowerPaint().apply(new_pipe, image, mask, powerpaint_model, powerpaint_clip, dtype, fitting, function, scale, start_at, end_at)

        vae = new_pipe['vae']
        if encode == 'none':
            if inpaint_mode == 'fooocus_inpaint':
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
        elif encode == 'vae_encode_inpaint':
            latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
            new_pipe['samples'] = latent
            if inpaint_mode == 'fooocus_inpaint':
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
        elif encode == 'inpaint_model_conditioning':
            if inpaint_mode == 'fooocus_inpaint':
                latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
                new_pipe['samples'] = latent
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, 0, noise_mask=noise_mask)
            else:
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, grow_mask_by, noise_mask=noise_mask)
        elif encode == 'different_diffusion':
            if inpaint_mode == 'fooocus_inpaint':
                latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
                new_pipe['samples'] = latent
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, 0, noise_mask=noise_mask)
            else:
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, grow_mask_by, noise_mask=noise_mask)
            cls = ALL_NODE_CLASS_MAPPINGS['DifferentialDiffusion']
            if cls is not None:
                model, = cls().apply(new_pipe['model'])
                new_pipe['model'] = model
            else:
                raise Exception("Differential Diffusion not found,please update comfyui")

        return (new_pipe,)

NODE_CLASS_MAPPINGS = {
    "easy applyFooocusInpaint": applyFooocusInpaint,
    "easy applyBrushNet": applyBrushNet,
    "easy applyPowerPaint": applyPowerPaint,
    "easy applyInpaint": applyInpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy applyFooocusInpaint": "Easy Apply Fooocus Inpaint",
    "easy applyBrushNet": "Easy Apply BrushNet",
    "easy applyPowerPaint": "Easy Apply PowerPaint",
    "easy applyInpaint": "Easy Apply Inpaint"
}