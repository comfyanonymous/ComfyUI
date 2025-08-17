import re
import torch
import comfy.utils, comfy.sample, comfy.samplers, comfy.controlnet, comfy.model_base, comfy.model_management, comfy.sampler_helpers, comfy.supported_models
from comfy_extras.nodes_compositing import JoinImageWithAlpha
from comfy.clip_vision import load as load_clip_vision

from nodes import  NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from ..config import *

from ..libs.log import log_node_info, log_node_warn
from ..libs.utils import get_local_filepath, get_sd_version
from ..libs.wildcards import process_with_loras
from ..libs.controlnet import easyControlnet
from ..libs.conditioning import prompt_to_cond
from ..libs import cache as backend_cache

from .. import easyCache

class applyLoraPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "positive": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "negative": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "positive", "negative")
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, clip, positive, negative=None):
        model, clip, positive, _, _, _  = process_with_loras(positive, model, clip, 'Positive', easyCache=easyCache)
        if negative is not None:
            model, clip, negative, _, _, _  = process_with_loras(negative, model, clip, 'Negative', easyCache=easyCache)
        
        return (model, clip, positive, negative if negative is not None else "")
    
class applyLoraStack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_stack": ("LORA_STACK",),
                "model": ("MODEL",),
            },
            "optional": {
                "optional_clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, lora_stack, model, optional_clip=None):
        clip = None
        if lora_stack is not None and len(lora_stack) > 0:
            for lora in lora_stack:
                lora = {"lora_name": lora[0], "model": model, "clip": optional_clip, "model_strength": lora[1],
                        "clip_strength": lora[2]}
                model, clip = easyCache.load_lora(lora, model, optional_clip, use_cache=False)
        return (model, optional_clip if clip is None else clip)

class applyControlnetStack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet_stack": ("CONTROL_NET_STACK",),
                "pipe": ("PIPE_LINE",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, controlnet_stack, pipe):

        positive = pipe['positive']
        negative = pipe['negative']
        model = pipe['model']
        vae = pipe['vae']

        if controlnet_stack is not None and len(controlnet_stack) >0:
            for controlnet in controlnet_stack:
                positive, negative = easyControlnet().apply(controlnet[0], controlnet[5], positive, negative, controlnet[1], start_percent=controlnet[2], end_percent=controlnet[3], control_net=None, scale_soft_weights=controlnet[4], mask=None, easyCache=easyCache, use_cache=False, model=model, vae=vae)

        new_pipe = {
            **pipe,
            "positive": positive,
            "negetive": negative,
        }
        del pipe

        return (new_pipe,)

# 风格对齐
from ..libs.styleAlign import styleAlignBatch, SHARE_NORM_OPTIONS, SHARE_ATTN_OPTIONS
class styleAlignedBatchAlign:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "align"
    CATEGORY = "EasyUse/Adapter"

    def align(self, model, share_norm, share_attn, scale):
        return (styleAlignBatch(model, share_norm, share_attn, scale),)

# 光照对齐
from ..modules.ic_light import ICLight, VAEEncodeArgMax
class icLightApply:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(IC_LIGHT_MODELS.keys()),),
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "lighting": (['None', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light', 'Circle Light'],{"default": "None"}),
                "source": (['Use Background Image', 'Use Flipped Background Image', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light', 'Ambient'],{"default": "Use Background Image"}),
                "remove_bg": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "lighting_image")
    FUNCTION = "apply"
    CATEGORY = "EasyUse/Adapter"

    def batch(self, image1, image2):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear",
                                                "center").movedim(1, -1)
        s = torch.cat((image1, image2), dim=0)
        return s

    def removebg(self, image):
        if "easy imageRemBg" not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception("Please re-install ComfyUI-Easy-Use")
        cls = ALL_NODE_CLASS_MAPPINGS['easy imageRemBg']
        results = cls().remove('RMBG-1.4', image, 'Hide', 'ComfyUI')
        if "result" in results:
            image, _ = results['result']
            return image

    def apply(self, mode, model, image, vae, lighting, source, remove_bg):
        model_type = get_sd_version(model)
        if model_type == 'sdxl':
            raise Exception("IC Light model is not supported for SDXL now")

        batch_size, height, width, channel = image.shape
        if channel == 3:
            # remove bg
            if mode == 'Foreground' or batch_size == 1:
                if remove_bg:
                    image = self.removebg(image)
                else:
                    mask = torch.full((1, height, width), 1.0, dtype=torch.float32, device="cpu")
                    image, = JoinImageWithAlpha().join_image_with_alpha(image, mask)

        iclight = ICLight()
        if mode == 'Foreground':
          lighting_image = iclight.generate_lighting_image(image, lighting)
        else:
          lighting_image = iclight.generate_source_image(image, source)
          if source not in ['Use Background Image', 'Use Flipped Background Image']:
              _, height, width, _ = lighting_image.shape
              mask = torch.full((1, height, width), 1.0, dtype=torch.float32, device="cpu")
              lighting_image, = JoinImageWithAlpha().join_image_with_alpha(lighting_image, mask)
              if batch_size < 2:
                image = self.batch(image, lighting_image)
              else:
                original_image = [img.unsqueeze(0) for img in image]
                original_image = self.removebg(original_image[0])
                image = self.batch(original_image, lighting_image)

        latent, = VAEEncodeArgMax().encode(vae, image)
        key = 'iclight_' + mode + '_' + model_type
        model_path = get_local_filepath(IC_LIGHT_MODELS[mode]['sd1']["model_url"],
                                        os.path.join(folder_paths.models_dir, "unet"))
        ic_model = None
        if key in backend_cache.cache:
            log_node_info("easy icLightApply", f"Using icLightModel {mode+'_'+model_type} Cached")
            _, ic_model = backend_cache.cache[key][1]
            m, _ = iclight.apply(model_path, model, latent, ic_model)
        else:
            m, ic_model = iclight.apply(model_path, model, latent, ic_model)
            backend_cache.update_cache(key, 'iclight', (False, ic_model))
        return (m, lighting_image)


def insightface_loader(provider, name='buffalo_l'):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)
    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name=name, root=path, providers=[provider + 'ExecutionProvider', ])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

# Apply Ipadapter
class ipadapter:

    def __init__(self):
        self.normal_presets = [
            'LIGHT - SD1.5 only (low strength)',
            'STANDARD (medium strength)',
            'VIT-G (medium strength)',
            'PLUS (high strength)',
            'PLUS (kolors genernal)',
            'REGULAR - FLUX and SD3.5 only (high strength)',
            'PLUS FACE (portraits)',
            'FULL FACE - SD1.5 only (portraits stronger)',
            'COMPOSITION'
        ]
        self.faceid_presets = [
            'FACEID',
            'FACEID PLUS - SD1.5 only',
            "FACEID PLUS KOLORS",
            'FACEID PLUS V2',
            'FACEID PORTRAIT (style transfer)',
            'FACEID PORTRAIT UNNORM - SDXL only (strong)'
        ]
        self.weight_types = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise']
        self.presets = self.normal_presets + self.faceid_presets


    def error(self):
        raise Exception(f"[ERROR] To use ipadapterApply, you need to install 'ComfyUI_IPAdapter_plus'")

    def get_clipvision_file(self, preset, node_name):
        preset = preset.lower()
        clipvision_list = folder_paths.get_filename_list("clip_vision")
        if preset.startswith("regular"):
            # pattern = 'sigclip.vision.patch14.384'
            pattern = 'siglip.so400m.patch14.384'
        elif preset.startswith("plus (kolors") or preset.startswith("faceid plus kolors"):
            pattern = 'Vit.Large.patch14.336.(bin|safetensors)$'
        elif preset.startswith("vit-g"):
            pattern = '(ViT.bigG.14.*39B.b160k|ipadapter.*sdxl|sdxl.*model.(bin|safetensors))'
        else:
            pattern = '(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model.(bin|safetensors))'
        clipvision_files = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]
        clipvision_name = clipvision_files[0] if len(clipvision_files)>0 else None
        clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_name) if clipvision_name else None
        # if clipvision_name is not None:
        #     log_node_info(node_name, f"Using {clipvision_name}")
        return clipvision_file, clipvision_name

    def get_ipadapter_file(self, preset, model_type, node_name):
        preset = preset.lower()
        ipadapter_list = folder_paths.get_filename_list("ipadapter")
        is_insightface = False
        lora_pattern = None
        is_sdxl = model_type == 'sdxl'
        is_flux = model_type == 'flux'

        if preset.startswith("light"):
            if is_sdxl:
                raise Exception("light model is not supported for SDXL")
            pattern = 'sd15.light.v11.(safetensors|bin)$'
            # if light model v11 is not found, try with the old version
            if not [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]:
                pattern = 'sd15.light.(safetensors|bin)$'
        elif preset.startswith("standard"):
            if is_sdxl:
                pattern = 'ip.adapter.sdxl.vit.h.(safetensors|bin)$'
            else:
                pattern = 'ip.adapter.sd15.(safetensors|bin)$'
        elif preset.startswith("vit-g"):
            if is_sdxl:
                pattern = 'ip.adapter.sdxl.(safetensors|bin)$'
            else:
                pattern = 'sd15.vit.g.(safetensors|bin)$'
        elif preset.startswith("regular"):
            if is_flux:
                pattern = 'ip.adapter.flux.1.dev.(safetensors|bin)$'
            else:
                pattern = 'ip.adapter.sd35.(safetensors|bin)$'
        elif preset.startswith("plus (high"):
            if is_sdxl:
                pattern = 'plus.sdxl.vit.h.(safetensors|bin)$'
            else:
                pattern = 'ip.adapter.plus.sd15.(safetensors|bin)$'
        elif preset.startswith("plus (kolors"):
            if is_sdxl:
                pattern = 'plus.gener(nal|al).(safetensors|bin)$'
            else:
                raise Exception("kolors model is not supported for SD15")
        elif preset.startswith("plus face"):
            if is_sdxl:
                pattern = 'plus.face.sdxl.vit.h.(safetensors|bin)$'
            else:
                pattern = 'plus.face.sd15.(safetensors|bin)$'
        elif preset.startswith("full"):
            if is_sdxl:
                raise Exception("full face model is not supported for SDXL")
            pattern = 'full.face.sd15.(safetensors|bin)$'
        elif preset.startswith("composition"):
            if is_sdxl:
                pattern = 'plus.composition.sdxl.(safetensors|bin)$'
            else:
                pattern = 'plus.composition.sd15.(safetensors|bin)$'
        elif preset.startswith("faceid portrait ("):
            if is_sdxl:
                pattern = 'portrait.sdxl.(safetensors|bin)$'
            else:
                pattern = 'portrait.v11.sd15.(safetensors|bin)$'
                # if v11 is not found, try with the old version
                if not [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]:
                    pattern = 'portrait.sd15.(safetensors|bin)$'
            is_insightface = True
        elif preset.startswith("faceid portrait unnorm"):
            if is_sdxl:
                pattern = r'portrait.sdxl.unnorm.(safetensors|bin)$'
            else:
                raise Exception("portrait unnorm model is not supported for SD1.5")
            is_insightface = True
        elif preset == "faceid":
            if is_sdxl:
                pattern = 'faceid.sdxl.(safetensors|bin)$'
                lora_pattern = 'faceid.sdxl.lora.safetensors$'
            else:
                pattern = 'faceid.sd15.(safetensors|bin)$'
                lora_pattern = 'faceid.sd15.lora.safetensors$'
            is_insightface = True
        elif preset.startswith("faceid plus kolors"):
            if is_sdxl:
                pattern = '(kolors.ip.adapter.faceid.plus|ipa.faceid.plus).(safetensors|bin)$'
            else:
                raise Exception("faceid plus kolors model is not supported for SD1.5")
            is_insightface = True
        elif preset.startswith("faceid plus -"):
            if is_sdxl:
                raise Exception("faceid plus model is not supported for SDXL")
            pattern = 'faceid.plus.sd15.(safetensors|bin)$'
            lora_pattern = 'faceid.plus.sd15.lora.safetensors$'
            is_insightface = True
        elif preset.startswith("faceid plus v2"):
            if is_sdxl:
                pattern = 'faceid.plusv2.sdxl.(safetensors|bin)$'
                lora_pattern = 'faceid.plusv2.sdxl.lora.safetensors$'
            else:
                pattern = 'faceid.plusv2.sd15.(safetensors|bin)$'
                lora_pattern = 'faceid.plusv2.sd15.lora.safetensors$'
            is_insightface = True
        else:
            raise Exception(f"invalid type '{preset}'")

        ipadapter_files = [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]
        ipadapter_name = ipadapter_files[0] if len(ipadapter_files)>0 else None
        ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_name) if ipadapter_name else None
        # if ipadapter_name is not None:
        #     log_node_info(node_name, f"Using {ipadapter_name}")

        return ipadapter_file, ipadapter_name, is_insightface, lora_pattern

    def get_lora_pattern(self, file):
        basename = os.path.basename(file)
        lora_pattern = None
        if re.search(r'faceid.sdxl.(safetensors|bin)$', basename, re.IGNORECASE):
            lora_pattern = 'faceid.sdxl.lora.safetensors$'
        elif re.search(r'faceid.sd15.(safetensors|bin)$', basename, re.IGNORECASE):
            lora_pattern = 'faceid.sd15.lora.safetensors$'
        elif re.search(r'faceid.plus.sd15.(safetensors|bin)$', basename, re.IGNORECASE):
            lora_pattern = 'faceid.plus.sd15.lora.safetensors$'
        elif re.search(r'faceid.plusv2.sdxl.(safetensors|bin)$', basename, re.IGNORECASE):
            lora_pattern = 'faceid.plusv2.sdxl.lora.safetensors$'
        elif re.search(r'faceid.plusv2.sd15.(safetensors|bin)$', basename, re.IGNORECASE):
            lora_pattern = 'faceid.plusv2.sd15.lora.safetensors$'

        return lora_pattern

    def get_lora_file(self, preset, pattern, model_type, model, model_strength, clip_strength, clip=None):
        lora_list = folder_paths.get_filename_list("loras")
        lora_files = [e for e in lora_list if re.search(pattern, e, re.IGNORECASE)]
        lora_name = lora_files[0] if lora_files else None
        if lora_name:
            return easyCache.load_lora({"model": model, "clip": clip, "lora_name": lora_name, "model_strength":model_strength, "clip_strength":clip_strength},)
        else:
            if "lora_url" in IPADAPTER_MODELS[preset][model_type]:
                lora_name = get_local_filepath(IPADAPTER_MODELS[preset][model_type]["lora_url"], os.path.join(folder_paths.models_dir, "loras"))
                return easyCache.load_lora({"model": model, "clip": clip, "lora_name": lora_name, "model_strength":model_strength, "clip_strength":clip_strength},)
            return (model, clip)

    def ipadapter_model_loader(self, file):
        model = comfy.utils.load_torch_file(file, safe_load=False)

        if file.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
            del st_model

        model_keys = model.keys()
        if "adapter_modules" in model_keys:
            model["ip_adapter"] = model["adapter_modules"]
            model["faceidplusv2"] = True
            del model['adapter_modules']

        if not "ip_adapter" in model_keys or not model["ip_adapter"]:
            raise Exception("invalid IPAdapter model {}".format(file))

        if 'plusv2' in file.lower():
            model["faceidplusv2"] = True

        if 'unnorm' in file.lower():
            model["portraitunnorm"] = True

        return model

    def load_model(self, model, preset, lora_model_strength, provider="CPU", clip_vision=None, optional_ipadapter=None, cache_mode='none', node_name='easy ipadapterApply'):
        pipeline = {"clipvision": {'file': None, 'model': None}, "ipadapter": {'file': None, 'model': None},
                    "insightface": {'provider': None, 'model': None}}
        ipadapter, insightface, is_insightface, lora_pattern = None, None, None, None
        if optional_ipadapter is not None:
            pipeline = optional_ipadapter
            if not clip_vision:
                clip_vision = pipeline['clipvision']['model']
            ipadapter = pipeline['ipadapter']['model']
            if 'insightface' in pipeline:
                insightface = pipeline['insightface']['model']
                lora_pattern = self.get_lora_pattern(pipeline['ipadapter']['file'])

        # 1. Load the clipvision model
        if not clip_vision:
            clipvision_file, clipvision_name = self.get_clipvision_file(preset, node_name)
            if clipvision_file is None:
                if preset.lower().startswith("regular"):
                    # model_url = IPADAPTER_CLIPVISION_MODELS["sigclip_vision_patch14_384"]["model_url"]
                    # clipvision_file = get_local_filepath(model_url, IPADAPTER_DIR, "sigclip_vision_patch14_384.bin")
                    from huggingface_hub import snapshot_download
                    import shutil
                    CLIP_PATH = os.path.join(folder_paths.models_dir, "clip_vision", "google--siglip-so400m-patch14-384")
                    print("CLIP_VISION not found locally. Downloading google/siglip-so400m-patch14-384...")
                    try:
                        snapshot_download(
                            repo_id="google/siglip-so400m-patch14-384",
                            local_dir=os.path.join(folder_paths.models_dir, "clip_vision",
                                                   "cache--google--siglip-so400m-patch14-384"),
                            local_dir_use_symlinks=False,
                            resume_download=True
                        )
                        shutil.move(os.path.join(folder_paths.models_dir, "clip_vision",
                                                 "cache--google--siglip-so400m-patch14-384"), CLIP_PATH)
                        print(f"CLIP_VISION has been downloaded to {CLIP_PATH}")
                    except Exception as e:
                        print(f"Error downloading CLIP model: {e}")
                        raise
                    clipvision_file = CLIP_PATH
                elif preset.lower().startswith("plus (kolors"):
                    model_url = IPADAPTER_CLIPVISION_MODELS["clip-vit-large-patch14-336"]["model_url"]
                    clipvision_file = get_local_filepath(model_url, IPADAPTER_DIR, "clip-vit-large-patch14-336.bin")
                else:
                    model_url = IPADAPTER_CLIPVISION_MODELS["clip-vit-h-14-laion2B-s32B-b79K"]["model_url"]
                    clipvision_file = get_local_filepath(model_url, IPADAPTER_DIR, "clip-vit-h-14-laion2B-s32B-b79K.safetensors")
                clipvision_name = os.path.basename(model_url)
            if clipvision_file == pipeline['clipvision']['file']:
                clip_vision = pipeline['clipvision']['model']
            elif cache_mode in ["all", "clip_vision only"] and clipvision_name in backend_cache.cache:
                log_node_info("easy ipadapterApply", f"Using ClipVisonModel {clipvision_name} Cached")
                _, clip_vision = backend_cache.cache[clipvision_name][1]
            else:
                if preset.lower().startswith("regular"):
                    from transformers import SiglipVisionModel, AutoProcessor
                    image_encoder_path = os.path.dirname(clipvision_file)
                    image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
                    clip_image_processor = AutoProcessor.from_pretrained(image_encoder_path)
                    clip_vision = {
                        'image_encoder': image_encoder,
                        'clip_image_processor': clip_image_processor
                    }
                else:
                    clip_vision = load_clip_vision(clipvision_file)
                log_node_info("easy ipadapterApply", f"Using ClipVisonModel {clipvision_name}")
                if cache_mode in ["all", "clip_vision only"]:
                    backend_cache.update_cache(clipvision_name, 'clip_vision', (False, clip_vision))
            pipeline['clipvision']['file'] = clipvision_file
            pipeline['clipvision']['model'] = clip_vision
        # 2. Load the ipadapter model
        model_type = get_sd_version(model)
        if not ipadapter:
            ipadapter_file, ipadapter_name, is_insightface, lora_pattern = self.get_ipadapter_file(preset, model_type, node_name)
            if ipadapter_file is None:
                model_url = IPADAPTER_MODELS[preset][model_type]["model_url"]
                local_file_name = IPADAPTER_MODELS[preset][model_type]['model_file_name'] if "model_file_name" in IPADAPTER_MODELS[preset][model_type] else None
                ipadapter_file = get_local_filepath(model_url, IPADAPTER_DIR, local_file_name)
                ipadapter_name = os.path.basename(model_url)
            if ipadapter_file == pipeline['ipadapter']['file']:
                ipadapter = pipeline['ipadapter']['model']
            elif cache_mode in ["all", "ipadapter only"] and ipadapter_name in backend_cache.cache:
                log_node_info("easy ipadapterApply", f"Using IpAdapterModel {ipadapter_name} Cached")
                _, ipadapter = backend_cache.cache[ipadapter_name][1]
            else:
                ipadapter = self.ipadapter_model_loader(ipadapter_file)
                pipeline['ipadapter']['file'] = ipadapter_file
                log_node_info("easy ipadapterApply", f"Using IpAdapterModel {ipadapter_name}")
                if cache_mode in ["all", "ipadapter only"]:
                    backend_cache.update_cache(ipadapter_name, 'ipadapter', (False, ipadapter))

            pipeline['ipadapter']['model'] = ipadapter

        # 3. Load the lora model if needed
        if lora_pattern is not None:
            if lora_model_strength > 0:
              model, _ = self.get_lora_file(preset, lora_pattern, model_type, model, lora_model_strength, 1)

        # 4. Load the insightface model if needed
        if is_insightface:
            if not insightface:
                icache_key = 'insightface-' + provider
                if provider == pipeline['insightface']['provider']:
                    insightface = pipeline['insightface']['model']
                elif cache_mode in ["all", "insightface only"] and icache_key in backend_cache.cache:
                    log_node_info("easy ipadapterApply", f"Using InsightFaceModel {icache_key} Cached")
                    _, insightface = backend_cache.cache[icache_key][1]
                else:
                    insightface = insightface_loader(provider, 'antelopev2' if preset == 'FACEID PLUS KOLORS' else 'buffalo_l')
                    if cache_mode in ["all", "insightface only"]:
                        backend_cache.update_cache(icache_key, 'insightface',(False, insightface))
                pipeline['insightface']['provider'] = provider
                pipeline['insightface']['model'] = insightface

        return (model, pipeline,)

class ipadapterApply(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        presets = cls().presets
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "preset": (presets,),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"], {"default": "CUDA"}),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"], {"default": "all"},),
                "use_tiled": ("BOOLEAN", {"default": False},),
            },

            "optional": {
                "attn_mask": ("MASK",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", "IPADAPTER",)
    RETURN_NAMES = ("model", "images", "masks", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, image, preset, lora_strength, provider, weight, weight_faceidv2, start_at, end_at, cache_mode, use_tiled, attn_mask=None, optional_ipadapter=None, weight_kolors=None):
        images, masks = image, [None]
        model, ipadapter = self.load_model(model, preset, lora_strength, provider, clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)
        if preset == 'REGULAR - FLUX and SD3.5 only (high strength)':
            from ..modules.ipadapter import InstantXFluxIpadapterApply, InstantXSD3IpadapterApply
            model_type = get_sd_version(model)
            if model_type == 'flux':
                model, images = InstantXFluxIpadapterApply().apply_ipadapter(model, ipadapter, image, weight, start_at, end_at, provider)
            elif model_type == 'sd3':
                model, images = InstantXSD3IpadapterApply().apply_ipadapter(model, ipadapter, image, weight, start_at, end_at, provider)
        elif use_tiled and preset not in self.faceid_presets:
            if "IPAdapterTiled" not in ALL_NODE_CLASS_MAPPINGS:
                self.error()
            cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiled"]
            model, images, masks = cls().apply_tiled(model, ipadapter, image, weight, "linear", start_at, end_at, sharpening=0.0, combine_embeds="concat", image_negative=None, attn_mask=attn_mask, clip_vision=None, embeds_scaling='V only')
        else:
            if preset in ['FACEID PLUS KOLORS', 'FACEID PLUS V2', 'FACEID PORTRAIT (style transfer)']:
                if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
                if weight_kolors is None:
                    weight_kolors = weight
                model, images = cls().apply_ipadapter(model, ipadapter, start_at=start_at, end_at=end_at, weight=weight, weight_type="linear", combine_embeds="concat", weight_faceidv2=weight_faceidv2, image=image, image_negative=None, clip_vision=None, attn_mask=attn_mask, insightface=None, embeds_scaling='V only', weight_kolors=weight_kolors)
            else:
                if "IPAdapter" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapter"]
                model, images = cls().apply_ipadapter(model, ipadapter, image, weight, start_at, end_at, weight_type='standard', attn_mask=attn_mask)
        if images is None:
            images = image
        return (model, images, masks, ipadapter,)

class ipadapterApplyAdvanced(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        presets = ipa_cls.presets
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "preset": (presets,),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"], {"default": "CUDA"}),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (weight_types,),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only","ipadapter only", "all", "none"], {"default": "all"},),
                "use_tiled": ("BOOLEAN", {"default": False},),
                "use_batch": ("BOOLEAN", {"default": False},),
                "sharpening": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },

            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "optional_ipadapter": ("IPADAPTER",),
                "layer_weights": ("STRING", {"default": "", "multiline": True, "placeholder": "Mad Scientist Layer Weights"}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", "IPADAPTER",)
    RETURN_NAMES = ("model", "images", "masks", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, image, preset, lora_strength, provider, weight, weight_faceidv2, weight_type, combine_embeds, start_at, end_at, embeds_scaling, cache_mode, use_tiled, use_batch, sharpening, weight_style=1.0, weight_composition=1.0, image_style=None, image_composition=None, expand_style=False, image_negative=None, clip_vision=None, attn_mask=None, optional_ipadapter=None, layer_weights=None, weight_kolors=None):
        images, masks = image, [None]
        model, ipadapter = self.load_model(model, preset, lora_strength, provider, clip_vision=clip_vision, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)

        if weight_kolors is None:
            weight_kolors = weight

        if layer_weights:
            if "IPAdapterMS" not in ALL_NODE_CLASS_MAPPINGS:
                self.error()
            cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
            model, images = cls().apply_ipadapter(model, ipadapter, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, combine_embeds=combine_embeds, weight_faceidv2=weight_faceidv2, image=image, image_negative=image_negative, weight_style=weight_style, weight_composition=weight_composition, image_style=image_style, image_composition=image_composition, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling, layer_weights=layer_weights, weight_kolors=weight_kolors)
        elif use_tiled:
            if use_batch:
                if "IPAdapterTiledBatch" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiledBatch"]
            else:
                if "IPAdapterTiled" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiled"]
            model, images, masks = cls().apply_tiled(model, ipadapter, image=image, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, sharpening=sharpening, combine_embeds=combine_embeds, image_negative=image_negative, attn_mask=attn_mask, clip_vision=clip_vision, embeds_scaling=embeds_scaling)
        else:
            if use_batch:
                if "IPAdapterBatch" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterBatch"]
            else:
                if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
            model, images = cls().apply_ipadapter(model, ipadapter, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, combine_embeds=combine_embeds, weight_faceidv2=weight_faceidv2, image=image, image_negative=image_negative, weight_style=1.0, weight_composition=1.0, image_style=image_style, image_composition=image_composition, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling, weight_kolors=weight_kolors)
        if images is None:
            images = image
        return (model, images, masks, ipadapter)

class ipadapterApplyFaceIDKolors(ipadapterApplyAdvanced):

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        presets = ipa_cls.presets
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "preset": (['FACEID PLUS KOLORS'], {"default":"FACEID PLUS KOLORS"}),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"], {"default": "CUDA"}),
                "weight": ("FLOAT", {"default": 0.8, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05}),
                "weight_kolors": ("FLOAT", {"default": 0.8, "min": -1, "max": 5.0, "step": 0.05}),
                "weight_type": (weight_types,),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"], {"default": "all"},),
                "use_tiled": ("BOOLEAN", {"default": False},),
                "use_batch": ("BOOLEAN", {"default": False},),
                "sharpening": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },

            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }


class ipadapterStyleComposition(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        normal_presets = ipa_cls.normal_presets
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "image_style": ("IMAGE",),
                "preset": (normal_presets,),
                "weight_style": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "weight_composition": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "expand_style": ("BOOLEAN", {"default": False}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"], {"default": "average"}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"],
                               {"default": "all"},),
            },
            "optional": {
                "image_composition": ("IMAGE",),
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }

    CATEGORY = "EasyUse/Adapter"

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter",)
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, preset, weight_style, weight_composition, expand_style, combine_embeds, start_at, end_at, embeds_scaling, cache_mode, image_style=None , image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, optional_ipadapter=None):
        model, ipadapter = self.load_model(model, preset, 0, 'CPU', clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)

        if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()
        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]

        model, image = cls().apply_ipadapter(model, ipadapter, start_at=start_at, end_at=end_at, weight_style=weight_style, weight_composition=weight_composition, weight_type='linear', combine_embeds=combine_embeds, weight_faceidv2=weight_composition, image_style=image_style, image_composition=image_composition, image_negative=image_negative, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling)
        return (model, ipadapter)

class ipadapterApplyEncoder(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        normal_presets = ipa_cls.normal_presets
        max_embeds_num = 4
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "image1": ("IMAGE",),
                "preset": (normal_presets,),
                "num_embeds":  ("INT", {"default": 2, "min": 1, "max": max_embeds_num}),
            },
            "optional": {}
        }

        for i in range(1, max_embeds_num + 1):
            if i > 1:
                inputs["optional"][f"image{i}"] = ("IMAGE",)
        for i in range(1, max_embeds_num + 1):
            inputs["optional"][f"mask{i}"] = ("MASK",)
            inputs["optional"][f"weight{i}"] = ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05})
        inputs["optional"]["combine_method"] = (["concat", "add", "subtract", "average", "norm average", "max", "min"],)
        inputs["optional"]["optional_ipadapter"] = ("IPADAPTER",)
        inputs["optional"]["pos_embeds"] = ("EMBEDS",)
        inputs["optional"]["neg_embeds"] = ("EMBEDS",)
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP_VISION","IPADAPTER", "EMBEDS", "EMBEDS", )
    RETURN_NAMES = ("model", "clip_vision","ipadapter", "pos_embed", "neg_embed",)
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def batch(self, embeds, method):
        if method == 'concat' and len(embeds) == 1:
            return (embeds[0],)

        embeds = [embed for embed in embeds if embed is not None]
        embeds = torch.cat(embeds, dim=0)

        if method == "add":
            embeds = torch.sum(embeds, dim=0).unsqueeze(0)
        elif method == "subtract":
            embeds = embeds[0] - torch.mean(embeds[1:], dim=0)
            embeds = embeds.unsqueeze(0)
        elif method == "average":
            embeds = torch.mean(embeds, dim=0).unsqueeze(0)
        elif method == "norm average":
            embeds = torch.mean(embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        elif method == "max":
            embeds = torch.max(embeds, dim=0).values.unsqueeze(0)
        elif method == "min":
            embeds = torch.min(embeds, dim=0).values.unsqueeze(0)

        return embeds

    def apply(self, **kwargs):
        model = kwargs['model']
        clip_vision = kwargs['clip_vision']
        preset = kwargs['preset']
        if 'optional_ipadapter' in kwargs:
            ipadapter = kwargs['optional_ipadapter']
        else:
            model, ipadapter = self.load_model(model, preset, 0, 'CPU', clip_vision=clip_vision, optional_ipadapter=None, cache_mode='none')

        if "IPAdapterEncoder" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()
        encoder_cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterEncoder"]
        pos_embeds = kwargs["pos_embeds"] if "pos_embeds" in kwargs else []
        neg_embeds = kwargs["neg_embeds"] if "neg_embeds" in kwargs else []
        for i in range(1, kwargs['num_embeds'] + 1):
            if f"image{i}" not in kwargs:
                raise Exception(f"image{i} is required")
            kwargs[f"mask{i}"] = kwargs[f"mask{i}"] if f"mask{i}" in kwargs else None
            kwargs[f"weight{i}"] = kwargs[f"weight{i}"] if f"weight{i}" in kwargs else 1.0

            pos, neg = encoder_cls().encode(ipadapter, kwargs[f"image{i}"], kwargs[f"weight{i}"], kwargs[f"mask{i}"], clip_vision=clip_vision)
            pos_embeds.append(pos)
            neg_embeds.append(neg)

        pos_embeds = self.batch(pos_embeds, kwargs['combine_method'])
        neg_embeds = self.batch(neg_embeds, kwargs['combine_method'])

        return (model,clip_vision, ipadapter, pos_embeds, neg_embeds)

class ipadapterApplyEmbeds(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "ipadapter": ("IPADAPTER",),
                "pos_embed": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (weight_types,),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
            },

            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, ipadapter, clip_vision, pos_embed, weight, weight_type, start_at, end_at, embeds_scaling, attn_mask=None, neg_embed=None,):
        if "IPAdapterEmbeds" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()

        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterEmbeds"]
        model, image = cls().apply_ipadapter(model, ipadapter, pos_embed, weight, weight_type, start_at, end_at, neg_embed=neg_embed, attn_mask=attn_mask, clip_vision=clip_vision, embeds_scaling=embeds_scaling)

        return (model, ipadapter)

class ipadapterApplyRegional(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "positive": ("STRING", {"default": "", "placeholder": "positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "negative",  "multiline": True}),
                "image_weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 3.0, "step": 0.05}),
                "prompt_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "weight_type": (weight_types,),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },

            "optional": {
                "mask": ("MASK",),
                "optional_ipadapter_params": ("IPADAPTER_PARAMS",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "IPADAPTER_PARAMS", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "ipadapter_params", "positive", "negative")
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, pipe, image, positive, negative, image_weight, prompt_weight, weight_type, start_at, end_at, mask=None, optional_ipadapter_params=None, prompt=None, my_unique_id=None):
        model = pipe['model']

        if positive == '':
            positive = pipe['loader_settings']['positive']
        if negative == '':
            negative = pipe['loader_settings']['negative']

        if "clip" not in pipe or not pipe['clip']:
            if "chatglm3_model" in pipe:
                from ..modules.kolors.text_encode import chatglm3_adv_text_encode
                chatglm3_model = pipe['chatglm3_model']
                # text encode
                log_node_warn("Positive encoding...")
                positive_embeddings_final = chatglm3_adv_text_encode(chatglm3_model, positive, False)
                log_node_warn("Negative encoding...")
                negative_embeddings_final = chatglm3_adv_text_encode(chatglm3_model, negative, False)
        else:
            clip = pipe['clip']
            clip_skip = pipe['loader_settings']['clip_skip']
            a1111_prompt_style = pipe['loader_settings']['a1111_prompt_style']
            pipe_lora_stack = pipe['loader_settings']['lora_stack']
            positive_token_normalization = pipe['loader_settings']['positive_token_normalization']
            positive_weight_interpretation = pipe['loader_settings']['positive_weight_interpretation']
            negative_token_normalization = pipe['loader_settings']['negative_token_normalization']
            negative_weight_interpretation = pipe['loader_settings']['negative_weight_interpretation']

            positive_embeddings_final, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip, clip_skip, pipe_lora_stack, positive, positive_token_normalization, positive_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)
            negative_embeddings_final, negative_wildcard_prompt, model, clip = prompt_to_cond('negative', model, clip, clip_skip, pipe_lora_stack, negative, negative_token_normalization, negative_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)

        #ipadapter regional
        if "IPAdapterRegionalConditioning" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()

        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterRegionalConditioning"]
        ipadapter_params, new_positive_embeds, new_negative_embeds = cls().conditioning(image, image_weight, prompt_weight, weight_type, start_at, end_at, mask=mask, positive=positive_embeddings_final, negative=negative_embeddings_final)

        if optional_ipadapter_params is not None:
            positive_embeds = pipe['positive'] + new_positive_embeds
            negative_embeds = pipe['negative'] + new_negative_embeds
            _ipadapter_params = {
                "image": optional_ipadapter_params["image"] + ipadapter_params["image"],
                "attn_mask": optional_ipadapter_params["attn_mask"] + ipadapter_params["attn_mask"],
                "weight": optional_ipadapter_params["weight"] + ipadapter_params["weight"],
                "weight_type": optional_ipadapter_params["weight_type"] + ipadapter_params["weight_type"],
                "start_at": optional_ipadapter_params["start_at"] + ipadapter_params["start_at"],
                "end_at": optional_ipadapter_params["end_at"] + ipadapter_params["end_at"],
            }
            ipadapter_params = _ipadapter_params
            del _ipadapter_params
        else:
            positive_embeds = new_positive_embeds
            negative_embeds = new_negative_embeds

        new_pipe = {
            **pipe,
            "positive": positive_embeds,
            "negative": negative_embeds,
        }

        del pipe

        return (new_pipe, ipadapter_params, positive_embeds, negative_embeds)

class ipadapterApplyFromParams(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        normal_presets = ipa_cls.normal_presets
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (normal_presets,),
                "ipadapter_params": ("IPADAPTER_PARAMS",),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average", "max", "min"],),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"],
                               {"default": "insightface only"}),
            },

            "optional": {
                "optional_ipadapter": ("IPADAPTER",),
                "image_negative": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, preset, ipadapter_params, combine_embeds, embeds_scaling, cache_mode, optional_ipadapter=None, image_negative=None,):
        model, ipadapter = self.load_model(model, preset, 0, 'CPU', clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)
        if "IPAdapterFromParams" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()
        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterFromParams"]
        model, image = cls().apply_ipadapter(model, ipadapter, clip_vision=None, combine_embeds=combine_embeds, embeds_scaling=embeds_scaling, image_negative=image_negative, ipadapter_params=ipadapter_params)

        return (model, ipadapter)

#Apply InstantID
class instantID:

    def error(self):
        raise Exception(f"[ERROR] To use instantIDApply, you need to install 'ComfyUI_InstantID'")

    def run(self, pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps=None, mask=None, control_net=None, positive=None, negative=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        instantid_model, insightface_model, face_embeds = None, None, None
        model = pipe['model']
        # Load InstantID
        cache_key = 'instantID'
        if cache_key in backend_cache.cache:
            log_node_info("easy instantIDApply","Using InstantIDModel Cached")
            _, instantid_model = backend_cache.cache[cache_key][1]
        if "InstantIDModelLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_instant_cls = ALL_NODE_CLASS_MAPPINGS["InstantIDModelLoader"]
            instantid_model, = load_instant_cls().load_model(instantid_file)
            backend_cache.update_cache(cache_key, 'instantid', (False, instantid_model))
        else:
            self.error()
        icache_key = 'insightface-' + insightface
        if icache_key in backend_cache.cache:
            log_node_info("easy instantIDApply", f"Using InsightFaceModel {insightface} Cached")
            _, insightface_model = backend_cache.cache[icache_key][1]
        elif "InstantIDFaceAnalysis" in ALL_NODE_CLASS_MAPPINGS:
            load_insightface_cls = ALL_NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]
            insightface_model, = load_insightface_cls().load_insight_face(insightface)
            backend_cache.update_cache(icache_key, 'insightface', (False, insightface_model))
        else:
            self.error()

        # Apply InstantID
        if "ApplyInstantID" in ALL_NODE_CLASS_MAPPINGS:
            instantid_apply = ALL_NODE_CLASS_MAPPINGS['ApplyInstantID']
            if control_net is None:
                control_net = easyCache.load_controlnet(control_net_name, cn_soft_weights)
            model, positive, negative = instantid_apply().apply_instantid(instantid_model, insightface_model, control_net, image, model, positive, negative, start_at, end_at, weight=weight, ip_weight=None, cn_strength=cn_strength, noise=noise, image_kps=image_kps, mask=mask)
        else:
            self.error()

        new_pipe = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        del pipe

        return (new_pipe, model, positive, negative)

class instantIDApply(instantID):

    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required":{
                     "pipe": ("PIPE_LINE",),
                     "image": ("IMAGE",),
                     "instantid_file": (folder_paths.get_filename_list("instantid"),),
                     "insightface": (["CPU", "CUDA", "ROCM"],),
                     "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                     "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     "cn_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
                     "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                     "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, }),
                },
                "optional": {
                    "image_kps": ("IMAGE",),
                    "mask": ("MASK",),
                    "control_net": ("CONTROL_NET",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"
                },
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "model", "positive", "negative")

    FUNCTION = "apply"
    CATEGORY = "EasyUse/Adapter"


    def apply(self, pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps=None, mask=None, control_net=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        positive = pipe['positive']
        negative = pipe['negative']
        return self.run(pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps, mask, control_net, positive, negative, prompt, extra_pnginfo, my_unique_id)

#Apply InstantID Advanced
class instantIDApplyAdvanced(instantID):

    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required":{
                     "pipe": ("PIPE_LINE",),
                     "image": ("IMAGE",),
                     "instantid_file": (folder_paths.get_filename_list("instantid"),),
                     "insightface": (["CPU", "CUDA", "ROCM"],),
                     "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                     "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     "cn_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
                     "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                     "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, }),
                },
                "optional": {
                    "image_kps": ("IMAGE",),
                    "mask": ("MASK",),
                    "control_net": ("CONTROL_NET",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"
                },
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "model", "positive", "negative")

    FUNCTION = "apply_advanced"
    CATEGORY = "EasyUse/Adapter"

    def apply_advanced(self, pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps=None, mask=None, control_net=None, positive=None, negative=None, prompt=None, extra_pnginfo=None, my_unique_id=None):

        positive = positive if positive is not None else pipe['positive']
        negative = negative if negative is not None else pipe['negative']

        return self.run(pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps, mask, control_net, positive, negative, prompt, extra_pnginfo, my_unique_id)

class applyPulID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_file": (folder_paths.get_filename_list("pulid"),),
                "insightface": (["CPU", "CUDA", "ROCM"],),
                "image": ("IMAGE",),
                "method": (["fidelity", "style", "neutral"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "attn_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "run"
    CATEGORY = "EasyUse/Adapter"

    def error(self):
        raise Exception(f"[ERROR] To use pulIDApply, you need to install 'ComfyUI_PulID'")

    def run(self, model, image, pulid_file, insightface, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        pulid_model, insightface_model, eva_clip = None, None, None
        # Load PulID
        cache_key = 'pulID'
        if cache_key in backend_cache.cache:
            log_node_info("easy pulIDApply","Using InstantIDModel Cached")
            _, pulid_model = backend_cache.cache[cache_key][1]
        if "PulidModelLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_pulid_cls = ALL_NODE_CLASS_MAPPINGS["PulidModelLoader"]
            pulid_model, = load_pulid_cls().load_model(pulid_file)
            backend_cache.update_cache(cache_key, 'pulid', (False, pulid_model))
        else:
            self.error()
        # Load Insightface
        icache_key = 'insightface-' + insightface
        if icache_key in backend_cache.cache:
            log_node_info("easy pulIDApply", f"Using InsightFaceModel {insightface} Cached")
            _, insightface_model = backend_cache.cache[icache_key][1]
        elif "PulidInsightFaceLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_insightface_cls = ALL_NODE_CLASS_MAPPINGS["PulidInsightFaceLoader"]
            insightface_model, = load_insightface_cls().load_insightface(insightface)
            backend_cache.update_cache(icache_key, 'insightface', (False, insightface_model))
        else:
            self.error()
        # Load Eva clip
        ecache_key = 'eva_clip'
        if ecache_key in backend_cache.cache:
            log_node_info("easy pulIDApply", f"Using EVAClipModel Cached")
            _, eva_clip = backend_cache.cache[ecache_key][1]
        elif "PulidEvaClipLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_evaclip_cls = ALL_NODE_CLASS_MAPPINGS["PulidEvaClipLoader"]
            eva_clip, = load_evaclip_cls().load_eva_clip()
            backend_cache.update_cache(ecache_key, 'eva_clip', (False, eva_clip))
        else:
            self.error()

        # Apply PulID
        if method is not None:
            if "ApplyPulid" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['ApplyPulid']
                model, = cls().apply_pulid(model, pulid=pulid_model, eva_clip=eva_clip, face_analysis=insightface_model, image=image, weight=weight, method=method, start_at=start_at, end_at=end_at, attn_mask=attn_mask)
            else:
                self.error()
        else:
            if "ApplyPulidAdvanced" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['ApplyPulidAdvanced']
                model, = cls().apply_pulid(model, pulid=pulid_model, eva_clip=eva_clip, face_analysis=insightface_model, image=image, weight=weight, projection=projection, fidelity=fidelity, noise=noise, start_at=start_at, end_at=end_at, attn_mask=attn_mask)
            else:
                self.error()

        return (model,)

class applyPulIDADV(applyPulID):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_file": (folder_paths.get_filename_list("pulid"),),
                "insightface": (["CPU", "CUDA", "ROCM"],),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "projection": (["ortho_v2", "ortho", "none"], {"default":"ortho_v2"}),
                "fidelity": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1}),
                "noise": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "attn_mask": ("MASK",),
            },
        }



NODE_CLASS_MAPPINGS = {
    "easy loraPromptApply": applyLoraPrompt,
    "easy loraStackApply": applyLoraStack,
    "easy controlnetStackApply": applyControlnetStack,
    "easy ipadapterApply": ipadapterApply,
    "easy ipadapterApplyADV": ipadapterApplyAdvanced,
    "easy ipadapterApplyFaceIDKolors": ipadapterApplyFaceIDKolors,
    "easy ipadapterApplyEncoder": ipadapterApplyEncoder,
    "easy ipadapterApplyEmbeds": ipadapterApplyEmbeds,
    "easy ipadapterApplyRegional": ipadapterApplyRegional,
    "easy ipadapterApplyFromParams": ipadapterApplyFromParams,
    "easy ipadapterStyleComposition": ipadapterStyleComposition,
    "easy instantIDApply": instantIDApply,
    "easy instantIDApplyADV": instantIDApplyAdvanced,
    "easy pulIDApply": applyPulID,
    "easy pulIDApplyADV": applyPulIDADV,
    "easy styleAlignedBatchAlign": styleAlignedBatchAlign,
    "easy icLightApply": icLightApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy loraPromptApply": "Easy Apply LoraPrompt",
    "easy loraStackApply": "Easy Apply LoraStack",
    "easy controlnetStackApply": "Easy Apply CnetStack",
    "easy ipadapterApply": "Easy Apply IPAdapter",
    "easy ipadapterApplyADV": "Easy Apply IPAdapter (Advanced)",
    "easy ipadapterApplyFaceIDKolors": "Easy Apply IPAdapter (FaceID Kolors)",
    "easy ipadapterStyleComposition": "Easy Apply IPAdapter (StyleComposition)",
    "easy ipadapterApplyEncoder": "Easy Apply IPAdapter (Encoder)",
    "easy ipadapterApplyRegional": "Easy Apply IPAdapter (Regional)",
    "easy ipadapterApplyEmbeds": "Easy Apply IPAdapter (Embeds)",
    "easy ipadapterApplyFromParams": "Easy Apply IPAdapter (From Params)",
    "easy instantIDApply": "Easy Apply InstantID",
    "easy instantIDApplyADV": "Easy Apply InstantID (Advanced)",
    "easy pulIDApply": "Easy Apply PuLID",
    "easy pulIDApplyADV": "Easy Apply PuLID (Advanced)",
    "easy styleAlignedBatchAlign": "Easy Apply StyleAlign",
    "easy icLightApply": "Easy Apply ICLight"
}