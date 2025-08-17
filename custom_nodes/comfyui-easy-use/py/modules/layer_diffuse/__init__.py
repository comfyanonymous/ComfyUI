#credit to huchenlei for this module
#from https://github.com/huchenlei/ComfyUI-layerdiffuse
import torch
import comfy.model_management
import comfy.lora
import copy
from typing import Optional
from enum import Enum
from comfy.utils import load_torch_file
from comfy.conds import CONDRegular
from comfy_extras.nodes_compositing import JoinImageWithAlpha
try:
    from .model import ModelPatcher, TransparentVAEDecoder, calculate_weight_adjust_channel
except:
    ModelPatcher, TransparentVAEDecoder, calculate_weight_adjust_channel = None, None, None
from .attension_sharing import AttentionSharingPatcher
from ...config import LAYER_DIFFUSION, LAYER_DIFFUSION_DIR, LAYER_DIFFUSION_VAE
from ...libs.utils import to_lora_patch_dict, get_local_filepath, get_sd_version

load_layer_model_state_dict = load_torch_file
class LayerMethod(Enum):
    FG_ONLY_ATTN = "Attention Injection"
    FG_ONLY_CONV = "Conv Injection"
    FG_TO_BLEND = "Foreground"
    FG_BLEND_TO_BG = "Foreground to Background"
    BG_TO_BLEND = "Background"
    BG_BLEND_TO_FG = "Background to Foreground"
    EVERYTHING = "Everything"

class LayerDiffuse:

    def __init__(self) -> None:
        self.vae_transparent_decoder = None
        self.frames = 1

    def get_layer_diffusion_method(self, method, has_blend_latent):
        method = LayerMethod(method)
        if method == LayerMethod.BG_TO_BLEND and has_blend_latent:
            method = LayerMethod.BG_BLEND_TO_FG
        elif method == LayerMethod.FG_TO_BLEND and has_blend_latent:
            method = LayerMethod.FG_BLEND_TO_BG
        return method

    def apply_layer_c_concat(self, cond, uncond, c_concat):
        def write_c_concat(cond):
            new_cond = []
            for t in cond:
                n = [t[0], t[1].copy()]
                if "model_conds" not in n[1]:
                    n[1]["model_conds"] = {}
                n[1]["model_conds"]["c_concat"] = CONDRegular(c_concat)
                new_cond.append(n)
            return new_cond

        return (write_c_concat(cond), write_c_concat(uncond))

    def apply_layer_diffusion(self, model, method, weight, samples, blend_samples, positive, negative, image=None, additional_cond=(None, None, None)):
        control_img: Optional[torch.TensorType] = None
        sd_version = get_sd_version(model)
        model_url = LAYER_DIFFUSION[method.value][sd_version]["model_url"]

        if image is not None:
            image = image.movedim(-1, 1)

        try:
            if hasattr(comfy.lora, "calculate_weight"):
                comfy.lora.calculate_weight = calculate_weight_adjust_channel(comfy.lora.calculate_weight)
            else:
                ModelPatcher.calculate_weight = calculate_weight_adjust_channel(ModelPatcher.calculate_weight)
        except:
            pass

        if method in [LayerMethod.FG_ONLY_CONV, LayerMethod.FG_ONLY_ATTN] and sd_version == 'sd1':
            self.frames = 1
        elif method in [LayerMethod.BG_TO_BLEND, LayerMethod.FG_TO_BLEND, LayerMethod.BG_BLEND_TO_FG, LayerMethod.FG_BLEND_TO_BG] and sd_version == 'sd1':
            self.frames = 2
            batch_size, _, height, width = samples['samples'].shape
            if batch_size % 2 != 0:
                raise Exception(f"The batch size should be a multiple of 2. 批次大小需为2的倍数")
            control_img = image
        elif method == LayerMethod.EVERYTHING and sd_version == 'sd1':
            batch_size, _, height, width = samples['samples'].shape
            self.frames = 3
            if batch_size % 3 != 0:
                raise Exception(f"The batch size should be a multiple of 3. 批次大小需为3的倍数")
        if model_url is None:
            raise Exception(f"{method.value} is not supported for {sd_version} model")

        model_path = get_local_filepath(model_url, LAYER_DIFFUSION_DIR)
        layer_lora_state_dict = load_layer_model_state_dict(model_path)
        work_model = model.clone()
        if sd_version == 'sd1':
            patcher = AttentionSharingPatcher(
                work_model, self.frames, use_control=control_img is not None
            )
            patcher.load_state_dict(layer_lora_state_dict, strict=True)
            if control_img is not None:
                patcher.set_control(control_img)
        else:
            layer_lora_patch_dict = to_lora_patch_dict(layer_lora_state_dict)
            work_model.add_patches(layer_lora_patch_dict, weight)

        # cond_contact
        if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV]:
            samp_model = work_model
        elif sd_version == 'sdxl':
            if method in [LayerMethod.BG_TO_BLEND, LayerMethod.FG_TO_BLEND]:
                c_concat = model.model.latent_format.process_in(samples["samples"])
            else:
                c_concat = model.model.latent_format.process_in(torch.cat([samples["samples"], blend_samples["samples"]], dim=1))
            samp_model, positive, negative = (work_model,) + self.apply_layer_c_concat(positive, negative, c_concat)
        elif sd_version == 'sd1':
            if method in [LayerMethod.BG_TO_BLEND, LayerMethod.BG_BLEND_TO_FG]:
                additional_cond = (additional_cond[0], None)
            elif method in [LayerMethod.FG_TO_BLEND, LayerMethod.FG_BLEND_TO_BG]:
                additional_cond = (additional_cond[1], None)

            work_model.model_options.setdefault("transformer_options", {})
            work_model.model_options["transformer_options"]["cond_overwrite"] = [
                cond[0][0] if cond is not None else None
                for cond in additional_cond
            ]
            samp_model = work_model

        return samp_model, positive, negative

    def join_image_with_alpha(self, image, alpha):
        out = image.movedim(-1, 1)
        if out.shape[1] == 3:  # RGB
            out = torch.cat([out, torch.ones_like(out[:, :1, :, :])], dim=1)
        for i in range(out.shape[0]):
            out[i, 3, :, :] = alpha
        return out.movedim(1, -1)

    def image_to_alpha(self, image, latent):
        pixel = image.movedim(-1, 1)  # [B, H, W, C] => [B, C, H, W]
        decoded = []
        sub_batch_size = 16
        for start_idx in range(0, latent.shape[0], sub_batch_size):
            decoded.append(
                self.vae_transparent_decoder.decode_pixel(
                    pixel[start_idx: start_idx + sub_batch_size],
                    latent[start_idx: start_idx + sub_batch_size],
                )
            )
        pixel_with_alpha = torch.cat(decoded, dim=0)
        # [B, C, H, W] => [B, H, W, C]
        pixel_with_alpha = pixel_with_alpha.movedim(1, -1)
        image = pixel_with_alpha[..., 1:]
        alpha = pixel_with_alpha[..., 0]

        alpha = 1.0 - alpha
        new_images, = JoinImageWithAlpha().join_image_with_alpha(image, alpha)
        return new_images, alpha

    def make_3d_mask(self, mask):
        if len(mask.shape) == 4:
            return mask.squeeze(0)

        elif len(mask.shape) == 2:
            return mask.unsqueeze(0)

        return mask

    def masks_to_list(self, masks):
        if masks is None:
            empty_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            return ([empty_mask],)

        res = []

        for mask in masks:
            res.append(mask)

        return [self.make_3d_mask(x) for x in res]

    def layer_diffusion_decode(self, layer_diffusion_method, latent, blend_samples, samp_images, model):
        alpha = []
        if layer_diffusion_method is not None:
            sd_version = get_sd_version(model)
            if sd_version not in ['sdxl', 'sd1']:
                raise Exception(f"Only SDXL and SD1.5 model supported for Layer Diffusion")
            method = self.get_layer_diffusion_method(layer_diffusion_method, blend_samples is not None)
            sd15_allow = True if sd_version == 'sd1' and method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.EVERYTHING, LayerMethod.BG_TO_BLEND, LayerMethod.BG_BLEND_TO_FG] else False
            sdxl_allow = True if sd_version == 'sdxl' and method in [LayerMethod.FG_ONLY_CONV, LayerMethod.FG_ONLY_ATTN, LayerMethod.BG_BLEND_TO_FG] else False
            if sdxl_allow or sd15_allow:
                if self.vae_transparent_decoder is None:
                    model_url = LAYER_DIFFUSION_VAE['decode'][sd_version]["model_url"]
                    if model_url is None:
                        raise Exception(f"{method.value} is not supported for {sd_version} model")
                    decoder_file = get_local_filepath(model_url, LAYER_DIFFUSION_DIR)
                    self.vae_transparent_decoder = TransparentVAEDecoder(
                        load_torch_file(decoder_file),
                        device=comfy.model_management.get_torch_device(),
                        dtype=(torch.float16 if comfy.model_management.should_use_fp16() else torch.float32),
                    )
                if method in [LayerMethod.EVERYTHING, LayerMethod.BG_BLEND_TO_FG, LayerMethod.BG_TO_BLEND]:
                    new_images = []
                    sliced_samples = copy.copy({"samples": latent})
                    for index in range(len(samp_images)):
                        if index % self.frames == 0:
                            img = samp_images[index::self.frames]
                            alpha_images, _alpha = self.image_to_alpha(img, sliced_samples["samples"][index::self.frames])
                            alpha.append(self.make_3d_mask(_alpha[0]))
                            new_images.append(alpha_images[0])
                        else:
                            new_images.append(samp_images[index])
                else:
                    new_images, alpha = self.image_to_alpha(samp_images, latent)
            else:
                new_images = samp_images
        else:
            new_images = samp_images


        return (new_images, samp_images, alpha)