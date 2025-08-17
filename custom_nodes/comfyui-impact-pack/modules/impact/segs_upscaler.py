from impact import impact_sampling
from comfy import model_management
from impact import utils
from PIL import Image
import nodes
import torch
import inspect
import logging
import comfy

try:
    from comfy_extras import nodes_differential_diffusion
except Exception:
    logging.info("[Impact Pack] ComfyUI is an outdated version. The DifferentialDiffusion feature will be disabled.")


# Implementation based on `https://github.com/lingondricka2/Upscaler-Detailer`

# code from comfyroll --->
# https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/blob/main/nodes/functions_upscale.py

def upscale_with_model(upscale_model, image):
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
    return s


def apply_resize_image(image: Image.Image, original_width, original_height, rounding_modulus, mode='scale', supersample='true', factor: int = 2, width: int = 1024, height: int = 1024,
                       resample='bicubic'):
    # Calculate the new width and height based on the given mode and parameters
    if mode == 'rescale':
        new_width, new_height = int(original_width * factor), int(original_height * factor)
    else:
        m = rounding_modulus
        original_ratio = original_height / original_width
        height = int(width * original_ratio)

        new_width = width if width % m == 0 else width + (m - width % m)
        new_height = height if height % m == 0 else height + (m - height % m)

    # Define a dictionary of resampling filters
    resample_filters = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'lanczos': 1}

    # Apply supersample
    if supersample == 'true':
        image = image.resize((new_width * 8, new_height * 8), resample=Image.Resampling(resample_filters[resample]))

    # Resize the image using the given resampling filter
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))

    return resized_image


def upscaler(image, upscale_model, rescale_factor, resampling_method, supersample, rounding_modulus):
    if upscale_model is not None:
        up_image = upscale_with_model(upscale_model, image)
    else:
        up_image = image

    pil_img = utils.tensor2pil(image)
    original_width, original_height = pil_img.size
    scaled_image = utils.pil2tensor(apply_resize_image(utils.tensor2pil(up_image), original_width, original_height, rounding_modulus, 'rescale',
                                                 supersample, rescale_factor, 1024, resampling_method))
    return scaled_image

# <---


def img2img_segs(image, model, clip, vae, seed, steps, cfg, sampler_name, scheduler,
                 positive, negative, denoise, noise_mask, control_net_wrapper=None,
                 inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

    original_image_size = image.shape[1:3]

    # Match to original image size
    if original_image_size[0] % 8 > 0 or original_image_size[1] % 8 > 0:
        scale = 8/min(original_image_size[0], original_image_size[1]) + 1
        w = int(original_image_size[1] * scale)
        h = int(original_image_size[0] * scale)
        image = utils.tensor_resize(image, w, h)

    if noise_mask is not None:
        noise_mask = utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

        if noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().apply(model)[0]

    if control_net_wrapper is not None:
        positive, negative, _ = control_net_wrapper.apply(positive, negative, image, noise_mask)

    # prepare mask
    if noise_mask is not None and inpaint_model:
        imc_encode = nodes.InpaintModelConditioning().encode
        if 'noise_mask' in inspect.signature(imc_encode).parameters:
            positive, negative, latent_image = imc_encode(positive, negative, image, vae, mask=noise_mask, noise_mask=True)
        else:
            logging.info("[Impact Pack] ComfyUI is an outdated version.")
            positive, negative, latent_image = imc_encode(positive, negative, image, vae, noise_mask)
    else:
        latent_image = utils.to_latent_image(image, vae)
        if noise_mask is not None:
            latent_image['noise_mask'] = noise_mask

    refined_latent = latent_image

    # ksampler
    refined_latent = impact_sampling.ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, refined_latent, denoise, scheduler_func=scheduler_func_opt)

    # non-latent downscale - latent downscale cause bad quality
    refined_image = vae.decode(refined_latent['samples'])

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # Match to original image size
    if refined_image.shape[1:3] != original_image_size:
        refined_image = utils.tensor_resize(refined_image, original_image_size[1], original_image_size[0])

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image
