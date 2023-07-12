import torch
import comfy.model_management
import comfy.samplers
import math
import numpy as np

def prepare_noise(latent_image, seed, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = noise_mask.round()
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    if noise_mask.shape[0] < shape[0]:
        noise_mask = noise_mask.repeat(math.ceil(shape[0] / noise_mask.shape[0]), 1, 1, 1)[:shape[0]]
    noise_mask = noise_mask.to(device)
    return noise_mask

def broadcast_cond(cond, batch, device):
    """broadcasts conditioning to the batch size"""
    copy = []
    for p in cond:
        t = p[0]
        if t.shape[0] < batch:
            t = torch.cat([t] * batch)
        t = t.to(device)
        copy += [[t] + p[1:]]
    return copy

def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c[1]:
            models += [c[1][model_type]]
    return models

def load_additional_models(positive, negative, dtype):
    """loads additional models in positive and negative conditioning"""
    control_nets = get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")
    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1].to(dtype) for x in gligen]
    models = control_nets + gligen
    comfy.model_management.load_controlnet_gpu(models)
    return models

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        m.cleanup()

def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    device = comfy.model_management.get_torch_device()

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    real_model = None
    comfy.model_management.load_model_gpu(model)
    real_model = model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())

    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.cpu()

    cleanup_additional_models(models)
    return samples
