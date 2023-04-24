import torch
import comfy.model_management


def prepare_noise(latent_image, seed, skip=0):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    for _ in range(skip):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    return noise

def prepare_mask(noise_mask, noise):
    """ensures noise mask is of proper dimensions"""
    device = comfy.model_management.get_torch_device()
    noise_mask = torch.nn.functional.interpolate(noise_mask[None,None,], size=(noise.shape[2], noise.shape[3]), mode="bilinear")
    noise_mask = noise_mask.round()
    noise_mask = torch.cat([noise_mask] * noise.shape[1], dim=1)
    noise_mask = torch.cat([noise_mask] * noise.shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask

def broadcast_cond(cond, noise):
    """broadcasts conditioning to the noise batch size"""
    device = comfy.model_management.get_torch_device()
    copy = []
    for p in cond:
        t = p[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        t = t.to(device)
        copy += [[t] + p[1:]]
    return copy

def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c[1]:
            models += [c[1][model_type]]
    return models

def load_additional_models(positive, negative):
    """loads additional models in positive and negative conditioning"""
    control_nets = get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")
    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1] for x in gligen]
    models = control_nets + gligen
    comfy.model_management.load_controlnet_gpu(models)
    return models

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        m.cleanup()