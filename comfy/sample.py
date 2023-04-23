import torch
import comfy.model_management


def prepare_noise(latent, seed):
    """creates random noise given a LATENT and a seed"""
    latent_image = latent["samples"]
    batch_index = 0
    if "batch_index" in latent:
        batch_index = latent["batch_index"]

    generator = torch.manual_seed(seed)
    for i in range(batch_index):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    return noise

def create_mask(latent, noise):
    """creates a mask for a given LATENT and noise"""
    noise_mask = None
    device = comfy.model_management.get_torch_device()
    if "noise_mask" in latent:
        noise_mask = latent['noise_mask']
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

def load_c_nets(positive, negative):
    """loads control nets in positive and negative conditioning"""
    def get_models(cond):
        models = []
        for c in cond:
            if 'control' in c[1]:
                models += [c[1]['control']]
            if 'gligen' in c[1]:
                models += [c[1]['gligen'][1]]
        return models

    return get_models(positive) + get_models(negative)

def load_additional_models(positive, negative):
    """loads additional models in positive and negative conditioning"""
    models = load_c_nets(positive, negative)
    comfy.model_management.load_controlnet_gpu(models)
    return models

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        m.cleanup()