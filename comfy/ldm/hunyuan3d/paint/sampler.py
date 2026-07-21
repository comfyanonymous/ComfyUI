# Torch-native multiview diffusion driver for the Hunyuan3D 2.1 paint UNet.
# Reimplements the diffusers DDIM (v-prediction, zero-terminal-SNR, trailing) schedule
# and the reference pipeline's triple-batch, view-scaled classifier-free guidance loop
# without any diffusers dependency.

import numpy as np
import torch
from einops import rearrange

# SD-2.x VAE scaling factor (the released paint model is built on stable-diffusion-2-1).
SD_SCALING_FACTOR = 0.18215


def rescale_zero_terminal_snr(alphas_cumprod):
    """Force the terminal alpha_cumprod to 0 (diffusers rescale_zero_terminal_snr)."""
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    a0 = alphas_bar_sqrt[0].clone()
    aT = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= aT
    alphas_bar_sqrt *= a0 / (a0 - aT)
    return alphas_bar_sqrt ** 2


class DDIMVScheduler:
    """DDIM scheduler with v-prediction, scaled-linear betas, zero-terminal-SNR and
    trailing timestep spacing - matching hunyuan3d-paintpbr-v2-1/scheduler_config.json."""

    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, device="cpu"):
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = rescale_zero_terminal_snr(alphas_cumprod)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.final_alpha_cumprod = torch.tensor(1.0, dtype=torch.float64, device=device)  # set_alpha_to_one
        self.num_train_timesteps = num_train_timesteps
        self.init_noise_sigma = 1.0
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device="cpu"):
        step_ratio = self.num_train_timesteps / num_inference_steps
        timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64) - 1  # trailing
        self.timesteps = torch.from_numpy(timesteps).to(device)
        return self.timesteps

    def step(self, model_output, step_index, sample):
        t = int(self.timesteps[step_index])
        prev_t = int(self.timesteps[step_index + 1]) if step_index + 1 < len(self.timesteps) else -1

        out_dtype = model_output.dtype
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        model_output = model_output.double()
        sample = sample.double()
        # v-prediction
        pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        pred_eps = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        # DDIM step, eta = 0
        prev_sample = alpha_prod_prev.sqrt() * pred_x0 + (1 - alpha_prod_prev).sqrt() * pred_eps
        return prev_sample.to(out_dtype)


def _cam_mapping(azim):
    azim = float(azim) % 360.0
    if 0 <= azim < 90:
        return azim / 90.0 + 1.0
    elif 90 <= azim < 330:
        return 2.0
    else:
        return -azim / 90.0 + 5.0


@torch.no_grad()
def generate_multiview(model, config, ref_latent, normal_latents, position_latents, position_maps,
                       dino_features=None, camera_azims=None, num_inference_steps=15, guidance_scale=3.0,
                       seed=0, device="cpu", dtype=torch.float32):
    """Run the multiview PBR diffusion loop.

    Shapes (V = number of views):
        ref_latent       (1, C, H, W)   VAE latent of the reference image (SD-scaled)
        normal_latents   (V, C, H, W)   VAE latents of the world-space normal maps
        position_latents (V, C, H, W)   VAE latents of the position maps
        position_maps    (V, 3, Hp, Wp) raw position maps in [0, 1] (for PoseRoPE)
        dino_features    (1, L, D) | None  precomputed DINOv2 tokens for the reference

    Returns a dict mapping each pbr token (e.g. "albedo", "mr") to a (V, C, H, W) latent.
    """
    pbr_setting = list(config["pbr_setting"])
    n_pbr = len(pbr_setting)
    V = normal_latents.shape[0]
    C, H, W = ref_latent.shape[1:]

    if camera_azims is None:
        camera_azims = [0] * V

    scheduler = DDIMVScheduler(device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)

    # learned material text-clip embeddings -> (1, n_pbr, tokens, cross_dim)
    tokens = [getattr(model.unet, f"learned_text_clip_{t}").to(device=device, dtype=dtype) for t in pbr_setting]
    prompt_embeds = torch.stack(tokens, dim=0).unsqueeze(0)  # (1, n_pbr, 77, cross)

    do_cfg = guidance_scale > 1.0
    n_cfg = 3 if do_cfg else 1

    enc = prompt_embeds.repeat(n_cfg, 1, 1, 1)
    ref_cfg = ref_latent.unsqueeze(1).repeat(n_cfg, 1, 1, 1, 1)  # (n_cfg,1,C,H,W)
    normal_cfg = normal_latents.unsqueeze(0).repeat(n_cfg, 1, 1, 1, 1)
    position_cfg = position_latents.unsqueeze(0).repeat(n_cfg, 1, 1, 1, 1)
    posmap_cfg = position_maps.unsqueeze(0).repeat(n_cfg, 1, 1, 1, 1) if position_maps is not None else None

    ref_scale = torch.as_tensor([0.0, 1.0, 1.0], device=device, dtype=dtype) if do_cfg else 1.0

    dino_cfg = None
    if dino_features is not None:
        z = torch.zeros_like(dino_features)
        dino_cfg = torch.cat([z, z, dino_features], dim=0) if do_cfg else dino_features
        dino_cfg = dino_cfg.to(device=device, dtype=dtype)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn(n_pbr * V, C, H, W, generator=generator).to(device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    view_scale = torch.tensor([_cam_mapping(a) for a in camera_azims], device=device, dtype=dtype)
    view_scale = view_scale.repeat(n_pbr)[:, None, None, None]  # (n_pbr*V,1,1,1)

    cache = {}
    for i in range(len(scheduler.timesteps)):
        t = scheduler.timesteps[i]
        sample = rearrange(latents, "(n_pbr n) c h w -> n_pbr n c h w", n_pbr=n_pbr, n=V)
        sample = sample.unsqueeze(0).repeat(n_cfg, 1, 1, 1, 1, 1)  # (n_cfg,n_pbr,V,C,H,W)

        noise_pred = model(sample, t, enc, dino_hidden_states=dino_cfg, ref_latents=ref_cfg,
                           embeds_normal=normal_cfg, embeds_position=position_cfg, position_maps=posmap_cfg,
                           mva_scale=1.0, ref_scale=ref_scale, cache=cache)

        if do_cfg:
            noise_pred = rearrange(noise_pred, "(n_cfg m) c h w -> n_cfg m c h w", n_cfg=3)
            uncond, ref, full = noise_pred[0], noise_pred[1], noise_pred[2]
            noise_pred = uncond + guidance_scale * view_scale * (ref - uncond)
            noise_pred = noise_pred + guidance_scale * view_scale * (full - ref)
        latents = scheduler.step(noise_pred, i, latents)

    return {pbr_setting[p]: latents[p * V:(p + 1) * V] for p in range(n_pbr)}
