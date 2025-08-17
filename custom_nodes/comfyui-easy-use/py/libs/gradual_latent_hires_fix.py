import torch
from torchvision.transforms.functional import gaussian_blur
from comfy.k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d, BrownianTreeNoiseSampler
from tqdm.auto import trange

@torch.no_grad()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # make upscale info
    upscale_steps = []
    step = start_step - 1
    while step < end_step - 1:
        upscale_steps.append(step)
        step += upscale_n_step
    height, width = x.shape[2:]
    upscale_shapes = [
        (int(height * (((upscale_ratio - 1) / i) + 1)), int(width * (((upscale_ratio - 1) / i) + 1)))
        for i in reversed(range(1, len(upscale_steps) + 1))
    ]
    upscale_info = {k: v for k, v in zip(upscale_steps, upscale_shapes)}

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            # Resize
            if i in upscale_info:
                x = torch.nn.functional.interpolate(x, size=upscale_info[i], mode="bicubic", align_corners=False)
                if unsharp_strength > 0:
                    blurred = gaussian_blur(x, kernel_size=unsharp_kernel_size, sigma=unsharp_sigma)
                    x = x + unsharp_strength * (x - blurred)

            noise_sampler = default_noise_sampler(x)
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x + noise * sigma_up * s_noise
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    # make upscale info
    upscale_steps = []
    step = start_step - 1
    while step < end_step - 1:
        upscale_steps.append(step)
        step += upscale_n_step
    height, width = x.shape[2:]
    upscale_shapes = [
        (int(height * (((upscale_ratio - 1) / i) + 1)), int(width * (((upscale_ratio - 1) / i) + 1)))
        for i in reversed(range(1, len(upscale_steps) + 1))
    ]
    upscale_info = {k: v for k, v in zip(upscale_steps, upscale_shapes)}

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            # Resize
            if i in upscale_info:
                x = torch.nn.functional.interpolate(x, size=upscale_info[i], mode="bicubic", align_corners=False)
                if unsharp_strength > 0:
                    blurred = gaussian_blur(x, kernel_size=unsharp_kernel_size, sigma=unsharp_sigma)
                    x = x + unsharp_strength * (x - blurred)
            noise_sampler = default_noise_sampler(x)
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x + noise * sigma_up * s_noise
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
):
    """DPM-Solver++(2M) SDE."""

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    # make upscale info
    upscale_steps = []
    step = start_step - 1
    while step < end_step - 1:
        upscale_steps.append(step)
        step += upscale_n_step
    height, width = x.shape[2:]
    upscale_shapes = [
        (int(height * (((upscale_ratio - 1) / i) + 1)), int(width * (((upscale_ratio - 1) / i) + 1)))
        for i in reversed(range(1, len(upscale_steps) + 1))
    ]
    upscale_info = {k: v for k, v in zip(upscale_steps, upscale_shapes)}

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                # Resize
                if i in upscale_info:
                    x = torch.nn.functional.interpolate(x, size=upscale_info[i], mode="bicubic", align_corners=False)
                    if unsharp_strength > 0:
                        blurred = gaussian_blur(x, kernel_size=unsharp_kernel_size, sigma=unsharp_sigma)
                        x = x + unsharp_strength * (x - blurred)
                    denoised = None  # 次ステップとサイズがあわないのでとりあえずNoneにしておく。
                noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_lcm(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    eta=None,
    s_noise=None,
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # make upscale info
    upscale_steps = []
    step = start_step - 1
    while step < end_step - 1:
        upscale_steps.append(step)
        step += upscale_n_step
    height, width = x.shape[2:]
    upscale_shapes = [
        (int(height * (((upscale_ratio - 1) / i) + 1)), int(width * (((upscale_ratio - 1) / i) + 1)))
        for i in reversed(range(1, len(upscale_steps) + 1))
    ]
    upscale_info = {k: v for k, v in zip(upscale_steps, upscale_shapes)}

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            # Resize
            if i in upscale_info:
                x = torch.nn.functional.interpolate(x, size=upscale_info[i], mode="bicubic", align_corners=False)
                if unsharp_strength > 0:
                    blurred = gaussian_blur(x, kernel_size=unsharp_kernel_size, sigma=unsharp_sigma)
                    x = x + unsharp_strength * (x - blurred)
            noise_sampler = default_noise_sampler(x)
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])

    return x
