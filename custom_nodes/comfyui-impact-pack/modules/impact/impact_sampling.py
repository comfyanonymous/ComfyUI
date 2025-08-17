import logging

import nodes
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import samplers
from comfy_extras import nodes_custom_sampler
import latent_preview
import comfy
import torch
import math
import comfy.model_management as mm


try:
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
    import node_helpers
except Exception:
    logging.warning("\n#############################################\n[Impact Pack] ComfyUI is an outdated version.\n#############################################\n")
    raise Exception("[Impact Pack] ComfyUI is an outdated version.")


def calculate_sigmas(model, sampler, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
        steps += 1
        discard_penultimate_sigma = True

    if scheduler.startswith('AYS'):
        sigmas = nodes.NODE_CLASS_MAPPINGS['AlignYourStepsScheduler']().get_sigmas(scheduler[4:], steps, denoise=1.0)[0]
    elif scheduler.startswith('GITS[coeff='):
        sigmas = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']().get_sigmas(float(scheduler[11:-1]), steps, denoise=1.0)[0]
    elif scheduler == 'LTXV[default]':
        sigmas = nodes.NODE_CLASS_MAPPINGS['LTXVScheduler']().get_sigmas(20, 2.05, 0.95, True, 0.1)[0]
    elif scheduler.startswith('OSS'):
        sigmas = nodes.NODE_CLASS_MAPPINGS['OptimalStepsScheduler']().get_sigmas(scheduler[4:], steps, denoise=1.0)[0]
    else:
        sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if 'extra_args' in kwargs and 'seed' in kwargs['extra_args']:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs['extra_args'].get("seed", None)
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def ksampler(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name in ["dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu"]:
        if sampler_name == "dpmpp_sde":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_sde
        elif sampler_name == "dpmpp_sde_gpu":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_sde_gpu
        elif sampler_name == "dpmpp_2m_sde":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_2m_sde
        elif sampler_name == "dpmpp_2m_sde_gpu":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_2m_sde_gpu
        elif sampler_name == "dpmpp_3m_sde":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_3m_sde
        elif sampler_name == "dpmpp_3m_sde_gpu":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_3m_sde_gpu

        def sampler_function_wrapper(model, x, sigmas, **kwargs):
            if 'noise_sampler' not in kwargs:
                kwargs['noise_sampler'] = get_noise_sampler(x, 'gpu' not in sampler_name, total_sigmas, **kwargs)

            return orig_sampler_function(model, x, sigmas, **kwargs)

        sampler_function = sampler_function_wrapper

    else:
        return comfy.samplers.sampler_object(sampler_name)

    return samplers.KSAMPLER(sampler_function, extra_options, inpaint_options)


# modified version of SamplerCustom.sample
def sample_with_custom_noise(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, noise=None, callback=None):
    latent = latent_image
    latent_image = latent["samples"]

    if hasattr(comfy.sample, 'fix_empty_latent_channels'):
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    out = latent.copy()
    out['samples'] = latent_image

    if noise is None:
        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(out)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    preview_callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    if callback is not None:
        def touched_callback(step, x0, x, total_steps):
            callback(step, x0, x, total_steps)
            preview_callback(step, x0, x, total_steps)
    else:
        touched_callback = preview_callback

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    device = mm.get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    if negative != 'NegativePlaceholder':
        # This way is incompatible with Advanced ControlNet, yet.
        # guider = comfy.samplers.CFGGuider(model)
        # guider.set_conds(positive, negative)
        # guider.set_cfg(cfg)
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                                             noise_mask=noise_mask, callback=touched_callback,
                                             disable_pbar=disable_pbar, seed=noise_seed)
    else:
        guider = nodes_custom_sampler.Guider_Basic(model)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": cfg})
        guider.set_conds(positive)
        samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=touched_callback, disable_pbar=disable_pbar, seed=noise_seed)

    samples = samples.to(comfy.model_management.intermediate_device())

    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return out, out_denoised


# When sampling one step at a time, it mitigates the problem. (especially for _sde series samplers)
def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                     latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0, sampler_opt=None, noise=None, callback=None, scheduler_func=None):

    if scheduler_func is not None:
        total_sigmas = scheduler_func(model, sampler_name, steps)
    else:
        if sampler_opt is None:
            total_sigmas = calculate_sigmas(model, sampler_name, scheduler, steps)
        else:
            total_sigmas = calculate_sigmas(model, "", scheduler, steps)

    sigmas = total_sigmas

    if end_at_step is not None and end_at_step < (len(total_sigmas) - 1):
        sigmas = total_sigmas[:end_at_step + 1]
        if not return_with_leftover_noise:
            sigmas[-1] = 0

    if start_at_step is not None:
        if start_at_step < (len(sigmas) - 1):
            sigmas = sigmas[start_at_step:] * sigma_ratio
        else:
            if latent_image is not None:
                return latent_image
            else:
                return {'samples': torch.zeros_like(noise)}

    if sampler_opt is None:
        impact_sampler = ksampler(sampler_name, total_sigmas)
    else:
        impact_sampler = sampler_opt

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image

    res = sample_with_custom_noise(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image, noise=noise, callback=callback)

    if return_with_leftover_noise:
        return res[0]
    else:
        return res[1]


def impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, sigma_ratio=1.0, sampler_opt=None, noise=None, scheduler_func=None):
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps
    return separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                            start_at_step, end_at_step, False, scheduler_func=scheduler_func)


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, sigma_factor=1.0, noise=None, scheduler_func=None, sampler_opt=None):

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        # Use separated_sample instead of KSampler for `AYS scheduler`
        # refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise * sigma_factor)[0]

        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + steps

        refined_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, start_at_step, end_at_step, False,
                                          sigma_ratio=sigma_factor, sampler_opt=sampler_opt, noise=noise, scheduler_func=scheduler_func)
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        # print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                       positive, negative, latent_image, start_at_step, end_at_step, True,
                                       sigma_ratio=sigma_factor, sampler_opt=sampler_opt, noise=noise, scheduler_func=scheduler_func)

        if 'noise_mask' in latent_image:
            # noise_latent = \
            #     impact_sampling.separated_sample(refiner_model, "enable", seed, advanced_steps, cfg, sampler_name,
            #                                      scheduler, refiner_positive, refiner_negative, latent_image, end_at_step,
            #                                      end_at_step, "enable")

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        # print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = separated_sample(refiner_model, False, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, False,
                                          sigma_ratio=sigma_factor, sampler_opt=sampler_opt, scheduler_func=scheduler_func)

    return refined_latent


class KSamplerAdvancedWrapper:
    params = None

    def __init__(self, model, cfg, sampler_name, scheduler, positive, negative, sampler_opt=None, sigma_factor=1.0, scheduler_func=None):
        self.params = model, cfg, sampler_name, scheduler, positive, negative, sigma_factor
        self.sampler_opt = sampler_opt
        self.scheduler_func = scheduler_func

    def clone_with_conditionings(self, positive, negative):
        model, cfg, sampler_name, scheduler, _, _, _ = self.params
        return KSamplerAdvancedWrapper(model, cfg, sampler_name, scheduler, positive, negative, self.sampler_opt)

    def sample_advanced(self, add_noise, seed, steps, latent_image, start_at_step, end_at_step, return_with_leftover_noise, hook=None,
                        recovery_mode="ratio additional", recovery_sampler="AUTO", recovery_sigma_ratio=1.0, noise=None):

        model, cfg, sampler_name, scheduler, positive, negative, sigma_factor = self.params
        # steps, start_at_step, end_at_step = self.compensate_denoise(steps, start_at_step, end_at_step)

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent = hook.pre_ksample_advanced(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                                                                                              positive, negative, latent_image, start_at_step, end_at_step,
                                                                                                                              return_with_leftover_noise)

        if recovery_mode != 'DISABLE' and sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu']:
            base_image = latent_image.copy()
            if recovery_mode == "ratio between":
                sigma_ratio = 1.0 - recovery_sigma_ratio
            else:
                sigma_ratio = 1.0
        else:
            base_image = None
            sigma_ratio = 1.0

        try:
            if sigma_ratio > 0:
                latent_image = separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                positive, negative, latent_image, start_at_step, end_at_step,
                                                return_with_leftover_noise, sigma_ratio=sigma_ratio * sigma_factor,
                                                sampler_opt=self.sampler_opt, noise=noise, scheduler_func=self.scheduler_func)
        except ValueError as e:
            if str(e) == 'sigma_min and sigma_max must not be 0':
                logging.warning("\nWARN: sampling skipped - sigma_min and sigma_max are 0")
                return latent_image

        if (recovery_sigma_ratio > 0 and recovery_mode != 'DISABLE' and
                sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu']):
            compensate = 0 if sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu'] else 2
            if recovery_sampler == "AUTO":
                recovery_sampler = 'dpm_fast' if sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu'] else 'dpmpp_2m'

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()

            noise_mask = latent_image['noise_mask']

            if len(noise_mask.shape) == 4:
                noise_mask = noise_mask.squeeze(0).squeeze(0)

            latent_image = latent_compositor.composite(base_image, latent_image, 0, 0, False, noise_mask)[0]

            try:
                latent_image = separated_sample(model, add_noise, seed, steps, cfg, recovery_sampler, scheduler,
                                                positive, negative, latent_image, start_at_step-compensate, end_at_step, return_with_leftover_noise,
                                                sigma_ratio=recovery_sigma_ratio * sigma_factor, sampler_opt=self.sampler_opt, scheduler_func=self.scheduler_func)
            except ValueError as e:
                if str(e) == 'sigma_min and sigma_max must not be 0':
                    logging.warning("\nWARN: sampling skipped - sigma_min and sigma_max are 0")

        return latent_image


class KSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=None):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.scheduler_func = scheduler_func

    def sample(self, latent_image, hook=None):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

        return impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, scheduler_func=self.scheduler_func)
