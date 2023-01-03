import k_diffusion.sampling
import k_diffusion.external
import torch
import contextlib

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        if len(uncond[0]) == len(cond[0]) and x.shape[0] * x.shape[2] * x.shape[3] <= (96 * 96): #TODO check memory instead
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        else:
            cond = self.inner_model(x, sigma, cond=cond)
            uncond = self.inner_model(x, sigma, cond=uncond)
        return uncond + (cond - uncond) * cond_scale


def simple_scheduler(model, steps):
    sigs = []
    ss = len(model.sigmas) / steps
    for x in range(steps):
        sigs += [float(model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


class KSampler:
    SCHEDULERS = ["karras", "normal", "simple"]
    SAMPLERS = ["sample_euler", "sample_euler_ancestral", "sample_heun", "sample_dpm_2", "sample_dpm_2_ancestral",
                "sample_lms", "sample_dpm_fast", "sample_dpm_adaptive", "sample_dpmpp_2s_ancestral", "sample_dpmpp_sde",
                "sample_dpmpp_2m"]

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None):
        self.model = model
        if self.model.parameterization == "v":
            self.model_wrap = k_diffusion.external.CompVisVDenoiser(self.model, quantize=True)
        else:
            self.model_wrap = k_diffusion.external.CompVisDenoiser(self.model, quantize=True)
        self.model_k = CFGDenoiser(self.model_wrap)
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.sigma_min=float(self.model_wrap.sigmas[0])
        self.sigma_max=float(self.model_wrap.sigmas[-1])
        self.set_steps(steps, denoise)

    def _calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['sample_dpm_2', 'sample_dpm_2_ancestral']:
            steps += 1
            discard_penultimate_sigma = True

        if self.scheduler == "karras":
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device)
        elif self.scheduler == "normal":
            sigmas = self.model_wrap.get_sigmas(steps).to(self.device)
        elif self.scheduler == "simple":
            sigmas = simple_scheduler(self.model_wrap, steps).to(self.device)
        else:
            print("error invalid scheduler", self.scheduler)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None:
            self.sigmas = self._calculate_sigmas(steps)
        else:
            new_steps = int(steps/denoise)
            sigmas = self._calculate_sigmas(new_steps)
            self.sigmas = sigmas[-(steps + 1):]


    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None):
        sigmas = self.sigmas
        sigma_min = self.sigma_min

        if last_step is not None:
            sigma_min = sigmas[last_step]
            sigmas = sigmas[:last_step + 1]
        if start_step is not None:
            sigmas = sigmas[start_step:]


        noise *= sigmas[0]
        if latent_image is not None:
            noise += latent_image

        if self.model.model.diffusion_model.dtype == torch.float16:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(self.device):
            if self.sampler == "sample_dpm_fast":
                samples = k_diffusion.sampling.sample_dpm_fast(self.model_k, noise, sigma_min, sigmas[0], self.steps, extra_args={"cond":positive, "uncond":negative, "cond_scale": cfg})
            elif self.sampler == "sample_dpm_adaptive":
                samples = k_diffusion.sampling.sample_dpm_adaptive(self.model_k, noise, sigma_min, sigmas[0], extra_args={"cond":positive, "uncond":negative, "cond_scale": cfg})
            else:
                samples = getattr(k_diffusion.sampling, self.sampler)(self.model_k, noise, sigmas, extra_args={"cond":positive, "uncond":negative, "cond_scale": cfg})
        return samples.to(torch.float32)
