import k_diffusion.sampling
import k_diffusion.external
import torch
import contextlib

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        if len(uncond[0]) == len(cond[0]) and x.shape[0] * x.shape[2] * x.shape[3] < (96 * 96): #TODO check memory instead
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        else:
            cond = self.inner_model(x, sigma, cond=cond)
            uncond = self.inner_model(x, sigma, cond=uncond)
        return uncond + (cond - uncond) * cond_scale

class CFGDenoiserComplex(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale):
        def calc_cond(cond, x_in, sigma):
            out_cond = torch.zeros_like(x_in)
            out_count = torch.ones_like(x_in)/100000.0
            sigma_cmp = sigma[0]

            for x in cond:
                area = (x_in.shape[2], x_in.shape[3], 0, 0)
                strength = 1.0
                min_sigma = 0.0
                max_sigma = 999.0
                if 'area' in x[1]:
                    area = x[1]['area']
                if 'strength' in x[1]:
                    strength = x[1]['strength']
                if 'min_sigma' in x[1]:
                    min_sigma = x[1]['min_sigma']
                if 'max_sigma' in x[1]:
                    max_sigma = x[1]['max_sigma']
                if sigma_cmp < min_sigma or sigma_cmp > max_sigma:
                    continue
                input_x = x_in[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
                mult = torch.ones_like(input_x) * strength

                rr = 8
                if area[2] != 0:
                    for t in range(rr):
                        mult[:,:,area[2]+t:area[2]+1+t,:] *= ((1.0/rr) * (t + 1))
                if (area[0] + area[2]) < x_in.shape[2]:
                    for t in range(rr):
                        mult[:,:,area[0] + area[2] - 1 - t:area[0] + area[2] - t,:] *= ((1.0/rr) * (t + 1))
                if area[3] != 0:
                    for t in range(rr):
                        mult[:,:,:,area[3]+t:area[3]+1+t] *= ((1.0/rr) * (t + 1))
                if (area[1] + area[3]) < x_in.shape[3]:
                    for t in range(rr):
                        mult[:,:,:,area[1] + area[3] - 1 - t:area[1] + area[3] - t] *= ((1.0/rr) * (t + 1))

                out_cond[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] += self.inner_model(input_x, sigma, cond=x[0]) * mult
                out_count[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] += mult
                del input_x
                del mult

            out_cond /= out_count
            del out_count
            return out_cond

        cond = calc_cond(cond, x, sigma)
        uncond = calc_cond(uncond, x, sigma)

        return uncond + (cond - uncond) * cond_scale

def simple_scheduler(model, steps):
    sigs = []
    ss = len(model.sigmas) / steps
    for x in range(steps):
        sigs += [float(model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c[1]:
        return

    c_area = c[1]['area']
    smallest = None
    for x in conds:
        if 'area' in x[1]:
            a = x[1]['area']
            if c_area[2] >= a[2] and c_area[3] >= a[3]:
                if a[0] + a[2] >= c_area[0] + c_area[2]:
                    if a[1] + a[3] >= c_area[1] + c_area[3]:
                        if smallest is None:
                            smallest = x
                        elif 'area' not in smallest[1]:
                            smallest = x
                        else:
                            if smallest[1]['area'][0] * smallest[1]['area'][1] > a[0] * a[1]:
                                smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest[1]:
        if smallest[1]['area'] == c_area:
            return
    n = c[1].copy()
    conds += [[smallest[0], n]]

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
        self.model_k = CFGDenoiserComplex(self.model_wrap)
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


    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False):
        sigmas = self.sigmas
        sigma_min = self.sigma_min

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigma_min = sigmas[last_step]
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        noise *= sigmas[0]
        if latent_image is not None:
            noise += latent_image

        positive = positive[:]
        negative = negative[:]
        #make sure each cond area has an opposite one with the same area
        for c in positive:
            create_cond_with_same_area_if_none(negative, c)
        for c in negative:
            create_cond_with_same_area_if_none(positive, c)

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
