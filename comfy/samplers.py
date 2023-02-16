from .k_diffusion import sampling as k_diffusion_sampling
from .k_diffusion import external as k_diffusion_external
from .extra_samplers import uni_pc
import torch
import contextlib
import model_management

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

def sampling_function(model_function, x, sigma, uncond, cond, cond_scale, cond_concat=None):
        def get_area_and_mult(cond, x_in, cond_concat_in):
            area = (x_in.shape[2], x_in.shape[3], 0, 0)
            strength = 1.0
            min_sigma = 0.0
            max_sigma = 999.0
            if 'area' in cond[1]:
                area = cond[1]['area']
            if 'strength' in cond[1]:
                strength = cond[1]['strength']

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
            conditionning = {}
            conditionning['c_crossattn'] = cond[0]
            if cond_concat_in is not None and len(cond_concat_in) > 0:
                cropped = []
                for x in cond_concat_in:
                    cr = x[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
                    cropped.append(cr)
                conditionning['c_concat'] = torch.cat(cropped, dim=1)
            return (input_x, mult, conditionning, area)

        def cond_equal_size(c1, c2):
            if c1.keys() != c2.keys():
                return False
            if 'c_crossattn' in c1:
                if c1['c_crossattn'].shape != c2['c_crossattn'].shape:
                    return False
            if 'c_concat' in c1:
                if c1['c_concat'].shape != c2['c_concat'].shape:
                    return False
            return True

        def cond_cat(c_list):
            c_crossattn = []
            c_concat = []
            for x in c_list:
                if 'c_crossattn' in x:
                    c_crossattn.append(x['c_crossattn'])
                if 'c_concat' in x:
                    c_concat.append(x['c_concat'])
            out = {}
            if len(c_crossattn) > 0:
                out['c_crossattn'] = [torch.cat(c_crossattn)]
            if len(c_concat) > 0:
                out['c_concat'] = [torch.cat(c_concat)]
            return out

        def calc_cond_uncond_batch(model_function, cond, uncond, x_in, sigma, max_total_area, cond_concat_in):
            out_cond = torch.zeros_like(x_in)
            out_count = torch.ones_like(x_in)/100000.0

            out_uncond = torch.zeros_like(x_in)
            out_uncond_count = torch.ones_like(x_in)/100000.0

            COND = 0
            UNCOND = 1

            to_run = []
            for x in cond:
                p = get_area_and_mult(x, x_in, cond_concat_in)
                if p is None:
                    continue

                to_run += [(p, COND)]
            for x in uncond:
                p = get_area_and_mult(x, x_in, cond_concat_in)
                if p is None:
                    continue

                to_run += [(p, UNCOND)]

            while len(to_run) > 0:
                first = to_run[0]
                first_shape = first[0][0].shape
                to_batch_temp = []
                for x in range(len(to_run)):
                    if to_run[x][0][0].shape == first_shape:
                        if cond_equal_size(to_run[x][0][2], first[0][2]):
                            to_batch_temp += [x]

                to_batch_temp.reverse()
                to_batch = to_batch_temp[:1]

                for i in range(1, len(to_batch_temp) + 1):
                    batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                    if (len(batch_amount) * first_shape[0] * first_shape[2] * first_shape[3] < max_total_area):
                        to_batch = batch_amount
                        break

                input_x = []
                mult = []
                c = []
                cond_or_uncond = []
                area = []
                for x in to_batch:
                    o = to_run.pop(x)
                    p = o[0]
                    input_x += [p[0]]
                    mult += [p[1]]
                    c += [p[2]]
                    area += [p[3]]
                    cond_or_uncond += [o[1]]

                batch_chunks = len(cond_or_uncond)
                input_x = torch.cat(input_x)
                c = cond_cat(c)
                sigma_ = torch.cat([sigma] * batch_chunks)

                output = model_function(input_x, sigma_, cond=c).chunk(batch_chunks)
                del input_x

                for o in range(batch_chunks):
                    if cond_or_uncond[o] == COND:
                        out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                        out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
                    else:
                        out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                        out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
                del mult

            out_cond /= out_count
            del out_count
            out_uncond /= out_uncond_count
            del out_uncond_count

            return out_cond, out_uncond


        max_total_area = model_management.maximum_batch_area()
        cond, uncond = calc_cond_uncond_batch(model_function, cond, uncond, x, sigma, max_total_area, cond_concat)
        return uncond + (cond - uncond) * cond_scale

class CFGDenoiserComplex(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, cond_concat=None):
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + (self.latent_image + self.noise * sigma) * latent_mask
        out = sampling_function(self.inner_model, x, sigma, uncond, cond, cond_scale, cond_concat)
        if denoise_mask is not None:
            out *= denoise_mask

        if denoise_mask is not None:
            out += self.latent_image * latent_mask
        return out

def simple_scheduler(model, steps):
    sigs = []
    ss = len(model.sigmas) / steps
    for x in range(steps):
        sigs += [float(model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def blank_inpaint_image_like(latent_image):
    blank_image = torch.ones_like(latent_image)
    # these are the values for "zero" in pixel space translated to latent space
    # the proper way to do this is to apply the mask to the image in pixel space and then send it through the VAE
    # unfortunately that gives zero flexibility so I did things like this instead which hopefully works
    blank_image[:,0] *= 0.8223
    blank_image[:,1] *= -0.6876
    blank_image[:,2] *= 0.6364
    blank_image[:,3] *= 0.1380
    return blank_image

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
                "sample_dpmpp_2m", "uni_pc"]

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None):
        self.model = model
        if self.model.parameterization == "v":
            self.model_wrap = k_diffusion_external.CompVisVDenoiser(self.model, quantize=True)
        else:
            self.model_wrap = k_diffusion_external.CompVisDenoiser(self.model, quantize=True)
        self.model_k = CFGDenoiserComplex(self.model_wrap)
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.sigma_min=float(self.model_wrap.sigma_min)
        self.sigma_max=float(self.model_wrap.sigma_max)
        self.set_steps(steps, denoise)

    def _calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['sample_dpm_2', 'sample_dpm_2_ancestral']:
            steps += 1
            discard_penultimate_sigma = True

        if self.scheduler == "karras":
            sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device)
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


    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None):
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

        extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg}

        if hasattr(self.model, 'concat_keys'):
            cond_concat = []
            for ck in self.model.concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask[:,:1])
                    elif ck == "masked_image":
                        cond_concat.append(latent_image) #NOTE: the latent_image should be masked by the mask in pixel space
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:,:1])
                    elif ck == "masked_image":
                        cond_concat.append(blank_inpaint_image_like(noise))
            extra_args["cond_concat"] = cond_concat

        with precision_scope(self.device):
            if self.sampler == "uni_pc":
                samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, extra_args=extra_args, noise_mask=denoise_mask)
            else:
                extra_args["denoise_mask"] = denoise_mask
                self.model_k.latent_image = latent_image
                self.model_k.noise = noise

                noise = noise * sigmas[0]

                if latent_image is not None:
                    noise += latent_image
                if self.sampler == "sample_dpm_fast":
                    samples = k_diffusion_sampling.sample_dpm_fast(self.model_k, noise, sigma_min, sigmas[0], self.steps, extra_args=extra_args)
                elif self.sampler == "sample_dpm_adaptive":
                    samples = k_diffusion_sampling.sample_dpm_adaptive(self.model_k, noise, sigma_min, sigmas[0], extra_args=extra_args)
                else:
                    samples = getattr(k_diffusion_sampling, self.sampler)(self.model_k, noise, sigmas, extra_args=extra_args)

        return samples.to(torch.float32)
