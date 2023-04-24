from .k_diffusion import sampling as k_diffusion_sampling
from .k_diffusion import external as k_diffusion_external
from .extra_samplers import uni_pc
import torch
import contextlib
from comfy import model_management
from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.modules.diffusionmodules.util import make_ddim_timesteps

#The main sampling function shared by all the samplers
#Returns predicted noise
def sampling_function(model_function, x, timestep, uncond, cond, cond_scale, cond_concat=None, model_options={}):
        def get_area_and_mult(cond, x_in, cond_concat_in, timestep_in):
            area = (x_in.shape[2], x_in.shape[3], 0, 0)
            strength = 1.0
            if 'area' in cond[1]:
                area = cond[1]['area']
            if 'strength' in cond[1]:
                strength = cond[1]['strength']

            adm_cond = None
            if 'adm_encoded' in cond[1]:
                adm_cond = cond[1]['adm_encoded']

            input_x = x_in[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
            mult = torch.ones_like(input_x) * strength

            rr = 8
            if area[2] != 0:
                for t in range(rr):
                    mult[:,:,t:1+t,:] *= ((1.0/rr) * (t + 1))
            if (area[0] + area[2]) < x_in.shape[2]:
                for t in range(rr):
                    mult[:,:,area[0] - 1 - t:area[0] - t,:] *= ((1.0/rr) * (t + 1))
            if area[3] != 0:
                for t in range(rr):
                    mult[:,:,:,t:1+t] *= ((1.0/rr) * (t + 1))
            if (area[1] + area[3]) < x_in.shape[3]:
                for t in range(rr):
                    mult[:,:,:,area[1] - 1 - t:area[1] - t] *= ((1.0/rr) * (t + 1))
            conditionning = {}
            conditionning['c_crossattn'] = cond[0]
            if cond_concat_in is not None and len(cond_concat_in) > 0:
                cropped = []
                for x in cond_concat_in:
                    cr = x[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
                    cropped.append(cr)
                conditionning['c_concat'] = torch.cat(cropped, dim=1)

            if adm_cond is not None:
                conditionning['c_adm'] = adm_cond

            control = None
            if 'control' in cond[1]:
                control = cond[1]['control']

            patches = None
            if 'gligen' in cond[1]:
                gligen = cond[1]['gligen']
                patches = {}
                gligen_type = gligen[0]
                gligen_model = gligen[1]
                if gligen_type == "position":
                    gligen_patch = gligen_model.set_position(input_x.shape, gligen[2], input_x.device)
                else:
                    gligen_patch = gligen_model.set_empty(input_x.shape, input_x.device)

                patches['middle_patch'] = [gligen_patch]

            return (input_x, mult, conditionning, area, control, patches)

        def cond_equal_size(c1, c2):
            if c1 is c2:
                return True
            if c1.keys() != c2.keys():
                return False
            if 'c_crossattn' in c1:
                if c1['c_crossattn'].shape != c2['c_crossattn'].shape:
                    return False
            if 'c_concat' in c1:
                if c1['c_concat'].shape != c2['c_concat'].shape:
                    return False
            if 'c_adm' in c1:
                if c1['c_adm'].shape != c2['c_adm'].shape:
                    return False
            return True

        def can_concat_cond(c1, c2):
            if c1[0].shape != c2[0].shape:
                return False

            #control
            if (c1[4] is None) != (c2[4] is None):
                return False
            if c1[4] is not None:
                if c1[4] is not c2[4]:
                    return False

            #patches
            if (c1[5] is None) != (c2[5] is None):
                return False
            if (c1[5] is not None):
                if c1[5] is not c2[5]:
                    return False

            return cond_equal_size(c1[2], c2[2])

        def cond_cat(c_list):
            c_crossattn = []
            c_concat = []
            c_adm = []
            for x in c_list:
                if 'c_crossattn' in x:
                    c_crossattn.append(x['c_crossattn'])
                if 'c_concat' in x:
                    c_concat.append(x['c_concat'])
                if 'c_adm' in x:
                    c_adm.append(x['c_adm'])
            out = {}
            if len(c_crossattn) > 0:
                out['c_crossattn'] = [torch.cat(c_crossattn)]
            if len(c_concat) > 0:
                out['c_concat'] = [torch.cat(c_concat)]
            if len(c_adm) > 0:
                out['c_adm'] = torch.cat(c_adm)
            return out

        def calc_cond_uncond_batch(model_function, cond, uncond, x_in, timestep, max_total_area, cond_concat_in, model_options):
            out_cond = torch.zeros_like(x_in)
            out_count = torch.ones_like(x_in)/100000.0

            out_uncond = torch.zeros_like(x_in)
            out_uncond_count = torch.ones_like(x_in)/100000.0

            COND = 0
            UNCOND = 1

            to_run = []
            for x in cond:
                p = get_area_and_mult(x, x_in, cond_concat_in, timestep)
                if p is None:
                    continue

                to_run += [(p, COND)]
            for x in uncond:
                p = get_area_and_mult(x, x_in, cond_concat_in, timestep)
                if p is None:
                    continue

                to_run += [(p, UNCOND)]

            while len(to_run) > 0:
                first = to_run[0]
                first_shape = first[0][0].shape
                to_batch_temp = []
                for x in range(len(to_run)):
                    if can_concat_cond(to_run[x][0], first[0]):
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
                control = None
                patches = None
                for x in to_batch:
                    o = to_run.pop(x)
                    p = o[0]
                    input_x += [p[0]]
                    mult += [p[1]]
                    c += [p[2]]
                    area += [p[3]]
                    cond_or_uncond += [o[1]]
                    control = p[4]
                    patches = p[5]

                batch_chunks = len(cond_or_uncond)
                input_x = torch.cat(input_x)
                c = cond_cat(c)
                timestep_ = torch.cat([timestep] * batch_chunks)

                if control is not None:
                    c['control'] = control.get_control(input_x, timestep_, c['c_crossattn'], len(cond_or_uncond))

                transformer_options = {}
                if 'transformer_options' in model_options:
                    transformer_options = model_options['transformer_options'].copy()

                if patches is not None:
                    if "patches" in transformer_options:
                        cur_patches = transformer_options["patches"].copy()
                        for p in patches:
                            if p in cur_patches:
                                cur_patches[p] = cur_patches[p] + patches[p]
                            else:
                                cur_patches[p] = patches[p]
                    else:
                        transformer_options["patches"] = patches

                c['transformer_options'] = transformer_options

                output = model_function(input_x, timestep_, cond=c).chunk(batch_chunks)
                del input_x

                model_management.throw_exception_if_processing_interrupted()

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
        cond, uncond = calc_cond_uncond_batch(model_function, cond, uncond, x, timestep, max_total_area, cond_concat, model_options)
        if "sampler_cfg_function" in model_options:
            return model_options["sampler_cfg_function"](cond, uncond, cond_scale)
        else:
            return uncond + (cond - uncond) * cond_scale


class CompVisVDenoiser(k_diffusion_external.DiscreteVDDPMDenoiser):
    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_v(self, x, t, cond, **kwargs):
        return self.inner_model.apply_model(x, t, cond, **kwargs)


class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.alphas_cumprod = model.alphas_cumprod
    def apply_model(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, model_options={}):
        out = sampling_function(self.inner_model.apply_model, x, timestep, uncond, cond, cond_scale, cond_concat, model_options=model_options)
        return out


class KSamplerX0Inpaint(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, cond_concat=None, model_options={}):
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + (self.latent_image + self.noise * sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1))) * latent_mask
        out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, cond_concat=cond_concat, model_options=model_options)
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

def ddim_scheduler(model, steps):
    sigs = []
    ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=steps, num_ddpm_timesteps=model.inner_model.inner_model.num_timesteps, verbose=False)
    for x in range(len(ddim_timesteps) - 1, -1, -1):
        ts = ddim_timesteps[x]
        if ts > 999:
            ts = 999
        sigs.append(model.t_to_sigma(torch.tensor(ts)))
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def blank_inpaint_image_like(latent_image):
    blank_image = torch.ones_like(latent_image)
    # these are the values for "zero" in pixel space translated to latent space
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

def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x[1]:
            if name in x[1] and x[1][name] is not None:
                cond_cnets.append(x[1][name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x[1]:
            if name in x[1] and x[1][name] is not None:
                uncond_cnets.append(x[1][name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o[1] and o[1][name] is not None:
            n = o[1].copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [[o[0], n]]
        else:
            n = o[1].copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = [o[0], n]


def encode_adm(noise_augmentor, conds, batch_size, device):
    for t in range(len(conds)):
        x = conds[t]
        if 'adm' in x[1]:
            adm_inputs = []
            weights = []
            noise_aug = []
            adm_in = x[1]["adm"]
            for adm_c in adm_in:
                adm_cond = adm_c[0].image_embeds
                weight = adm_c[1]
                noise_augment = adm_c[2]
                noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
                c_adm, noise_level_emb = noise_augmentor(adm_cond.to(device), noise_level=torch.tensor([noise_level], device=device))
                adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
                weights.append(weight)
                noise_aug.append(noise_augment)
                adm_inputs.append(adm_out)

            if len(noise_aug) > 1:
                adm_out = torch.stack(adm_inputs).sum(0)
                #TODO: add a way to control this
                noise_augment = 0.05
                noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
                c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim], noise_level=torch.tensor([noise_level], device=device))
                adm_out = torch.cat((c_adm, noise_level_emb), 1)
        else:
            adm_out = torch.zeros((1, noise_augmentor.time_embed.dim * 2), device=device)
        x[1] = x[1].copy()
        x[1]["adm_encoded"] = torch.cat([adm_out] * batch_size)

    return conds

def calculate_sigmas(model, steps, scheduler, sampler):
    """
    Returns a tensor containing the sigmas corresponding to the given model, number of steps, scheduler type and sample technique
    """
    if not (isinstance(model, CompVisVDenoiser) or isinstance(model, k_diffusion_external.CompVisDenoiser)):
        model = CFGNoisePredictor(model)
        if model.inner_model.parameterization == "v":
            model = CompVisVDenoiser(model, quantize=True)
        else:
            model = k_diffusion_external.CompVisDenoiser(model, quantize=True)
            
    sigmas = None

    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral']:
        steps += 1
        discard_penultimate_sigma = True

    if scheduler == "karras":
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=float(model.sigma_min), sigma_max=float(model.sigma_max))
    elif scheduler == "normal":
        sigmas = model.get_sigmas(steps)
    elif scheduler == "simple":
        sigmas = simple_scheduler(model, steps)
    elif scheduler == "ddim_uniform":
        sigmas = ddim_scheduler(model, steps)
    else:
        print("error invalid scheduler", scheduler)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas

class KSampler:
    SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
    SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.model_denoise = CFGNoisePredictor(self.model)
        if self.model.parameterization == "v":
            self.model_wrap = CompVisVDenoiser(self.model_denoise, quantize=True)
        else:
            self.model_wrap = k_diffusion_external.CompVisDenoiser(self.model_denoise, quantize=True)
        self.model_wrap.parameterization = self.model.parameterization
        self.model_k = KSamplerX0Inpaint(self.model_wrap)
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
        self.denoise = denoise
        self.model_options = model_options

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = calculate_sigmas(self.model_wrap, steps, self.scheduler, self.sampler).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = calculate_sigmas(self.model_wrap, new_steps, self.scheduler, self.sampler).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]


    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None):
        if sigmas is None:
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

        apply_empty_x_to_equal_area(positive, negative, 'control', lambda cond_cnets, x: cond_cnets[x])
        apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

        if self.model.model.diffusion_model.dtype == torch.float16:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        if hasattr(self.model, 'noise_augmentor'): #unclip
            positive = encode_adm(self.model.noise_augmentor, positive, noise.shape[0], self.device)
            negative = encode_adm(self.model.noise_augmentor, negative, noise.shape[0], self.device)

        extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": self.model_options}

        cond_concat = None
        if hasattr(self.model, 'concat_keys'): #inpaint
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

        if sigmas[0] != self.sigmas[0] or (self.denoise is not None and self.denoise < 1.0):
            max_denoise = False
        else:
            max_denoise = True

        with precision_scope(model_management.get_autocast_device(self.device)):
            if self.sampler == "uni_pc":
                samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask)
            elif self.sampler == "uni_pc_bh2":
                samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask, variant='bh2')
            elif self.sampler == "ddim":
                timesteps = []
                for s in range(sigmas.shape[0]):
                    timesteps.insert(0, self.model_wrap.sigma_to_t(sigmas[s]))
                noise_mask = None
                if denoise_mask is not None:
                    noise_mask = 1.0 - denoise_mask
                sampler = DDIMSampler(self.model, device=self.device)
                sampler.make_schedule_timesteps(ddim_timesteps=timesteps, verbose=False)
                z_enc = sampler.stochastic_encode(latent_image, torch.tensor([len(timesteps) - 1] * noise.shape[0]).to(self.device), noise=noise, max_denoise=max_denoise)
                samples, _ = sampler.sample_custom(ddim_timesteps=timesteps,
                                                     conditioning=positive,
                                                     batch_size=noise.shape[0],
                                                     shape=noise.shape[1:],
                                                     verbose=False,
                                                     unconditional_guidance_scale=cfg,
                                                     unconditional_conditioning=negative,
                                                     eta=0.0,
                                                     x_T=z_enc,
                                                     x0=latent_image,
                                                     denoise_function=sampling_function,
                                                     extra_args=extra_args,
                                                     mask=noise_mask,
                                                     to_zero=sigmas[-1]==0,
                                                     end_step=sigmas.shape[0] - 1)

            else:
                extra_args["denoise_mask"] = denoise_mask
                self.model_k.latent_image = latent_image
                self.model_k.noise = noise

                noise = noise * sigmas[0]

                if latent_image is not None:
                    noise += latent_image
                if self.sampler == "dpm_fast":
                    samples = k_diffusion_sampling.sample_dpm_fast(self.model_k, noise, sigma_min, sigmas[0], self.steps, extra_args=extra_args)
                elif self.sampler == "dpm_adaptive":
                    samples = k_diffusion_sampling.sample_dpm_adaptive(self.model_k, noise, sigma_min, sigmas[0], extra_args=extra_args)
                else:
                    samples = getattr(k_diffusion_sampling, "sample_{}".format(self.sampler))(self.model_k, noise, sigmas, extra_args=extra_args)

        return samples.to(torch.float32)
