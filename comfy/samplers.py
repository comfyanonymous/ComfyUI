from .k_diffusion import sampling as k_diffusion_sampling
from .k_diffusion import external as k_diffusion_external
from .extra_samplers import uni_pc
import torch
from comfy import model_management
from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.modules.diffusionmodules.util import make_ddim_timesteps
import math
from comfy import model_base

def lcm(a, b): #TODO: eventually replace by math.lcm (added in python3.9)
    return abs(a*b) // math.gcd(a, b)

#The main sampling function shared by all the samplers
#Returns predicted noise
def sampling_function(model_function, x, timestep, uncond, cond, cond_scale, cond_concat=None, model_options={}, seed=None):
        def get_area_and_mult(cond, x_in, cond_concat_in, timestep_in):
            area = (x_in.shape[2], x_in.shape[3], 0, 0)
            strength = 1.0
            if 'timestep_start' in cond[1]:
                timestep_start = cond[1]['timestep_start']
                if timestep_in[0] > timestep_start:
                    return None
            if 'timestep_end' in cond[1]:
                timestep_end = cond[1]['timestep_end']
                if timestep_in[0] < timestep_end:
                    return None
            if 'area' in cond[1]:
                area = cond[1]['area']
            if 'strength' in cond[1]:
                strength = cond[1]['strength']

            adm_cond = None
            if 'adm_encoded' in cond[1]:
                adm_cond = cond[1]['adm_encoded']

            input_x = x_in[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
            if 'mask' in cond[1]:
                # Scale the mask to the size of the input
                # The mask should have been resized as we began the sampling process
                mask_strength = 1.0
                if "mask_strength" in cond[1]:
                    mask_strength = cond[1]["mask_strength"]
                mask = cond[1]['mask']
                assert(mask.shape[1] == x_in.shape[2])
                assert(mask.shape[2] == x_in.shape[3])
                mask = mask[:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] * mask_strength
                mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
            else:
                mask = torch.ones_like(input_x)
            mult = mask * strength

            if 'mask' not in cond[1]:
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
                s1 = c1['c_crossattn'].shape
                s2 = c2['c_crossattn'].shape
                if s1 != s2:
                    if s1[0] != s2[0] or s1[2] != s2[2]: #these 2 cases should not happen
                        return False

                    mult_min = lcm(s1[1], s2[1])
                    diff = mult_min // min(s1[1], s2[1])
                    if diff > 4: #arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
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
            crossattn_max_len = 0
            for x in c_list:
                if 'c_crossattn' in x:
                    c = x['c_crossattn']
                    if crossattn_max_len == 0:
                        crossattn_max_len = c.shape[1]
                    else:
                        crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
                    c_crossattn.append(c)
                if 'c_concat' in x:
                    c_concat.append(x['c_concat'])
                if 'c_adm' in x:
                    c_adm.append(x['c_adm'])
            out = {}
            c_crossattn_out = []
            for c in c_crossattn:
                if c.shape[1] < crossattn_max_len:
                    c = c.repeat(1, crossattn_max_len // c.shape[1], 1) #padding with repeat doesn't change result
                c_crossattn_out.append(c)

            if len(c_crossattn_out) > 0:
                out['c_crossattn'] = [torch.cat(c_crossattn_out)]
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
            if uncond is not None:
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
                    c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

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

                if 'model_function_wrapper' in model_options:
                    output = model_options['model_function_wrapper'](model_function, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
                else:
                    output = model_function(input_x, timestep_, **c).chunk(batch_chunks)
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
        if math.isclose(cond_scale, 1.0):
            uncond = None

        cond, uncond = calc_cond_uncond_batch(model_function, cond, uncond, x, timestep, max_total_area, cond_concat, model_options)
        if "sampler_cfg_function" in model_options:
            args = {"cond": cond, "uncond": uncond, "cond_scale": cond_scale, "timestep": timestep}
            return model_options["sampler_cfg_function"](args)
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
    def apply_model(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, model_options={}, seed=None):
        out = sampling_function(self.inner_model.apply_model, x, timestep, uncond, cond, cond_scale, cond_concat, model_options=model_options, seed=seed)
        return out


class KSamplerX0Inpaint(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, cond_concat=None, model_options={}, seed=None):
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + (self.latent_image + self.noise * sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1))) * latent_mask
        out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, cond_concat=cond_concat, model_options=model_options, seed=seed)
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

def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty

def resolve_cond_masks(conditions, h, w, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if 'mask' in c[1]:
            mask = c[1]['mask']
            mask = mask.to(device=device)
            modified = c[1].copy()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[1] != h or mask.shape[2] != w:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

            if modified.get("set_area_to_bounds", False):
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified['area'] = (8, 8, 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified['area'] = area

            modified['mask'] = mask
            conditions[i] = [c[0], modified]

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

def calculate_start_end_timesteps(model, conds):
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        if 'start_percent' in x[1]:
            timestep_start = model.sigma_to_t(model.t_to_sigma(torch.tensor(x[1]['start_percent'] * 999.0)))
        if 'end_percent' in x[1]:
            timestep_end = model.sigma_to_t(model.t_to_sigma(torch.tensor(x[1]['end_percent'] * 999.0)))

        if (timestep_start is not None) or (timestep_end is not None):
            n = x[1].copy()
            if (timestep_start is not None):
                n['timestep_start'] = timestep_start
            if (timestep_end is not None):
                n['timestep_end'] = timestep_end
            conds[t] = [x[0], n]

def pre_run_control(model, conds):
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: model.sigma_to_t(model.t_to_sigma(torch.tensor(a) * 999.0))
        if 'control' in x[1]:
            x[1]['control'].pre_run(model.inner_model, percent_to_timestep_function)

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

def encode_adm(model, conds, batch_size, width, height, device, prompt_type):
    for t in range(len(conds)):
        x = conds[t]
        adm_out = None
        if 'adm' in x[1]:
            adm_out = x[1]["adm"]
        else:
            params = x[1].copy()
            params["width"] = params.get("width", width * 8)
            params["height"] = params.get("height", height * 8)
            params["prompt_type"] = params.get("prompt_type", prompt_type)
            adm_out = model.encode_adm(device=device, **params)

        if adm_out is not None:
            x[1] = x[1].copy()
            x[1]["adm_encoded"] = torch.cat([adm_out] * batch_size).to(device)

    return conds


class KSampler:
    SCHEDULERS = ["normal", "karras", "exponential", "simple", "ddim_uniform"]
    SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.model_denoise = CFGNoisePredictor(self.model)
        if self.model.model_type == model_base.ModelType.V_PREDICTION:
            self.model_wrap = CompVisVDenoiser(self.model_denoise, quantize=True)
        else:
            self.model_wrap = k_diffusion_external.CompVisDenoiser(self.model_denoise, quantize=True)

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

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['dpm_2', 'dpm_2_ancestral']:
            steps += 1
            discard_penultimate_sigma = True

        if self.scheduler == "karras":
            sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        elif self.scheduler == "exponential":
            sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        elif self.scheduler == "normal":
            sigmas = self.model_wrap.get_sigmas(steps)
        elif self.scheduler == "simple":
            sigmas = simple_scheduler(self.model_wrap, steps)
        elif self.scheduler == "ddim_uniform":
            sigmas = ddim_scheduler(self.model_wrap, steps)
        else:
            print("error invalid scheduler", self.scheduler)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
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

        resolve_cond_masks(positive, noise.shape[2], noise.shape[3], self.device)
        resolve_cond_masks(negative, noise.shape[2], noise.shape[3], self.device)

        calculate_start_end_timesteps(self.model_wrap, negative)
        calculate_start_end_timesteps(self.model_wrap, positive)

        #make sure each cond area has an opposite one with the same area
        for c in positive:
            create_cond_with_same_area_if_none(negative, c)
        for c in negative:
            create_cond_with_same_area_if_none(positive, c)

        pre_run_control(self.model_wrap, negative + positive)

        apply_empty_x_to_equal_area(list(filter(lambda c: c[1].get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
        apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

        if self.model.is_adm():
            positive = encode_adm(self.model, positive, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "positive")
            negative = encode_adm(self.model, negative, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "negative")

        if latent_image is not None:
            latent_image = self.model.process_latent_in(latent_image)

        extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": self.model_options, "seed":seed}

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


        if self.sampler == "uni_pc":
            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask, callback=callback, disable=disable_pbar)
        elif self.sampler == "uni_pc_bh2":
            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask, callback=callback, variant='bh2', disable=disable_pbar)
        elif self.sampler == "ddim":
            timesteps = []
            for s in range(sigmas.shape[0]):
                timesteps.insert(0, self.model_wrap.sigma_to_discrete_timestep(sigmas[s]))
            noise_mask = None
            if denoise_mask is not None:
                noise_mask = 1.0 - denoise_mask

            ddim_callback = None
            if callback is not None:
                total_steps = len(timesteps) - 1
                ddim_callback = lambda pred_x0, i: callback(i, pred_x0, None, total_steps)

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
                                                    img_callback=ddim_callback,
                                                    denoise_function=self.model_wrap.predict_eps_discrete_timestep,
                                                    extra_args=extra_args,
                                                    mask=noise_mask,
                                                    to_zero=sigmas[-1]==0,
                                                    end_step=sigmas.shape[0] - 1,
                                                    disable_pbar=disable_pbar)

        else:
            extra_args["denoise_mask"] = denoise_mask
            self.model_k.latent_image = latent_image
            self.model_k.noise = noise

            if max_denoise:
                noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
            else:
                noise = noise * sigmas[0]

            k_callback = None
            total_steps = len(sigmas) - 1
            if callback is not None:
                k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

            if latent_image is not None:
                noise += latent_image
            if self.sampler == "dpm_fast":
                samples = k_diffusion_sampling.sample_dpm_fast(self.model_k, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=k_callback, disable=disable_pbar)
            elif self.sampler == "dpm_adaptive":
                samples = k_diffusion_sampling.sample_dpm_adaptive(self.model_k, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=k_callback, disable=disable_pbar)
            else:
                samples = getattr(k_diffusion_sampling, "sample_{}".format(self.sampler))(self.model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar)

        return self.model.process_latent_out(samples.to(torch.float32))
