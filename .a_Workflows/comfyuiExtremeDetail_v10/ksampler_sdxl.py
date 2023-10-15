import torch
import comfy.model_management
import comfy.sample
import latent_preview

def prepare_mask(mask, shape):
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    mask = mask.expand((-1,shape[1],-1,-1))
    if mask.shape[0] < shape[0]:
        mask = mask.repeat((shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
    return mask
def remap_range(value, minIn, MaxIn, minOut, maxOut):
            if value > MaxIn: value = MaxIn;
            if value < minIn: value = minIn;
            finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
            return finalValue;

class KSamplerSDXLAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        ui_widgets = {"required":
                    {
                    "model_model": ("MODEL",),
                    "model_refiner": ("MODEL",),
                    "CONDITIONING_model_pos": ("CONDITIONING", ),
                    "CONDITIONING_model_neg": ("CONDITIONING", ),
                    "CONDITIONING_refiner_pos": ("CONDITIONING", ),
                    "CONDITIONING_refiner_neg": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),

                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0}),
                    # "cfg_rescale_multiplier": ("FLOAT", {"default": 1, "min": -1.0, "max": 2.0, "step": 0.1}),

                    "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),

                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "base_steps": ("INT", {"default": 12, "min": 0, "max": 10000}),
                    "refiner_steps": ("INT", {"default": 4, "min": 0, "max": 10000}),
                    "detail_level": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.1}),
                    "detail_from": (["penultimate_step","base_sample"], {"default": "penultimate_step"}),
                    "noise_source": (["CPU","GPU"], {"default": "CPU"}),
                    "auto_rescale_tonemap": (["enable","disable"], {"default": "enable"}),
                    "rescale_tonemap_to": ("FLOAT", {"default": 7.5, "min": 0, "max": 30.0, "step": 0.5}),
                    # "refiner_extra_noise": (["enable","disable"], {"default": "disable"}),
                    # "base_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                    # "noise_shift_end_refiner": ("INT", {"default": -1, "min": -10000, "max": 0})
                    },
                "optional": 
                    {
                    "SD15VAE": ("VAE", ),
                    "SDXLVAE": ("VAE", ),
                    }
                }
        return ui_widgets

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_sdxl"
    CATEGORY = "sampling"


    def patch_tonemap(self, model, multiplier):
        def sampler_tonemap_reinhard(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            noise_pred = (cond - uncond)
            noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred, dim=(1)) + 0.0000000001)[:,None]
            noise_pred /= noise_pred_vector_magnitude

            mean = torch.mean(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
            std = torch.std(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
            top = (std * 3 + mean) * multiplier

            #reinhard
            noise_pred_vector_magnitude *= (1.0 / top)
            new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
            new_magnitude *= top

            return uncond + noise_pred * new_magnitude * cond_scale

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_tonemap_reinhard)
        return m

    # def patch_model(self, model, multiplier):
    #     def rescale_cfg(args):
    #         cond = args["cond"]
    #         uncond = args["uncond"]
    #         cond_scale = args["cond_scale"]

    #         x_cfg = uncond + cond_scale * (cond - uncond)
    #         ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
    #         ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

    #         x_rescaled = x_cfg * (ro_pos / ro_cfg)
    #         x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

    #         return x_final

    #     m = model.clone()
    #     m.set_model_sampler_cfg_function(rescale_cfg)
    #     return m
    
    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        device = comfy.model_management.get_torch_device()
        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = latent_preview.get_previewer(device, model.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return out
    
    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return self.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

    def calc_sigma(self, model, sampler_name, scheduler, steps, start_at_step, end_at_step):
        device = comfy.model_management.get_torch_device()
        end_at_step = min(steps, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model
        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma_output = sigma.cpu().numpy()
        print("Calculated sigma:",sigma_output)
        return sigma_output
    
    def create_noisy_latents(self, source, seed, width, height, batch_size):
        torch.manual_seed(seed)
        if source == "CPU":
            device = "cpu"
        else:
            device = comfy.model_management.get_torch_device()
        noise = torch.randn((batch_size,  4, height // 8, width // 8), dtype=torch.float32, device=device).cpu()
        return {"samples":noise}
    
    def inject_noise(self, latents, strength, noise=None, mask=None):
        s = latents.copy()
        if noise is None:
            return s
        if latents["samples"].shape != noise["samples"].shape:
            print("warning, shapes in InjectNoise not the same, ignoring")
            return s
        noised = s["samples"].clone() + noise["samples"].clone() * strength
        if mask is not None:
            mask = prepare_mask(mask, noised.shape)
            noised = mask * noised + (1-mask) * latents["samples"]
        s["samples"] = noised
        return s

    # from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
    def slerp(self, val, low, high):
        dims = low.shape

        #flatten to batches
        low = low.reshape(dims[0], -1)
        high = high.reshape(dims[0], -1)

        low_norm = low/torch.norm(low, dim=1, keepdim=True)
        high_norm = high/torch.norm(high, dim=1, keepdim=True)

        # in case we divide by zero
        low_norm[low_norm != low_norm] = 0.0
        high_norm[high_norm != high_norm] = 0.0

        omega = torch.acos((low_norm*high_norm).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
        return res.reshape(dims)

    def slerp_latents(self, latents1, factor, latents2=None, mask=None):
        s = latents1.copy()
        if latents2 is None:
            return (s,)
        if latents1["samples"].shape != latents2["samples"].shape:
            print("warning, shapes in LatentSlerp not the same, ignoring")
            return (s,)
        slerped = self.slerp(factor, latents1["samples"].clone(), latents2["samples"].clone())
        if mask is not None:
            mask = prepare_mask(mask, slerped.shape)
            slerped = mask * slerped + (1-mask) * latents1["samples"]
        s["samples"] = slerped
        return s
    
    def compute_and_generate_noise(self,samples,seed,width,height,batch_size,model,sampler,scheduler,total_steps,start_at,end_at,source):
        noisy_latent = self.create_noisy_latents(source,seed,width,height,batch_size)
        sigma_balls  = self.calc_sigma(model,sampler,scheduler,total_steps,start_at,end_at)
        samples_output = self.inject_noise(samples,sigma_balls,noisy_latent)
        return samples_output

    def sample_sdxl(self, model_model, model_refiner, CONDITIONING_model_pos, CONDITIONING_model_neg, CONDITIONING_refiner_pos, CONDITIONING_refiner_neg, latent_image, seed, cfg_scale, sampler, scheduler, start_at_step, base_steps, refiner_steps,detail_level,detail_from,noise_source,auto_rescale_tonemap,rescale_tonemap_to,SD15VAE=None, SDXLVAE=None):

        # if cfg_rescale_multiplier != 1:
        #     model_model   = self.patch_model(model_model,cfg_rescale_multiplier)
        #     model_refiner = self.patch_model(model_refiner,cfg_rescale_multiplier)
        if auto_rescale_tonemap == "enable" and cfg_scale!=rescale_tonemap_to:
            scale_model = 1/cfg_scale*rescale_tonemap_to
            model_model   = self.patch_tonemap(model_model,scale_model)
            if sampler == "uni_pc" or sampler == "uni_pc_bh2":
                scale_model = 1/cfg_scale*7.5
            model_refiner = self.patch_tonemap(model_refiner,scale_model)
            
        for lat in latent_image['samples']:
            d, y, x = lat.size()
            break

        batch_size = len(latent_image['samples'])
        width  = x*8
        height = y*8

        base_start_at       = start_at_step
        base_end_at         = base_steps
        base_total_steps    = base_steps + refiner_steps
        refiner_start_at    = base_steps
        refiner_end_at      = base_steps + refiner_steps
        refiner_total_steps = base_steps + refiner_steps
        
        if sampler == "uni_pc" or sampler == "uni_pc_bh2":
            noisy_base   = self.compute_and_generate_noise(latent_image,seed,width,height,batch_size,model_model,sampler,scheduler,base_end_at-1,base_start_at,base_end_at-1,noise_source)
        else:
            noisy_base   = self.compute_and_generate_noise(latent_image,seed,width,height,batch_size,model_model,sampler,scheduler,base_end_at,base_start_at,base_end_at,noise_source)
        sample_model = self.sample(model_model,"disable",seed,base_total_steps,cfg_scale,sampler,scheduler,CONDITIONING_model_pos,CONDITIONING_model_neg,noisy_base,base_start_at,base_end_at,"disable")

        if SD15VAE is not None and SDXLVAE is not None:
            sample_model["samples"] = SD15VAE.decode(sample_model["samples"])
            sample_model["samples"] = SDXLVAE.encode(sample_model["samples"])
        
        if sampler == "uni_pc" or sampler == "uni_pc_bh2":
            sampler = "dpmpp_2m"
            scheduler = "karras"

        if detail_level < 0.9999 or detail_level > 1:
            if detail_from == "penultimate_step":
                if detail_level > 1:
                    noisy_latent_1 = self.compute_and_generate_noise(sample_model,seed,width,height,batch_size,model_refiner,sampler,scheduler,refiner_total_steps+1,refiner_start_at,refiner_end_at+1,noise_source)
                else:
                    noisy_latent_1 = self.compute_and_generate_noise(sample_model,seed,width,height,batch_size,model_refiner,sampler,scheduler,refiner_total_steps-1,refiner_start_at,refiner_end_at-1,noise_source)
            else:
                noisy_latent_1 = sample_model
            noisy_latent_2 = self.compute_and_generate_noise(sample_model,seed,width,height,batch_size,model_refiner,sampler,scheduler,refiner_total_steps, refiner_start_at,refiner_end_at,noise_source)
            if detail_level > 1:
                noisy_latent_3 = self.slerp_latents(noisy_latent_1,remap_range(detail_level,1,2,1,0),noisy_latent_2)
            else:
                noisy_latent_3 = self.slerp_latents(noisy_latent_1,detail_level,noisy_latent_2)
        else:
            noisy_latent_3 = self.compute_and_generate_noise(sample_model,seed,width,height,batch_size,model_refiner,sampler,scheduler,refiner_total_steps, refiner_start_at,refiner_end_at,noise_source)

        sample_refiner = self.sample(model_refiner,"disable",seed,refiner_total_steps,cfg_scale,sampler,scheduler,CONDITIONING_refiner_pos,CONDITIONING_refiner_neg,noisy_latent_3,refiner_start_at,refiner_end_at,"disable")

        return (sample_refiner,)
    
NODE_CLASS_MAPPINGS = {
    "KSamplerSDXLAdvanced": KSamplerSDXLAdvanced
}
