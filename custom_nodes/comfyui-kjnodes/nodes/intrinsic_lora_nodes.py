import folder_paths
import os
import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar, load_torch_file
import comfy.sample
from nodes import CLIPTextEncode

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_paths.add_model_folder_path("intrinsic_loras", os.path.join(script_directory, "intrinsic_loras"))

class Intrinsic_lora_sampling:
    def __init__(self):
        self.loaded_lora = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("intrinsic_loras"), ),
                "task": (
                [   
                    'depth map',
                    'surface normals',
                    'albedo',
                    'shading',
                ],
                {
                "default": 'depth map'
                    }),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
        },
            "optional": {
            "image": ("IMAGE",),
            "optional_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT",)
    FUNCTION = "onestepsample"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Sampler to use the intrinsic loras:  
https://github.com/duxiaodan/intrinsic-lora  
These LoRAs are tiny and thus included  
with this node pack.
"""

    def onestepsample(self, model, lora_name, clip, vae, text, task, per_batch, image=None, optional_latent=None):
        pbar = ProgressBar(3)

        if optional_latent is None:
            image_list = []
            for start_idx in range(0, image.shape[0], per_batch):
                sub_pixels = vae.vae_encode_crop_pixels(image[start_idx:start_idx+per_batch])
                image_list.append(vae.encode(sub_pixels[:,:,:,:3]))
            sample = torch.cat(image_list, dim=0)
        else:
            sample = optional_latent["samples"]
        noise = torch.zeros(sample.size(), dtype=sample.dtype, layout=sample.layout, device="cpu")
        prompt = task + "," + text
        positive, = CLIPTextEncode.encode(self, clip, prompt)
        negative = positive #negative shouldn't do anything in this scenario

        pbar.update(1)
     
        #custom model sampling to pass latent through as it is
        class X0_PassThrough(comfy.model_sampling.EPS):
            def calculate_denoised(self, sigma, model_output, model_input):
                return model_output
            def calculate_input(self, sigma, noise):
                return noise
        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        sampling_type = X0_PassThrough

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)

        #load lora
        model_clone = model.clone()
        lora_path = folder_paths.get_full_path("intrinsic_loras", lora_name)        
        lora = load_torch_file(lora_path, safe_load=True)
        self.loaded_lora = (lora_path, lora)

        model_clone_with_lora = comfy.sd.load_lora_for_models(model_clone, None, lora, 1.0, 0)[0]

        model_clone_with_lora.add_object_patch("model_sampling", model_sampling)

        samples = {"samples": comfy.sample.sample(model_clone_with_lora, noise, 1, 1.0, "euler", "simple", positive, negative, sample,
                                  denoise=1.0, disable_noise=True, start_step=0, last_step=1,
                                  force_full_denoise=True, noise_mask=None, callback=None, disable_pbar=True, seed=None)}
        pbar.update(1)

        decoded = []
        for start_idx in range(0, samples["samples"].shape[0], per_batch):
            decoded.append(vae.decode(samples["samples"][start_idx:start_idx+per_batch]))
        image_out = torch.cat(decoded, dim=0)

        pbar.update(1)

        if task == 'depth map':
            imax = image_out.max()
            imin = image_out.min()
            image_out = (image_out-imin)/(imax-imin)
            image_out = torch.max(image_out, dim=3, keepdim=True)[0].repeat(1, 1, 1, 3)
        elif task == 'surface normals':
            image_out = F.normalize(image_out * 2 - 1, dim=3) / 2 + 0.5
            image_out = 1.0 - image_out
        else:
            image_out = image_out.clamp(-1.,1.)
            
        return (image_out, samples,)