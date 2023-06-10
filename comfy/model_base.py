import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
import numpy as np

class BaseModel(torch.nn.Module):
    def __init__(self, unet_config, v_prediction=False):
        super().__init__()

        self.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=0.00085, linear_end=0.012, cosine_s=8e-3)
        self.diffusion_model = UNetModel(**unet_config)
        self.v_prediction = v_prediction
        if self.v_prediction:
            self.parameterization = "v"
        else:
            self.parameterization = "eps"
        if "adm_in_channels" in unet_config:
            self.adm_channels = unet_config["adm_in_channels"]
        else:
            self.adm_channels = 0
        print("v_prediction", v_prediction)
        print("adm", self.adm_channels)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}):
        if c_concat is not None:
            xc = torch.cat([x] + c_concat, dim=1)
        else:
            xc = x
        context = torch.cat(c_crossattn, 1)
        return self.diffusion_model(xc, t, context=context, y=c_adm, control=control, transformer_options=transformer_options)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def is_adm(self):
        return self.adm_channels > 0

class SD21UNCLIP(BaseModel):
    def __init__(self, unet_config, noise_aug_config, v_prediction=True):
        super().__init__(unet_config, v_prediction)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**noise_aug_config)

class SDInpaint(BaseModel):
    def __init__(self, unet_config, v_prediction=False):
        super().__init__(unet_config, v_prediction)
        self.concat_keys = ("mask", "masked_image")
