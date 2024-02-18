import torch
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
import math

class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma


class V_PREDICTION(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) - model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)

        self._register_schedule(given_betas=None, beta_schedule=beta_schedule, timesteps=1000, linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3)
        self.sigma_data = 1.0

    def _register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        # self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas.float())
        self.register_buffer('log_sigmas', sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(timestep.float().to(self.log_sigmas.device), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()


class ModelSamplingContinuousEDM(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        self.sigma_data = 1.0

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        sigma_min = sampling_settings.get("sigma_min", 0.002)
        sigma_max = sampling_settings.get("sigma_max", 120.0)
        self.set_sigma_range(sigma_min, sigma_max)

    def set_sigma_range(self, sigma_min, sigma_max):
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), 1000).exp()

        self.register_buffer('sigmas', sigmas) #for compatibility with some schedulers
        self.register_buffer('log_sigmas', sigmas.log())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent

        log_sigma_min = math.log(self.sigma_min)
        return math.exp((math.log(self.sigma_max) - log_sigma_min) * percent + log_sigma_min)

class StableCascadeSampling(ModelSamplingDiscrete):
    def __init__(self, model_config=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.num_timesteps = 1000
        self.shift = sampling_settings.get("shift", 1.0)
        cosine_s=8e-3
        self.cosine_s = torch.tensor(cosine_s)
        sigmas = torch.empty((self.num_timesteps), dtype=torch.float32)
        self._init_alpha_cumprod = torch.cos(self.cosine_s / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
        for x in range(self.num_timesteps):
            t = x / self.num_timesteps
            sigmas[x] = self.sigma(t)

        self.set_sigmas(sigmas)

    def sigma(self, timestep):
        alpha_cumprod = (torch.cos((timestep + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod)

        if self.shift != 1.0:
            var = alpha_cumprod
            logSNR = (var/(1-var)).log()
            logSNR += 2 * torch.log(1.0 / torch.tensor(self.shift))
            alpha_cumprod = logSNR.sigmoid()

        alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999)
        return ((1 - alpha_cumprod) / alpha_cumprod) ** 0.5

    def timestep(self, sigma):
        var = 1 / ((sigma * sigma) + 1)
        var = var.clamp(0, 1.0)
        s, min_var = self.cosine_s.to(var.device), self._init_alpha_cumprod.to(var.device)
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return t

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0

        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent))
