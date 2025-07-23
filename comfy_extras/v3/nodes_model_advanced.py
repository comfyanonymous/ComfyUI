from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_sampling
import comfy.sd
import node_helpers
import nodes
from comfy_api.v3 import io


class LCM(comfy.model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0 #timestep_scaling

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input


class ModelSamplingDiscreteDistilled(comfy.model_sampling.ModelSamplingDiscrete):
    original_timesteps = 50

    def __init__(self, model_config=None, zsnr=None):
        super().__init__(model_config, zsnr=zsnr)

        self.skip_steps = self.num_timesteps // self.original_timesteps

        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[self.num_timesteps - 1 - x * self.skip_steps]

        self.set_sigmas(sigmas_valid)

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return (dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(
            ((timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1)) / self.skip_steps).float(),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)


class ModelComputeDtype(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelComputeDtype_V3",
            category="advanced/debug/model",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("dtype", options=["default", "fp32", "fp16", "bf16"]),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, dtype):
        m = model.clone()
        m.set_model_compute_dtype(node_helpers.string_to_torch_dtype(dtype))
        return io.NodeOutput(m)


class ModelSamplingContinuousEDM(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelSamplingContinuousEDM_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "sampling", options=["v_prediction", "edm", "edm_playground_v2.5", "eps", "cosmos_rflow"]
                ),
                io.Float.Input("sigma_max", default=120.0, min=0.0, max=1000.0, step=0.001, round=False),
                io.Float.Input("sigma_min", default=0.002, min=0.0, max=1000.0, step=0.001, round=False),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingContinuousEDM
        latent_format = None
        sigma_data = 1.0
        if sampling == "eps":
            sampling_type = comfy.model_sampling.EPS
        elif sampling == "edm":
            sampling_type = comfy.model_sampling.EDM
            sigma_data = 0.5
        elif sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION
        elif sampling == "edm_playground_v2.5":
            sampling_type = comfy.model_sampling.EDM
            sigma_data = 0.5
            latent_format = comfy.latent_formats.SDXL_Playground_2_5()
        elif sampling == "cosmos_rflow":
            sampling_type = comfy.model_sampling.COSMOS_RFLOW
            sampling_base = comfy.model_sampling.ModelSamplingCosmosRFlow

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling)
        if latent_format is not None:
            m.add_object_patch("latent_format", latent_format)
        return io.NodeOutput(m)


class ModelSamplingContinuousV(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelSamplingContinuousV_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("sampling", options=["v_prediction"]),
                io.Float.Input("sigma_max", default=500.0, min=0.0, max=1000.0, step=0.001, round=False),
                io.Float.Input("sigma_min", default=0.03, min=0.0, max=1000.0, step=0.001, round=False),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        sigma_data = 1.0
        if sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingContinuousV, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(m)


class ModelSamplingDiscrete(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelSamplingDiscrete_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("sampling", options=["eps", "v_prediction", "lcm", "x0", "img_to_img"]),
                io.Boolean.Input("zsnr", default=False),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, sampling, zsnr):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        if sampling == "eps":
            sampling_type = comfy.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION
        elif sampling == "lcm":
            sampling_type = LCM
            sampling_base = ModelSamplingDiscreteDistilled
        elif sampling == "x0":
            sampling_type = comfy.model_sampling.X0
        elif sampling == "img_to_img":
            sampling_type = comfy.model_sampling.IMG_TO_IMG

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config, zsnr=zsnr)

        m.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(m)


class ModelSamplingFlux(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelSamplingFlux_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("max_shift", default=1.15, min=0.0, max=100.0, step=0.01),
                io.Float.Input("base_shift", default=0.5, min=0.0, max=100.0, step=0.01),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=8),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, max_shift, base_shift, width, height):
        m = model.clone()

        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(m)


class ModelSamplingSD3(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelSamplingSD3_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift", default=3.0, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, shift, multiplier: int | float = 1000):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        m.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(m)


class ModelSamplingAuraFlow(ModelSamplingSD3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelSamplingAuraFlow_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift", default=1.73, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, shift, multiplier: int | float = 1.0):
        return super().execute(model, shift, multiplier)


class ModelSamplingStableCascade(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ModelSamplingStableCascade_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift", default=2.0, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, shift):
        m = model.clone()

        sampling_base = comfy.model_sampling.StableCascadeSampling
        sampling_type = comfy.model_sampling.EPS

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift)
        m.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(m)


class RescaleCFG(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="RescaleCFG_V3",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("multiplier", default=0.7, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, multiplier):
        def rescale_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]

            #rescale cfg has to be done on v-pred model output
            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

            #rescalecfg
            x_cfg = uncond + cond_scale * (cond - uncond)
            ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
            ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

            x_rescaled = x_cfg * (ro_pos / ro_cfg)
            x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)

        m = model.clone()
        m.set_model_sampler_cfg_function(rescale_cfg)
        return io.NodeOutput(m)


NODES_LIST = [
    ModelSamplingAuraFlow,
    ModelComputeDtype,
    ModelSamplingContinuousEDM,
    ModelSamplingContinuousV,
    ModelSamplingDiscrete,
    ModelSamplingFlux,
    ModelSamplingSD3,
    ModelSamplingStableCascade,
    RescaleCFG,
]
