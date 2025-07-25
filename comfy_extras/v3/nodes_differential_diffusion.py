from __future__ import annotations

import torch

from comfy_api.latest import io


class DifferentialDiffusion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DifferentialDiffusion_V3",
            display_name="Differential Diffusion _V3",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model):
        model = model.clone()
        model.set_model_denoise_mask_function(cls.forward)
        return io.NodeOutput(model)

    @classmethod
    def forward(cls, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)


NODES_LIST = [
    DifferentialDiffusion,
]
