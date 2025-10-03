from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class EpsilonScaling(io.ComfyNode):
    """
    Implements the Epsilon Scaling method from 'Elucidating the Exposure Bias in Diffusion Models'
    (https://arxiv.org/abs/2308.15321v6).

    This method mitigates exposure bias by scaling the predicted noise during sampling,
    which can significantly improve sample quality. This implementation uses the "uniform schedule"
    recommended by the paper for its practicality and effectiveness.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Epsilon Scaling",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "scaling_factor",
                    default=1.005,
                    min=0.5,
                    max=1.5,
                    step=0.001,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, scaling_factor) -> io.NodeOutput:
        # Prevent division by zero, though the UI's min value should prevent this.
        if scaling_factor == 0:
            scaling_factor = 1e-9

        def epsilon_scaling_function(args):
            """
            This function is applied after the CFG guidance has been calculated.
            It recalculates the denoised latent by scaling the predicted noise.
            """
            denoised = args["denoised"]
            x = args["input"]

            noise_pred = x - denoised

            scaled_noise_pred = noise_pred / scaling_factor

            new_denoised = x - scaled_noise_pred

            return new_denoised

        # Clone the model patcher to avoid modifying the original model in place
        model_clone = model.clone()

        model_clone.set_model_sampler_post_cfg_function(epsilon_scaling_function)

        return io.NodeOutput(model_clone)


class EpsilonScalingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EpsilonScaling,
        ]

async def comfy_entrypoint() -> EpsilonScalingExtension:
    return EpsilonScalingExtension()
