import folder_paths
import comfy.sd
import comfy.model_sampling


def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5

class ModelSamplingDiscrete:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["eps", "v_prediction"],),
                              "zsnr": ("BOOLEAN", {"default": False}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, sampling, zsnr):
        m = model.clone()

        if sampling == "eps":
            sampling_type = comfy.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscrete, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced()
        if zsnr:
            model_sampling.set_sigmas(rescale_zero_terminal_snr_sigmas(model_sampling.sigmas))
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "ModelSamplingDiscrete": ModelSamplingDiscrete,
}
