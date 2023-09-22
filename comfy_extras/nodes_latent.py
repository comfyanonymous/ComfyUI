import comfy.utils

def reshape_latent_to(target_shape, latent):
    if latent.shape[1:] != target_shape[1:]:
        latent.movedim(1, -1)
        latent = comfy.utils.common_upscale(latent, target_shape[3], target_shape[2], "bilinear", "center")
        latent.movedim(-1, 1)
    return comfy.utils.repeat_to_batch_size(latent, target_shape[0])


class LatentAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples1": ("LATENT",), "samples2": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples1, samples2):
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 + s2
        return (samples_out,)

class LatentSubtract:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples1": ("LATENT",), "samples2": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples1, samples2):
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 - s2
        return (samples_out,)

class LatentMuliply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples, multiplier):
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1 * multiplier
        return (samples_out,)

NODE_CLASS_MAPPINGS = {
    "LatentAdd": LatentAdd,
    "LatentSubtract": LatentSubtract,
    "LatentMuliply": LatentMuliply,
}
