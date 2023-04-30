import comfy

class MuxLatent:

    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",),
                "weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "interpolate"

    CATEGORY = "latent"

    def interpolate(self, latent1, latent2, weight):
        # Ensure the latents have the same shape
        if latent1["samples"].shape != latent2["samples"].shape:
            raise ValueError("Latents must have the same shape")

        # Interpolate the latents using the weight
        interpolated_latent = latent1["samples"] * (1 - weight) + latent2["samples"] * weight

        return ({"samples": interpolated_latent},)

class LoadLatent:
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "filename": ("STRING", {"default": "ComfyUI_latent.npy"})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"

    CATEGORY = "latent"

    def load(self, filename):
        return ({"samples": comfy.utils.load_latent(filename)},)


class SaveLatent:
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "filename": ("STRING", {"default": "ComfyUI_latent.npy"})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "save"

    CATEGORY = "latent"

    def save(self, samples, filename):
        s = samples.copy()
        comfy.utils.save_latent(samples["samples"], filename)
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "MuxLatent": MuxLatent,
    "LoadLatenet": LoadLatent,
    "SaveLatent": SaveLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuxLatent": "Mux Latent",
    "LoadLatent": "Load Latent",
    "SaveLatent": "Save Latent",
}

