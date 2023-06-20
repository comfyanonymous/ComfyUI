

class ModelMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "_for_testing/model_merging"

    def merge(self, model1, model2, ratio):
        m = model1.clone()
        sd = model2.model_state_dict()
        for k in sd:
            m.add_patches({k: (sd[k], )}, 1.0 - ratio, ratio)
        return (m, )

class ModelMergeBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "_for_testing/model_merging"

    def merge(self, model1, model2, **kwargs):
        m = model1.clone()
        sd = model2.model_state_dict()
        default_ratio = next(iter(kwargs.values()))

        for k in sd:
            ratio = default_ratio
            k_unet = k[len("diffusion_model."):]

            for arg in kwargs:
                if k_unet.startswith(arg):
                    ratio = kwargs[arg]

            m.add_patches({k: (sd[k], )}, 1.0 - ratio, ratio)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "ModelMergeSimple": ModelMergeSimple,
    "ModelMergeBlocks": ModelMergeBlocks
}
