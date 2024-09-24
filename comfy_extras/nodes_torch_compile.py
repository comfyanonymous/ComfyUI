import torch

class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model):
        m = model.clone()
        m.add_object_patch("diffusion_model", torch.compile(model=m.get_model_object("diffusion_model")))
        return (m, )

NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
}
