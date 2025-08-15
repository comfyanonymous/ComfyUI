from comfy_api.torch_helpers import set_torch_compile_wrapper


class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "backend": (["inductor", "cudagraphs"],),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model, backend):
        m = model.clone()
        set_torch_compile_wrapper(model=m, backend=backend)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
}
