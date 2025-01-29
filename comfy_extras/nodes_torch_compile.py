import torch


class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor", "cudagraphs", "openvino"],),
            },
            "optional": {
                "openvino_device": (["CPU", "GPU", "NPU"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model, backend, openvino_device):
        if backend == "openvino":
            options = {"device": openvino_device}
            try:
                import openvino.torch
            except ImportError:
                raise ImportError(
                    "Could not import openvino python package. "
                    "Please install it with `pip install openvino`."
                )
        else:
            options = None
        m = model.clone()
        m.add_object_patch(
            "diffusion_model",
            torch.compile(
                model=m.get_model_object("diffusion_model"),
                backend=backend,
                options=options,
            ),
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
}
