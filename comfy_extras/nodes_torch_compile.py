import torch
import importlib


class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        if importlib.util.find_spec("openvino") is not None:
            import openvino as ov

            core = ov.Core()
            available_devices = core.available_devices
        else:
            available_devices = []

        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor", "cudagraphs", "openvino"],),
            },
            "optional": {
                "openvino_device": (available_devices,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model, backend, openvino_device):
        print(model.__class__.__name__)
        if backend == "openvino":
            options = {"device": openvino_device}
            try:
                import openvino.torch
            except ImportError:
                raise ImportError(
                    "Could not import openvino python package. "
                    "Please install it with `pip install openvino`."
                )
            import openvino.frontend.pytorch.torchdynamo.execute as ov_ex

            torch._dynamo.reset()
            ov_ex.compiled_cache.clear()
            ov_ex.req_cache.clear()
            ov_ex.partitioned_modules.clear()
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
