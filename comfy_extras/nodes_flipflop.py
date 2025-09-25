from comfy.ldm.flipflop_transformer import FLIPFLOP_REGISTRY

class FlipFlop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",), },
                }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    OUTPUT_NODE = False

    CATEGORY = "_for_testing"

    def patch(self, model):
        patch_cls = FLIPFLOP_REGISTRY.get(model.model.diffusion_model.__class__.__name__, None)
        if patch_cls is None:
            raise ValueError(f"Model {model.model.diffusion_model.__class__.__name__} not supported")

        model.model.diffusion_model = patch_cls.patch(model.model.diffusion_model)

        return (model,)

NODE_CLASS_MAPPINGS = {
    "FlipFlop": FlipFlop
}
