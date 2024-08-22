import torch
from .tools import VariantSupport
from comfy_execution.graph_utils import GraphBuilder

class TestLazyMixImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",{"lazy": True}),
                "image2": ("IMAGE",{"lazy": True}),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"

    CATEGORY = "Testing/Nodes"

    def check_lazy_status(self, mask, image1, image2):
        mask_min = mask.min()
        mask_max = mask.max()
        needed = []
        if image1 is None and (mask_min != 1.0 or mask_max != 1.0):
            needed.append("image1")
        if image2 is None and (mask_min != 0.0 or mask_max != 0.0):
            needed.append("image2")
        return needed

    # Not trying to handle different batch sizes here just to keep the demo simple
    def mix(self, mask, image1, image2):
        mask_min = mask.min()
        mask_max = mask.max()
        if mask_min == 0.0 and mask_max == 0.0:
            return (image1,)
        elif mask_min == 1.0 and mask_max == 1.0:
            return (image2,)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(3)
        if mask.shape[3] < image1.shape[3]:
            mask = mask.repeat(1, 1, 1, image1.shape[3])

        result = image1 * (1. - mask) + image2 * mask,
        return (result[0],)

class TestVariadicAverage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "variadic_average"

    CATEGORY = "Testing/Nodes"

    def variadic_average(self, input1, **kwargs):
        inputs = [input1]
        while 'input' + str(len(inputs) + 1) in kwargs:
            inputs.append(kwargs['input' + str(len(inputs) + 1)])
        return (torch.stack(inputs).mean(dim=0),)


class TestCustomIsChanged:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "should_change": ("BOOL", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "custom_is_changed"

    CATEGORY = "Testing/Nodes"

    def custom_is_changed(self, image, should_change=False):
        return (image,)
    
    @classmethod
    def IS_CHANGED(cls, should_change=False, *args, **kwargs):
        if should_change:
            return float("NaN")
        else:
            return False

class TestIsChangedWithConstants:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "custom_is_changed"

    CATEGORY = "Testing/Nodes"

    def custom_is_changed(self, image, value):
        return (image * value,)
    
    @classmethod
    def IS_CHANGED(cls, image, value):
        if image is None:
            return value
        else:
            return image.mean().item() * value

class TestCustomValidation1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("IMAGE,FLOAT",),
                "input2": ("IMAGE,FLOAT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "custom_validation1"

    CATEGORY = "Testing/Nodes"

    def custom_validation1(self, input1, input2):
        if isinstance(input1, float) and isinstance(input2, float):
            result = torch.ones([1, 512, 512, 3]) * input1 * input2
        else:
            result = input1 * input2
        return (result,)

    @classmethod
    def VALIDATE_INPUTS(cls, input1=None, input2=None):
        if input1 is not None:
            if not isinstance(input1, (torch.Tensor, float)):
                return f"Invalid type of input1: {type(input1)}"
        if input2 is not None:
            if not isinstance(input2, (torch.Tensor, float)):
                return f"Invalid type of input2: {type(input2)}"

        return True

class TestCustomValidation2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("IMAGE,FLOAT",),
                "input2": ("IMAGE,FLOAT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "custom_validation2"

    CATEGORY = "Testing/Nodes"

    def custom_validation2(self, input1, input2):
        if isinstance(input1, float) and isinstance(input2, float):
            result = torch.ones([1, 512, 512, 3]) * input1 * input2
        else:
            result = input1 * input2
        return (result,)

    @classmethod
    def VALIDATE_INPUTS(cls, input_types, input1=None, input2=None):
        if input1 is not None:
            if not isinstance(input1, (torch.Tensor, float)):
                return f"Invalid type of input1: {type(input1)}"
        if input2 is not None:
            if not isinstance(input2, (torch.Tensor, float)):
                return f"Invalid type of input2: {type(input2)}"

        if 'input1' in input_types:
            if input_types['input1'] not in ["IMAGE", "FLOAT"]:
                return f"Invalid type of input1: {input_types['input1']}"
        if 'input2' in input_types:
            if input_types['input2'] not in ["IMAGE", "FLOAT"]:
                return f"Invalid type of input2: {input_types['input2']}"

        return True

@VariantSupport()
class TestCustomValidation3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("IMAGE,FLOAT",),
                "input2": ("IMAGE,FLOAT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "custom_validation3"

    CATEGORY = "Testing/Nodes"

    def custom_validation3(self, input1, input2):
        if isinstance(input1, float) and isinstance(input2, float):
            result = torch.ones([1, 512, 512, 3]) * input1 * input2
        else:
            result = input1 * input2
        return (result,)

class TestCustomValidation4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("FLOAT",),
                "input2": ("FLOAT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "custom_validation4"

    CATEGORY = "Testing/Nodes"

    def custom_validation4(self, input1, input2):
        result = torch.ones([1, 512, 512, 3]) * input1 * input2
        return (result,)

    @classmethod
    def VALIDATE_INPUTS(cls, input1, input2):
        if input1 is not None:
            if not isinstance(input1, float):
                return f"Invalid type of input1: {type(input1)}"
        if input2 is not None:
            if not isinstance(input2, float):
                return f"Invalid type of input2: {type(input2)}"

        return True

class TestCustomValidation5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("FLOAT", {"min": 0.0, "max": 1.0}),
                "input2": ("FLOAT", {"min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "custom_validation5"

    CATEGORY = "Testing/Nodes"

    def custom_validation5(self, input1, input2):
        value = input1 * input2
        return (torch.ones([1, 512, 512, 3]) * value,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if kwargs['input2'] == 7.0:
            return "7s are not allowed. I've never liked 7s."
        return True

class TestDynamicDependencyCycle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("IMAGE",),
                "input2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dynamic_dependency_cycle"

    CATEGORY = "Testing/Nodes"

    def dynamic_dependency_cycle(self, input1, input2):
        g = GraphBuilder()
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)
        mix1 = g.node("TestLazyMixImages", image1=input1, mask=mask.out(0))
        mix2 = g.node("TestLazyMixImages", image1=mix1.out(0), image2=input2, mask=mask.out(0))

        # Create the cyle
        mix1.set_input("image2", mix2.out(0))

        return {
            "result": (mix2.out(0),),
            "expand": g.finalize(),
        }

class TestMixedExpansionReturns:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("FLOAT",),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE")
    FUNCTION = "mixed_expansion_returns"

    CATEGORY = "Testing/Nodes"

    def mixed_expansion_returns(self, input1):
        white_image = torch.ones([1, 512, 512, 3])
        if input1 <= 0.1:
            return (torch.ones([1, 512, 512, 3]) * 0.1, white_image)
        elif input1 <= 0.2:
            return {
                "result": (torch.ones([1, 512, 512, 3]) * 0.2, white_image),
            }
        else:
            g = GraphBuilder()
            mask = g.node("StubMask", value=0.3, height=512, width=512, batch_size=1)
            black = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
            white = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
            mix = g.node("TestLazyMixImages", image1=black.out(0), image2=white.out(0), mask=mask.out(0))
            return {
                "result": (mix.out(0), white_image),
                "expand": g.finalize(),
            }

TEST_NODE_CLASS_MAPPINGS = {
    "TestLazyMixImages": TestLazyMixImages,
    "TestVariadicAverage": TestVariadicAverage,
    "TestCustomIsChanged": TestCustomIsChanged,
    "TestIsChangedWithConstants": TestIsChangedWithConstants,
    "TestCustomValidation1": TestCustomValidation1,
    "TestCustomValidation2": TestCustomValidation2,
    "TestCustomValidation3": TestCustomValidation3,
    "TestCustomValidation4": TestCustomValidation4,
    "TestCustomValidation5": TestCustomValidation5,
    "TestDynamicDependencyCycle": TestDynamicDependencyCycle,
    "TestMixedExpansionReturns": TestMixedExpansionReturns,
}

TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestLazyMixImages": "Lazy Mix Images",
    "TestVariadicAverage": "Variadic Average",
    "TestCustomIsChanged": "Custom IsChanged",
    "TestIsChangedWithConstants": "IsChanged With Constants",
    "TestCustomValidation1": "Custom Validation 1",
    "TestCustomValidation2": "Custom Validation 2",
    "TestCustomValidation3": "Custom Validation 3",
    "TestCustomValidation4": "Custom Validation 4",
    "TestCustomValidation5": "Custom Validation 5",
    "TestDynamicDependencyCycle": "Dynamic Dependency Cycle",
    "TestMixedExpansionReturns": "Mixed Expansion Returns",
}
