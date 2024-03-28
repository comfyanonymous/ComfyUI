import torch
from .tools import VariantSupport

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

    def check_lazy_status(self, mask, image1 = None, image2 = None):
        mask_min = mask.min()
        mask_max = mask.max()
        needed = []
        if image1 is None and (mask_min != 1.0 or mask_max != 1.0):
            needed.append("image1")
        if image2 is None and (mask_min != 0.0 or mask_max != 0.0):
            needed.append("image2")
        return needed

    # Not trying to handle different batch sizes here just to keep the demo simple
    def mix(self, mask, image1 = None, image2 = None):
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
        print(result[0])
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

TEST_NODE_CLASS_MAPPINGS = {
    "TestLazyMixImages": TestLazyMixImages,
    "TestVariadicAverage": TestVariadicAverage,
    "TestCustomIsChanged": TestCustomIsChanged,
    "TestCustomValidation1": TestCustomValidation1,
    "TestCustomValidation2": TestCustomValidation2,
    "TestCustomValidation3": TestCustomValidation3,
}

TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestLazyMixImages": "Lazy Mix Images",
    "TestVariadicAverage": "Variadic Average",
    "TestCustomIsChanged": "Custom IsChanged",
    "TestCustomValidation1": "Custom Validation 1",
    "TestCustomValidation2": "Custom Validation 2",
    "TestCustomValidation3": "Custom Validation 3",
}
