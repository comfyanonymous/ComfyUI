import torch

class TestLazyMixImages:
    def __init__(self):
        pass

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
    def __init__(self):
        pass

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
    def __init__(self):
        pass

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

TEST_NODE_CLASS_MAPPINGS = {
    "TestLazyMixImages": TestLazyMixImages,
    "TestVariadicAverage": TestVariadicAverage,
    "TestCustomIsChanged": TestCustomIsChanged,
}

TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestLazyMixImages": "Lazy Mix Images",
    "TestVariadicAverage": "Variadic Average",
    "TestCustomIsChanged": "Custom IsChanged",
}
