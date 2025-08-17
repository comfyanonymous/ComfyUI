# ComfyUI - mxToolkit - Max Smirnov 2024
import nodes

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class mxSeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "X": ("INT", {"default": 0, "min": 0, "max": 4294967296}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("X",)

    FUNCTION = "main"
    CATEGORY = 'utils/mxToolkit'

    def main(self, X,):
        return (X,)


class mxStop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "In": (any,),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    RETURN_TYPES = (any,)

    FUNCTION = "main"
    CATEGORY = 'utils/mxToolkit'

    def main(self, In):
        out = In;
        nodes.interrupt_processing();
        return (out,)

class mxSlider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Xi": ("INT", {"default": 20, "min": -4294967296, "max": 4294967296}),
                "Xf": ("FLOAT", {"default": 20, "min": -4294967296, "max": 4294967296}),
                "isfloatX": ("INT", {"default": 0, "min": 0, "max": 1}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("X",)

    FUNCTION = "main"
    CATEGORY = 'utils/mxToolkit'

    def main(self, Xi, Xf, isfloatX):
        if isfloatX > 0:
            out = Xf
        else:
            out = Xi
        return (out,)

class mxSlider2D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Xi": ("INT", {"default": 512, "min": -4294967296, "max": 4294967296}),
                "Xf": ("FLOAT", {"default": 512, "min": -4294967296, "max": 4294967296}),
                "Yi": ("INT", {"default": 512, "min": -4294967296, "max": 4294967296}),
                "Yf": ("FLOAT", {"default": 512, "min": -4294967296, "max": 4294967296}),
                "isfloatX": ("INT", {"default": 0, "min": 0, "max": 1}),
                "isfloatY": ("INT", {"default": 0, "min": 0, "max": 1}),
            },
        }

    RETURN_TYPES = (any, any,)
    RETURN_NAMES = ("X","Y",)

    FUNCTION = "main"
    CATEGORY = 'utils/mxToolkit'

    def main(self, Xi, Xf, isfloatX, Yi, Yf, isfloatY):
        if isfloatX > 0:
            outX = Xf
        else:
            outX = Xi
        if isfloatY > 0:
            outY = Yf
        else:
            outY = Yi
        return (outX, outY,)


NODE_CLASS_MAPPINGS = {
    "mxSeed": mxSeed,
    "mxStop": mxStop,
    "mxSlider": mxSlider,
    "mxSlider2D": mxSlider2D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mxSeed": "Seed",
    "mxStop": "Stop",
    "mxSlider": "Slider",
    "mxSlider2D": "Slider 2D",
}