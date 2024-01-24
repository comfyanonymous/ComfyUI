

class MathAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value0": ("ANY_DATA",),
                "value1": ("ANY_DATA",)
            }
        }
        
    RETURN_TYPES = ("ANY_DATA",)
    FUNCTION = "add"

    CATEGORY = "aiyoh/math"
    
    
    def add(self, value0, value1):
        res = value0 + value1
        return (res,)
    
    
    
class MathSub:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value0": ("ANY_DATA",),
                "value1": ("ANY_DATA",)
            }
        }
        
    RETURN_TYPES = ("ANY_DATA",)
    FUNCTION = "sub"

    CATEGORY = "aiyoh/math"
    
    
    def sub(self, value0, value1):
        res = value0 - value1
        return (res,)
    
    
class MathDiv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value0": ("ANY_DATA",),
                "value1": ("ANY_DATA",)
            },
            "optional": {
                "return_int": ("BOOLEAN", {"default": True})
            }
        }
        
    RETURN_TYPES = ("ANY_DATA",)
    FUNCTION = "sub"

    CATEGORY = "aiyoh/math"
    
    
    def sub(self, value0, value1, return_int):
        res = value0 / value1
        if return_int:
            res = int(res)
        return (res,)


NODE_CLASS_MAPPINGS = {
    "MathAdd": MathAdd,
    "MathSub": MathSub,
    "MathDiv": MathDiv
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathAdd": "Math Add",
    "MathSub": "Math Sub",
    "MathDiv": "Math Div"
}


