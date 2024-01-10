

class LogicEqual:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value0": ("ANY_DATA",),
            },
            "optional": {
                "value1": ("ANY_DATA",)
            }
        }
        
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "is_equal"

    CATEGORY = "aiyoh/logic"
    
    
    def is_equal(self, value0, value1):
        res = False
        try:
            res = (value0 ==value1)
        except Exception as e:
            res = False
            
        return (res,)
    
    
    
class ChooseOne:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bool_val": ("BOOLEAN", {"default": True}),
                "on_true": ("ANY_DATA",),
            },
            "optional": {
                "on_false": ("ANY_DATA",)
            }
        }
        
    RETURN_TYPES = ("ANY_DATA",)
    FUNCTION = "choose"

    CATEGORY = "aiyoh/logic"
    
    
    def is_equal(self, bool_val, on_true, on_false = None):
        if bool_val:
            res = on_true
        else:
            res = on_false    
        return (res,)
    

    
NODE_CLASS_MAPPINGS = {
    "LogicEqual": LogicEqual,
    "ChooseOne": ChooseOne
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogicEqual": "Logic Equal",
    "ChooseOne": "Choose One"
} 