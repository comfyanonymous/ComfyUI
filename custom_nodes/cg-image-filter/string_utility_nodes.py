from comfy.comfy_types.node_typing import IO

class SplitByCommas:
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING")
    FUNCTION = "func"
    CATEGORY = "image_filter/helpers"
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = [False, False, False, False, False, True]

    DESCRIPTION = "Split the input string into up to five pieces. Splits on commas (or | or ^) and then strips whitespace from front and end."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "string" : ("STRING", {"default":""}), },
            "optional": { "split": ([",", "|", "^"], {}), },
        }
    
    def func(self, string:str, split:str=","):
        bits = [r.strip() for r in string.split(split)] 
        as_list = [b for b in bits]
        if len(bits)<=5: 
            bits += ["",]*(5-len(bits))
        else:
            bits = bits[:4] + [",".join(bits[4:]),]

        bits.append(as_list)
        return tuple(bits)
    
class AnyListToString:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"
    CATEGORY = "image_filter/helpers"
    INPUT_IS_LIST  = True
    OUTPUT_IS_LIST = (False,) 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "anything" : (IO.ANY, ), 
                "join" : ("STRING", {"default":""}), 
            }
        }
    
    def func(self, anything, join:str):
        return ( join[0].join( [f"{x}" for x in anything] ), )
    
class StringToInt:
    RETURN_TYPES = ("INT",)
    FUNCTION = "func"
    CATEGORY = "image_filter/helpers"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "string" : ("STRING", {"default":"", "forceInput":True, "tooltip":"whitespace will be stripped before parsing"}), 
                "default" : ("INT", {"default":0, "tooltip":"used if the string can't be parsed as an integer"}), 
            }
        }
    
    def func(self, string:str, default:int):
        try:    return (int(string.strip()),)
        except: return (default,)

class StringToFloat:
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "func"
    CATEGORY = "image_filter/helpers"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "string" : ("STRING", {"default":"", "forceInput":True, "tooltip":"whitespace will be stripped before parsing"}), 
                "default" : ("FLOAT", {"default":0, "tooltip":"used if the string can't be parsed as a float"}), 
            }
        }
    
    def func(self, string:str, default:float):
        try:    return (float(string.strip()),)
        except: return (default,)