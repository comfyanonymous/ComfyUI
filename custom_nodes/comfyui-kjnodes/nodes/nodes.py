import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json, re, os, io, time
import re
import importlib

from comfy import model_management
import folder_paths
from nodes import MAX_RESOLUTION
from comfy.utils import common_upscale, ProgressBar, load_torch_file
from comfy.comfy_types.node_typing import IO

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_paths.add_model_folder_path("kjnodes_fonts", os.path.join(script_directory, "fonts"))

class BOOLConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("BOOLEAN", {"default": True}),
        },
        }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "KJNodes/constants"

    def get_value(self, value):
        return (value,)
    
class INTConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
        },
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "KJNodes/constants"

    def get_value(self, value):
        return (value,)

class FloatConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("FLOAT", {"default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.00001}),
        },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "KJNodes/constants"

    def get_value(self, value):
        return (round(value, 6),)

class StringConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": '', "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "passtring"
    CATEGORY = "KJNodes/constants"

    def passtring(self, string):
        return (string, )

class StringConstantMultiline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "", "multiline": True}),
                "strip_newlines": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "stringify"
    CATEGORY = "KJNodes/constants"

    def stringify(self, string, strip_newlines):
        new_string = []
        for line in io.StringIO(string):
            if not line.strip().startswith("\n") and strip_newlines:
                line = line.replace("\n", '')
            new_string.append(line)
        new_string = "\n".join(new_string)

        return (new_string, )


    
class ScaleBatchPromptSchedule:
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "scaleschedule"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
Scales a batch schedule from Fizz' nodes BatchPromptSchedule
to a different frame count.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "input_str": ("STRING", {"forceInput": True,"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n"}),
                 "old_frame_count": ("INT", {"forceInput": True,"default": 1,"min": 1, "max": 4096, "step": 1}),
                 "new_frame_count": ("INT", {"forceInput": True,"default": 1,"min": 1, "max": 4096, "step": 1}),
                
        },
    } 
    
    def scaleschedule(self, old_frame_count, input_str, new_frame_count):
        pattern = r'"(\d+)"\s*:\s*"(.*?)"(?:,|\Z)'
        frame_strings = dict(re.findall(pattern, input_str))
        
        # Calculate the scaling factor
        scaling_factor = (new_frame_count - 1) / (old_frame_count - 1)
        
        # Initialize a dictionary to store the new frame numbers and strings
        new_frame_strings = {}
        
        # Iterate over the frame numbers and strings
        for old_frame, string in frame_strings.items():
            # Calculate the new frame number
            new_frame = int(round(int(old_frame) * scaling_factor))
            
            # Store the new frame number and corresponding string
            new_frame_strings[new_frame] = string
        
        # Format the output string
        output_str = ', '.join([f'"{k}":"{v}"' for k, v in sorted(new_frame_strings.items())])
        return (output_str,)


class GetLatentsFromBatchIndexed:
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "indexedlatentsfrombatch"
    CATEGORY = "KJNodes/latents"
    DESCRIPTION = """
Selects and returns the latents at the specified indices as an latent batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "latents": ("LATENT",),
                 "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
                 "latent_format": (["BCHW", "BTCHW", "BCTHW"], {"default": "BCHW"}),
        },
    } 
    
    def indexedlatentsfrombatch(self, latents, indexes, latent_format):
        
        samples = latents.copy()
        latent_samples = samples["samples"] 

        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
        # Select the latents at the specified indices
        if latent_format == "BCHW":
            chosen_latents = latent_samples[indices_tensor]
        elif latent_format == "BTCHW":
            chosen_latents = latent_samples[:, indices_tensor]
        elif latent_format == "BCTHW":
            chosen_latents = latent_samples[:, :, indices_tensor]

        samples["samples"] = chosen_latents
        return (samples,)
    

class ConditioningMultiCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1}),
                "operation": (["combine", "concat"], {"default": "combine"}),
                "conditioning_1": ("CONDITIONING", ),
                "conditioning_2": ("CONDITIONING", ),
            },
    }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("combined", "inputcount")
    FUNCTION = "combine"
    CATEGORY = "KJNodes/masking/conditioning"
    DESCRIPTION = """
Combines multiple conditioning nodes into one
"""

    def combine(self, inputcount, operation, **kwargs):
        from nodes import ConditioningCombine
        from nodes import ConditioningConcat
        cond_combine_node = ConditioningCombine()
        cond_concat_node = ConditioningConcat()
        cond = kwargs["conditioning_1"]
        for c in range(1, inputcount):
            new_cond = kwargs[f"conditioning_{c + 1}"]
            if operation == "combine":
                cond = cond_combine_node.combine(new_cond, cond)[0]
            elif operation == "concat":
                cond = cond_concat_node.concat(cond, new_cond)[0]
        return (cond, inputcount,)

class AppendStringsToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": '', "forceInput": True}),
                "string2": ("STRING", {"default": '', "forceInput": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "joinstring"
    CATEGORY = "KJNodes/text"

    def joinstring(self, string1, string2):
        if not isinstance(string1, list):
            string1 = [string1]
        if not isinstance(string2, list):
            string2 = [string2]
        
        joined_string = string1 + string2
        return (joined_string, )
    
class JoinStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": '', "forceInput": True}),
                "string2": ("STRING", {"default": '', "forceInput": True}),
                "delimiter": ("STRING", {"default": ' ', "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "joinstring"
    CATEGORY = "KJNodes/text"

    def joinstring(self, string1, string2, delimiter):
        joined_string = string1 + delimiter + string2
        return (joined_string, )
    
class JoinStringMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "string_1": ("STRING", {"default": '', "forceInput": True}),
                "string_2": ("STRING", {"default": '', "forceInput": True}),
                "delimiter": ("STRING", {"default": ' ', "multiline": False}),
                "return_list": ("BOOLEAN", {"default": False}),
            },
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Creates single string, or a list of strings, from  
multiple input strings.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, delimiter, **kwargs):
        string = kwargs["string_1"]
        return_list = kwargs["return_list"]
        strings = [string] # Initialize a list with the first string
        for c in range(1, inputcount):
            new_string = kwargs[f"string_{c + 1}"]
            if return_list:
                strings.append(new_string) # Add new string to the list
            else:
                string = string + delimiter + new_string
        if return_list:
            return (strings,) # Return the list of strings
        else:
            return (string,) # Return the combined string

class CondPassThrough:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
            }, 
    }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "passthrough"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
    Simply passes through the positive and negative conditioning,
    workaround for Set node not allowing bypassed inputs.
"""

    def passthrough(self, positive=None, negative=None):
        return (positive, negative,)

class ModelPassThrough:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {             
            },
            "optional": {
                "model": ("MODEL", ),
            }, 
    }

    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model",)
    FUNCTION = "passthrough"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
    Simply passes through the model,
    workaround for Set node not allowing bypassed inputs.
"""

    def passthrough(self, model=None):
            return (model,)

def append_helper(t, mask, c, set_area_to_bounds, strength):
        n = [t[0], t[1].copy()]
        _, h, w = mask.shape
        n[1]['mask'] = mask
        n[1]['set_area_to_bounds'] = set_area_to_bounds
        n[1]['mask_strength'] = strength
        c.append(n)  

class ConditioningSetMaskAndCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"
    DESCRIPTION = """
Bundles multiple conditioning mask and combine nodes into one,functionality is identical to ComfyUI native nodes
"""

    def append(self, positive_1, negative_1, positive_2, negative_2, mask_1, mask_2, set_cond_area, mask_1_strength, mask_2_strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        return (c, c2)

class ConditioningSetMaskAndCombine3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "positive_3": ("CONDITIONING", ),
                "negative_3": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"
    DESCRIPTION = """
Bundles multiple conditioning mask and combine nodes into one,functionality is identical to ComfyUI native nodes
"""

    def append(self, positive_1, negative_1, positive_2, positive_3, negative_2, negative_3, mask_1, mask_2, mask_3, set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        if len(mask_3.shape) < 3:
            mask_3 = mask_3.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in positive_3:
            append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        for t in negative_3:
            append_helper(t, mask_3, c2, set_area_to_bounds, mask_3_strength)
        return (c, c2)

class ConditioningSetMaskAndCombine4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "positive_3": ("CONDITIONING", ),
                "negative_3": ("CONDITIONING", ),
                "positive_4": ("CONDITIONING", ),
                "negative_4": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_4": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"
    DESCRIPTION = """
Bundles multiple conditioning mask and combine nodes into one,functionality is identical to ComfyUI native nodes
"""

    def append(self, positive_1, negative_1, positive_2, positive_3, positive_4, negative_2, negative_3, negative_4, mask_1, mask_2, mask_3, mask_4, set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        if len(mask_3.shape) < 3:
            mask_3 = mask_3.unsqueeze(0)
        if len(mask_4.shape) < 3:
            mask_4 = mask_4.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in positive_3:
            append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
        for t in positive_4:
            append_helper(t, mask_4, c, set_area_to_bounds, mask_4_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        for t in negative_3:
            append_helper(t, mask_3, c2, set_area_to_bounds, mask_3_strength)
        for t in negative_4:
            append_helper(t, mask_4, c2, set_area_to_bounds, mask_4_strength)
        return (c, c2)

class ConditioningSetMaskAndCombine5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "positive_3": ("CONDITIONING", ),
                "negative_3": ("CONDITIONING", ),
                "positive_4": ("CONDITIONING", ),
                "negative_4": ("CONDITIONING", ),
                "positive_5": ("CONDITIONING", ),
                "negative_5": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_4": ("MASK", ),
                "mask_5": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_5_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"
    DESCRIPTION = """
Bundles multiple conditioning mask and combine nodes into one,functionality is identical to ComfyUI native nodes
"""

    def append(self, positive_1, negative_1, positive_2, positive_3, positive_4, positive_5, negative_2, negative_3, negative_4, negative_5, mask_1, mask_2, mask_3, mask_4, mask_5, set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength, mask_5_strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        if len(mask_3.shape) < 3:
            mask_3 = mask_3.unsqueeze(0)
        if len(mask_4.shape) < 3:
            mask_4 = mask_4.unsqueeze(0)
        if len(mask_5.shape) < 3:
            mask_5 = mask_5.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in positive_3:
            append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
        for t in positive_4:
            append_helper(t, mask_4, c, set_area_to_bounds, mask_4_strength)
        for t in positive_5:
            append_helper(t, mask_5, c, set_area_to_bounds, mask_5_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        for t in negative_3:
            append_helper(t, mask_3, c2, set_area_to_bounds, mask_3_strength)
        for t in negative_4:
            append_helper(t, mask_4, c2, set_area_to_bounds, mask_4_strength)
        for t in negative_5:
            append_helper(t, mask_5, c2, set_area_to_bounds, mask_5_strength)
        return (c, c2)
    
class VRAM_Debug:
    
    @classmethod
    
    def INPUT_TYPES(s):
      return {
        "required": {
            
            "empty_cache": ("BOOLEAN", {"default": True}),
            "gc_collect": ("BOOLEAN", {"default": True}),
            "unload_all_models": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "any_input": (IO.ANY,),
            "image_pass": ("IMAGE",),
            "model_pass": ("MODEL",),
        }
	}
        
    RETURN_TYPES = (IO.ANY, "IMAGE","MODEL","INT", "INT",)
    RETURN_NAMES = ("any_output", "image_pass", "model_pass", "freemem_before", "freemem_after")
    FUNCTION = "VRAMdebug"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
Returns the inputs unchanged, they are only used as triggers,  
and performs comfy model management functions and garbage collection,  
reports free VRAM before and after the operations.
"""

    def VRAMdebug(self, gc_collect, empty_cache, unload_all_models, image_pass=None, model_pass=None, any_input=None):
        freemem_before = model_management.get_free_memory()
        print("VRAMdebug: free memory before: ", f"{freemem_before:,.0f}")
        if empty_cache:
            model_management.soft_empty_cache()
        if unload_all_models:
            model_management.unload_all_models()
        if gc_collect:
            import gc
            gc.collect()
        freemem_after = model_management.get_free_memory()
        print("VRAMdebug: free memory after: ", f"{freemem_after:,.0f}")
        print("VRAMdebug: freed memory: ", f"{freemem_after - freemem_before:,.0f}")
        return {"ui": {
            "text": [f"{freemem_before:,.0f}x{freemem_after:,.0f}"]}, 
            "result": (any_input, image_pass, model_pass, freemem_before, freemem_after) 
        }

class SomethingToString:
    @classmethod
    
    def INPUT_TYPES(s):
     return {
        "required": {
        "input": (IO.ANY, ),
    },
    "optional": {
        "prefix": ("STRING", {"default": ""}),
        "suffix": ("STRING", {"default": ""}),
    }
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "stringify"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Converts any type to a string.
"""

    def stringify(self, input, prefix="", suffix=""):
        if isinstance(input, (int, float, bool)):
            stringified = str(input)
        elif isinstance(input, list):
            stringified = ', '.join(str(item) for item in input)
        else:
            return
        if prefix: # Check if prefix is not empty
            stringified = prefix + stringified # Add the prefix
        if suffix: # Check if suffix is not empty
            stringified = stringified + suffix # Add the suffix

        return (stringified,)

class Sleep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (IO.ANY, ),
                "minutes": ("INT", {"default": 0, "min": 0, "max": 1439}),
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 59.99, "step": 0.01}),
            },
        }
    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "sleepdelay"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
Delays the execution for the input amount of time.
"""

    def sleepdelay(self, input, minutes, seconds):
        total_seconds = minutes * 60 + seconds
        time.sleep(total_seconds)
        return input,
    
class EmptyLatentImagePresets:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
        "required": {
             "dimensions": (
            [
                '512 x 512 (1:1)',
                '768 x 512 (1.5:1)',
                '960 x 512 (1.875:1)',
                '1024 x 512 (2:1)',
                '1024 x 576 (1.778:1)',
                '1536 x 640 (2.4:1)',
                '1344 x 768 (1.75:1)',
                '1216 x 832 (1.46:1)',
                '1152 x 896 (1.286:1)',
                '1024 x 1024 (1:1)',
            ],
            {
            "default": '512 x 512 (1:1)'
             }),
           
            "invert": ("BOOLEAN", {"default": False}),
            "batch_size": ("INT", {
            "default": 1,
            "min": 1,
            "max": 4096
            }),
        },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("Latent", "Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "KJNodes/latents"

    def generate(self, dimensions, invert, batch_size):
        from nodes import EmptyLatentImage
        result = [x.strip() for x in dimensions.split('x')]

        # Remove the aspect ratio part
        result[0] = result[0].split('(')[0].strip()
        result[1] = result[1].split('(')[0].strip()
        
        if invert:
            width = int(result[1].split(' ')[0])
            height = int(result[0])
        else:
            width = int(result[0])
            height = int(result[1].split(' ')[0])
        latent = EmptyLatentImage().generate(width, height, batch_size)[0]

        return (latent, int(width), int(height),)

class EmptyLatentImageCustomPresets:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            with open(os.path.join(script_directory, 'custom_dimensions.json')) as f:
                dimensions_dict = json.load(f)
        except FileNotFoundError:
            dimensions_dict = []
        return {
        "required": {
            "dimensions": (
                 [f"{d['label']} - {d['value']}" for d in dimensions_dict],
            ),
           
            "invert": ("BOOLEAN", {"default": False}),
            "batch_size": ("INT", {
            "default": 1,
            "min": 1,
            "max": 4096
            }),
        },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("Latent", "Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "KJNodes/latents"
    DESCRIPTION = """
Generates an empty latent image with the specified dimensions.  
The choices are loaded from 'custom_dimensions.json' in the nodes folder.
"""

    def generate(self, dimensions, invert, batch_size):
       from nodes import EmptyLatentImage
       # Split the string into label and value
       label, value = dimensions.split(' - ')
       # Split the value into width and height
       width, height = [x.strip() for x in value.split('x')]
   
       if invert:
           width, height = height, width
   
       latent = EmptyLatentImage().generate(int(width), int(height), batch_size)[0]
   
       return (latent, int(width), int(height),)

class WidgetToString:
    @classmethod
    def IS_CHANGED(cls,*,id,node_title,any_input,**kwargs):
        if any_input is not None and (id != 0 or node_title != ""):
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "id": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "widget_name": ("STRING", {"multiline": False}),
                "return_all": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                         "any_input": (IO.ANY, ),
                         "node_title": ("STRING", {"multiline": False}),
                         "allowed_float_decimals": ("INT", {"default": 2, "min": 0, "max": 10, "tooltip": "Number of decimal places to display for float values"}),
                         
                         },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT",
                       "unique_id": "UNIQUE_ID",},
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "get_widget_value"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Selects a node and it's specified widget and outputs the value as a string.  
If no node id or title is provided it will use the 'any_input' link and use that node.  
To see node id's, enable node id display from Manager badge menu.  
Alternatively you can search with the node title. Node titles ONLY exist if they  
are manually edited!  
The 'any_input' is required for making sure the node you want the value from exists in the workflow.
"""

    def get_widget_value(self, id, widget_name, extra_pnginfo, prompt, unique_id, return_all=False, any_input=None, node_title="", allowed_float_decimals=2):
        workflow = extra_pnginfo["workflow"]
        #print(json.dumps(workflow, indent=4))
        results = []
        node_id = None  # Initialize node_id to handle cases where no match is found
        link_id = None
        link_to_node_map = {}

        for node in workflow["nodes"]:
            if node_title:
                if "title" in node:
                    if node["title"] == node_title:
                        node_id = node["id"]
                        break
                else:
                    print("Node title not found.")
            elif id != 0:
                if node["id"] == id:
                    node_id = id
                    break
            elif any_input is not None:
                if node["type"] == "WidgetToString" and node["id"] == int(unique_id) and not link_id:
                    for node_input in node["inputs"]:
                        if node_input["name"] == "any_input":
                            link_id = node_input["link"]
                    
                # Construct a map of links to node IDs for future reference
                node_outputs = node.get("outputs", None)
                if not node_outputs:
                    continue
                for output in node_outputs:
                    node_links = output.get("links", None)
                    if not node_links:
                        continue
                    for link in node_links:
                        link_to_node_map[link] = node["id"]
                        if link_id and link == link_id:
                            break
        
        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("No matching node found for the given title or id")

        values = prompt[str(node_id)]
        if "inputs" in values:
            if return_all:
                # Format items based on type
                formatted_items = []
                for k, v in values["inputs"].items():
                    if isinstance(v, float):
                        item = f"{k}: {v:.{allowed_float_decimals}f}"
                    else:
                        item = f"{k}: {str(v)}"
                    formatted_items.append(item)
                results.append(', '.join(formatted_items))
            elif widget_name in values["inputs"]:
                v = values["inputs"][widget_name]
                if isinstance(v, float):
                    v = f"{v:.{allowed_float_decimals}f}"
                else:
                    v = str(v)
                return (v, )
            else:
                raise NameError(f"Widget not found: {node_id}.{widget_name}")
        return (', '.join(results).strip(', '), )

class DummyOut:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "any_input": (IO.ANY, ),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "dummy"
    CATEGORY = "KJNodes/misc"
    OUTPUT_NODE = True
    DESCRIPTION = """
Does nothing, used to trigger generic workflow output.    
A way to get previews in the UI without saving anything to disk.
"""

    def dummy(self, any_input):
        return (any_input,)
    
class FlipSigmasAdjusted:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     "divide_by_last_sigma": ("BOOLEAN", {"default": False}),
                     "divide_by": ("FLOAT", {"default": 1,"min": 1, "max": 255, "step": 0.01}),
                     "offset_by": ("INT", {"default": 1,"min": -100, "max": 100, "step": 1}),
                     }
                }
    RETURN_TYPES = ("SIGMAS", "STRING",)
    RETURN_NAMES = ("SIGMAS", "sigmas_string",)
    CATEGORY = "KJNodes/noise"
    FUNCTION = "get_sigmas_adjusted"

    def get_sigmas_adjusted(self, sigmas, divide_by_last_sigma, divide_by, offset_by):
        
        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001
        adjusted_sigmas = sigmas.clone()
        #offset sigma
        for i in range(1, len(sigmas)):
            offset_index = i - offset_by
            if 0 <= offset_index < len(sigmas):
                adjusted_sigmas[i] = sigmas[offset_index]
            else:
                adjusted_sigmas[i] = 0.0001 
        if adjusted_sigmas[0] == 0:
            adjusted_sigmas[0] = 0.0001  
        if divide_by_last_sigma:
            adjusted_sigmas = adjusted_sigmas / adjusted_sigmas[-1]

        sigma_np_array = adjusted_sigmas.numpy()
        array_string = np.array2string(sigma_np_array, precision=2, separator=', ', threshold=np.inf)
        adjusted_sigmas = adjusted_sigmas / divide_by
        return (adjusted_sigmas, array_string,)
    
class CustomSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "sigmas_string" :("STRING", {"default": "14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029","multiline": True}),
                     "interpolate_to_steps": ("INT", {"default": 10,"min": 0, "max": 255, "step": 1}),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    CATEGORY = "KJNodes/noise"
    FUNCTION = "customsigmas"
    DESCRIPTION = """
Creates a sigmas tensor from a string of comma separated values.  
Examples: 
   
Nvidia's optimized AYS 10 step schedule for SD 1.5:  
14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029  
SDXL:   
14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029  
SVD:  
700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002  
"""
    def customsigmas(self, sigmas_string, interpolate_to_steps):
        sigmas_list = sigmas_string.split(', ')
        sigmas_float_list = [float(sigma) for sigma in sigmas_list]
        sigmas_tensor = torch.FloatTensor(sigmas_float_list)
        if len(sigmas_tensor) != interpolate_to_steps + 1:
            sigmas_tensor = self.loglinear_interp(sigmas_tensor, interpolate_to_steps + 1)
        sigmas_tensor[-1] = 0
        return (sigmas_tensor.float(),)
     
    def loglinear_interp(self, t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        t_steps_np = t_steps.numpy()

        xs = np.linspace(0, 1, len(t_steps_np))
        ys = np.log(t_steps_np[::-1])
        
        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        
        interped_ys = np.exp(new_ys)[::-1].copy()
        interped_ys_tensor = torch.tensor(interped_ys)
        return interped_ys_tensor
    
class StringToFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "string" :("STRING", {"default": "1, 2, 3", "multiline": True}),
                     }
                }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    CATEGORY = "KJNodes/misc"
    FUNCTION = "createlist"

    def createlist(self, string):
        float_list = [float(x.strip()) for x in string.split(',')]
        return (float_list,)

 
class InjectNoiseToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents":("LATENT",),  
            "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 200.0, "step": 0.0001}),
            "noise":  ("LATENT",),
            "normalize": ("BOOLEAN", {"default": False}),
            "average": ("BOOLEAN", {"default": False}),
            },
            "optional":{
                "mask": ("MASK", ),
                "mix_randn_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
            }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "injectnoise"
    CATEGORY = "KJNodes/noise"
        
    def injectnoise(self, latents, strength, noise, normalize, average, mix_randn_amount=0, seed=None, mask=None):
        samples = latents["samples"].clone().cpu()
        noise = noise["samples"].clone().cpu()
        if samples.shape != samples.shape:
            raise ValueError("InjectNoiseToLatent: Latent and noise must have the same shape")
        if average:
            noised = (samples + noise) / 2
        else:
            noised = samples + noise * strength
        if normalize:
            noised = noised / noised.std()
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(noised.shape[2], noised.shape[3]), mode="bilinear")
            mask = mask.expand((-1,noised.shape[1],-1,-1))
            if mask.shape[0] < noised.shape[0]:
                mask = mask.repeat((noised.shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:noised.shape[0]]
            noised = mask * noised + (1-mask) * samples
        if mix_randn_amount > 0:
            if seed is not None:
                generator = torch.manual_seed(seed)
                rand_noise = torch.randn(noised.size(), dtype=noised.dtype, layout=noised.layout, generator=generator, device="cpu")
                noised = noised + (mix_randn_amount * rand_noise)
        
        return ({"samples":noised},)
 
class SoundReactive:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "sound_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 99999, "step": 0.01}),
            "start_range_hz": ("INT", {"default": 150, "min": 0, "max": 9999, "step": 1}),
            "end_range_hz": ("INT", {"default": 2000, "min": 0, "max": 9999, "step": 1}),
            "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 99999, "step": 0.01}),
            "smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "normalize": ("BOOLEAN", {"default": False}),
            },
            }
    
    RETURN_TYPES = ("FLOAT","INT",)
    RETURN_NAMES =("sound_level", "sound_level_int",)
    FUNCTION = "react"
    CATEGORY = "KJNodes/audio"
    DESCRIPTION = """
Reacts to the sound level of the input.  
Uses your browsers sound input options and requires.  
Meant to be used with realtime diffusion with autoqueue.
"""
        
    def react(self, sound_level, start_range_hz, end_range_hz, smoothing_factor, multiplier, normalize):

        sound_level *= multiplier

        if normalize:
            sound_level /= 255

        sound_level_int = int(sound_level)
        return (sound_level, sound_level_int, )     
       
class GenerateNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "multiplier": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 4096, "step": 0.01}),
            "constant_batch_noise": ("BOOLEAN", {"default": False}),
            "normalize": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model": ("MODEL", ),
                "sigmas": ("SIGMAS", ),
                "latent_channels": (['4', '16', ],),
                "shape": (["BCHW", "BCTHW","BTCHW",],),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generatenoise"
    CATEGORY = "KJNodes/noise"
    DESCRIPTION = """
Generates noise for injection or to be used as empty latents on samplers with add_noise off.
"""
        
    def generatenoise(self, batch_size, width, height, seed, multiplier, constant_batch_noise, normalize, sigmas=None, model=None, latent_channels=4, shape="BCHW"):

        generator = torch.manual_seed(seed)
        if shape == "BCHW":
            noise = torch.randn([batch_size, int(latent_channels), height // 8, width // 8], dtype=torch.float32, layout=torch.strided, generator=generator, device="cpu")
        elif shape == "BCTHW":
            noise = torch.randn([1, int(latent_channels), batch_size,height // 8, width // 8], dtype=torch.float32, layout=torch.strided, generator=generator, device="cpu")
        elif shape == "BTCHW":
            noise = torch.randn([1, batch_size, int(latent_channels), height // 8, width // 8], dtype=torch.float32, layout=torch.strided, generator=generator, device="cpu")
        if sigmas is not None:
            sigma = sigmas[0] - sigmas[-1]
            sigma /= model.model.latent_format.scale_factor
            noise *= sigma

        noise *=multiplier

        if normalize:
            noise = noise / noise.std()
        if constant_batch_noise:
            noise = noise[0].repeat(batch_size, 1, 1, 1)

        
        return ({"samples":noise}, )

def camera_embeddings(elevation, azimuth):
    elevation = torch.as_tensor([elevation])
    azimuth = torch.as_tensor([azimuth])
    embeddings = torch.stack(
        [
                torch.deg2rad(
                    (90 - elevation) - (90)
                ),  # Zero123 polar is 90-elevation
                torch.sin(torch.deg2rad(azimuth)),
                torch.cos(torch.deg2rad(azimuth)),
                torch.deg2rad(
                    90 - torch.full_like(elevation, 0)
                ),
        ], dim=-1).unsqueeze(1)

    return embeddings

def interpolate_angle(start, end, fraction):
    # Calculate the difference in angles and adjust for wraparound if necessary
    diff = (end - start + 540) % 360 - 180
    # Apply fraction to the difference
    interpolated = start + fraction * diff
    # Normalize the result to be within the range of -180 to 180
    return (interpolated + 180) % 360 - 180


class StableZero123_BatchSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
                              "azimuth_points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                              "elevation_points_string": ("STRING", {"default": "0:(0.0),\n7:(0.0),\n15:(0.0)\n", "multiline": True}),
                             }}
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "KJNodes/experimental"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, azimuth_points_string, elevation_points_string, interpolation):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)

        def ease_in(t):
            return t * t
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)
        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        
        # Parse the azimuth input string into a list of tuples
        azimuth_points = []
        azimuth_points_string = azimuth_points_string.rstrip(',\n')
        for point_str in azimuth_points_string.split(','):
            frame_str, azimuth_str = point_str.split(':')
            frame = int(frame_str.strip())
            azimuth = float(azimuth_str.strip()[1:-1]) 
            azimuth_points.append((frame, azimuth))
        # Sort the points by frame number
        azimuth_points.sort(key=lambda x: x[0])

        # Parse the elevation input string into a list of tuples
        elevation_points = []
        elevation_points_string = elevation_points_string.rstrip(',\n')
        for point_str in elevation_points_string.split(','):
            frame_str, elevation_str = point_str.split(':')
            frame = int(frame_str.strip())
            elevation_val = float(elevation_str.strip()[1:-1]) 
            elevation_points.append((frame, elevation_val))
        # Sort the points by frame number
        elevation_points.sort(key=lambda x: x[0])

        # Index of the next point to interpolate towards
        next_point = 1
        next_elevation_point = 1

        positive_cond_out = []
        positive_pooled_out = []
        negative_cond_out = []
        negative_pooled_out = []
        
        #azimuth interpolation
        for i in range(batch_size):
            # Find the interpolated azimuth for the current frame
            while next_point < len(azimuth_points) and i >= azimuth_points[next_point][0]:
                next_point += 1
            # If next_point is equal to the length of points, we've gone past the last point
            if next_point == len(azimuth_points):
                next_point -= 1  # Set next_point to the last index of points
            prev_point = max(next_point - 1, 0)  # Ensure prev_point is not less than 0

            # Calculate fraction
            if azimuth_points[next_point][0] != azimuth_points[prev_point][0]:  # Prevent division by zero
                fraction = (i - azimuth_points[prev_point][0]) / (azimuth_points[next_point][0] - azimuth_points[prev_point][0])
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                # Use the new interpolate_angle function
                interpolated_azimuth = interpolate_angle(azimuth_points[prev_point][1], azimuth_points[next_point][1], fraction)
            else:
                interpolated_azimuth = azimuth_points[prev_point][1]
            # Interpolate the elevation
            next_elevation_point = 1
            while next_elevation_point < len(elevation_points) and i >= elevation_points[next_elevation_point][0]:
                next_elevation_point += 1
            if next_elevation_point == len(elevation_points):
                next_elevation_point -= 1
            prev_elevation_point = max(next_elevation_point - 1, 0)

            if elevation_points[next_elevation_point][0] != elevation_points[prev_elevation_point][0]:
                fraction = (i - elevation_points[prev_elevation_point][0]) / (elevation_points[next_elevation_point][0] - elevation_points[prev_elevation_point][0])
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                interpolated_elevation = interpolate_angle(elevation_points[prev_elevation_point][1], elevation_points[next_elevation_point][1], fraction)
            else:
                interpolated_elevation = elevation_points[prev_elevation_point][1]

            cam_embeds = camera_embeddings(interpolated_elevation, interpolated_azimuth)
            cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

            positive_pooled_out.append(t)
            positive_cond_out.append(cond)
            negative_pooled_out.append(torch.zeros_like(t))
            negative_cond_out.append(torch.zeros_like(pooled))

        # Concatenate the conditions and pooled outputs
        final_positive_cond = torch.cat(positive_cond_out, dim=0)
        final_positive_pooled = torch.cat(positive_pooled_out, dim=0)
        final_negative_cond = torch.cat(negative_cond_out, dim=0)
        final_negative_pooled = torch.cat(negative_pooled_out, dim=0)

        # Structure the final output
        final_positive = [[final_positive_cond, {"concat_latent_image": final_positive_pooled}]]
        final_negative = [[final_negative_cond, {"concat_latent_image": final_negative_pooled}]]

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (final_positive, final_negative, {"samples": latent})

def linear_interpolate(start, end, fraction):
    return start + (end - start) * fraction

class SV3D_BatchSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 21, "min": 1, "max": 4096}),
                              "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
                              "azimuth_points_string": ("STRING", {"default": "0:(0.0),\n9:(180.0),\n20:(360.0)\n", "multiline": True}),
                              "elevation_points_string": ("STRING", {"default": "0:(0.0),\n9:(0.0),\n20:(0.0)\n", "multiline": True}),
                             }}
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
Allow scheduling of the azimuth and elevation conditions for SV3D.  
Note that SV3D is still a video model and the schedule needs to always go forward  
https://huggingface.co/stabilityai/sv3d
"""

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, azimuth_points_string, elevation_points_string, interpolation):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)

        def ease_in(t):
            return t * t
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)
        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        
        # Parse the azimuth input string into a list of tuples
        azimuth_points = []
        azimuth_points_string = azimuth_points_string.rstrip(',\n')
        for point_str in azimuth_points_string.split(','):
            frame_str, azimuth_str = point_str.split(':')
            frame = int(frame_str.strip())
            azimuth = float(azimuth_str.strip()[1:-1]) 
            azimuth_points.append((frame, azimuth))
        # Sort the points by frame number
        azimuth_points.sort(key=lambda x: x[0])

        # Parse the elevation input string into a list of tuples
        elevation_points = []
        elevation_points_string = elevation_points_string.rstrip(',\n')
        for point_str in elevation_points_string.split(','):
            frame_str, elevation_str = point_str.split(':')
            frame = int(frame_str.strip())
            elevation_val = float(elevation_str.strip()[1:-1]) 
            elevation_points.append((frame, elevation_val))
        # Sort the points by frame number
        elevation_points.sort(key=lambda x: x[0])

        # Index of the next point to interpolate towards
        next_point = 1
        next_elevation_point = 1
        elevations = []
        azimuths = []
        # For azimuth interpolation
        for i in range(batch_size):
            # Find the interpolated azimuth for the current frame
            while next_point < len(azimuth_points) and i >= azimuth_points[next_point][0]:
                next_point += 1
            if next_point == len(azimuth_points):
                next_point -= 1
            prev_point = max(next_point - 1, 0)

            if azimuth_points[next_point][0] != azimuth_points[prev_point][0]:
                fraction = (i - azimuth_points[prev_point][0]) / (azimuth_points[next_point][0] - azimuth_points[prev_point][0])
                # Apply the ease function to the fraction
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                interpolated_azimuth = linear_interpolate(azimuth_points[prev_point][1], azimuth_points[next_point][1], fraction)
            else:
                interpolated_azimuth = azimuth_points[prev_point][1]

            # Interpolate the elevation
            next_elevation_point = 1
            while next_elevation_point < len(elevation_points) and i >= elevation_points[next_elevation_point][0]:
                next_elevation_point += 1
            if next_elevation_point == len(elevation_points):
                next_elevation_point -= 1
            prev_elevation_point = max(next_elevation_point - 1, 0)

            if elevation_points[next_elevation_point][0] != elevation_points[prev_elevation_point][0]:
                fraction = (i - elevation_points[prev_elevation_point][0]) / (elevation_points[next_elevation_point][0] - elevation_points[prev_elevation_point][0])
                # Apply the ease function to the fraction
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                interpolated_elevation = linear_interpolate(elevation_points[prev_elevation_point][1], elevation_points[next_elevation_point][1], fraction)
            else:
                interpolated_elevation = elevation_points[prev_elevation_point][1]

            azimuths.append(interpolated_azimuth)
            elevations.append(interpolated_elevation)

        #print("azimuths", azimuths)
        #print("elevations", elevations)

        # Structure the final output
        final_positive = [[pooled, {"concat_latent_image": t, "elevation": elevations, "azimuth": azimuths}]]
        final_negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t),"elevation": elevations, "azimuth": azimuths}]]

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (final_positive, final_negative, {"samples": latent})

class LoadResAdapterNormalization:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "resadapter_path": (folder_paths.get_filename_list("checkpoints"), )
            } 
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_res_adapter"
    CATEGORY = "KJNodes/experimental"

    def load_res_adapter(self, model, resadapter_path):
        print("ResAdapter: Checking ResAdapter path")
        resadapter_full_path = folder_paths.get_full_path("checkpoints", resadapter_path)
        if not os.path.exists(resadapter_full_path):
            raise Exception("Invalid model path")
        else:
            print("ResAdapter: Loading ResAdapter normalization weights")
            from comfy.utils import load_torch_file
            prefix_to_remove = 'diffusion_model.'
            model_clone = model.clone()
            norm_state_dict = load_torch_file(resadapter_full_path)
            new_values = {key[len(prefix_to_remove):]: value for key, value in norm_state_dict.items() if key.startswith(prefix_to_remove)}
            print("ResAdapter: Attempting to add patches with ResAdapter weights")
            try:
                for key in model.model.diffusion_model.state_dict().keys():
                    if key in new_values:
                        original_tensor = model.model.diffusion_model.state_dict()[key]
                        new_tensor = new_values[key].to(model.model.diffusion_model.dtype)
                        if original_tensor.shape == new_tensor.shape:
                            model_clone.add_object_patch(f"diffusion_model.{key}.data", new_tensor)
                        else:
                            print("ResAdapter: No match for key: ",key)
            except:
                raise Exception("Could not patch model, this way of patching was added to ComfyUI on March 3rd 2024, is your ComfyUI up to date?")
            print("ResAdapter: Added resnet normalization patches")
            return (model_clone, )
        
class Superprompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instruction_prompt": ("STRING", {"default": 'Expand the following prompt to add more detail', "multiline": True}),
                "prompt": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 4096, "step": 1}),
            } 
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
# SuperPrompt
A T5 model fine-tuned on the SuperPrompt dataset for  
upsampling text prompts to more detailed descriptions.  
Meant to be used as a pre-generation step for text-to-image  
models that benefit from more detailed prompts.  
https://huggingface.co/roborovski/superprompt-v1
"""

    def process(self, instruction_prompt, prompt, max_new_tokens):
        device = model_management.get_torch_device()
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        checkpoint_path = os.path.join(script_directory, "models","superprompt-v1")
        if not os.path.exists(checkpoint_path):
                print(f"Downloading model to: {checkpoint_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="roborovski/superprompt-v1", 
                                  local_dir=checkpoint_path, 
                                  local_dir_use_symlinks=False)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False)

        model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, device_map=device)
        model.to(device)
        input_text = instruction_prompt + ": " + prompt
  
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids,  max_new_tokens=max_new_tokens)
        out = (tokenizer.decode(outputs[0]))
        out = out.replace('<pad>', '')
        out = out.replace('</s>', '')
        
        return (out, )


class CameraPoseVisualizer:
                
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pose_file_path": ("STRING", {"default": '', "multiline": False}),
            "base_xval": ("FLOAT", {"default": 0.2,"min": 0, "max": 100, "step": 0.01}),
            "zval": ("FLOAT", {"default": 0.3,"min": 0, "max": 100, "step": 0.01}),
            "scale": ("FLOAT", {"default": 1.0,"min": 0.01, "max": 10.0, "step": 0.01}),
            "use_exact_fx": ("BOOLEAN", {"default": False}),
            "relative_c2w": ("BOOLEAN", {"default": True}),
            "use_viewer": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cameractrl_poses": ("CAMERACTRL_POSES", {"default": None}),
            }
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
Visualizes the camera poses, from Animatediff-Evolved CameraCtrl Pose  
or a .txt file with RealEstate camera intrinsics and coordinates, in a 3D plot. 
"""
        
    def plot(self, pose_file_path, scale, base_xval, zval, use_exact_fx, relative_c2w, use_viewer, cameractrl_poses=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from torchvision.transforms import ToTensor

        x_min = -2.0 * scale
        x_max = 2.0 * scale
        y_min = -2.0 * scale
        y_max = 2.0 * scale
        z_min = -2.0 * scale
        z_max = 2.0 * scale
        plt.rcParams['text.color'] = '#999999'
        self.fig = plt.figure(figsize=(18, 7))
        self.fig.patch.set_facecolor('#353535')
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_facecolor('#353535') # Set the background color here
        self.ax.grid(color='#999999', linestyle='-', linewidth=0.5)
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        self.ax.set_xlabel('x', color='#999999')
        self.ax.set_ylabel('y', color='#999999')
        self.ax.set_zlabel('z', color='#999999')
        for text in self.ax.get_xticklabels() + self.ax.get_yticklabels() + self.ax.get_zticklabels():
            text.set_color('#999999')
        print('initialize camera pose visualizer')

        if pose_file_path != "":
            with open(pose_file_path, 'r') as f:
                poses = f.readlines()
                w2cs = [np.asarray([float(p) for p in pose.strip().split(' ')[7:]]).reshape(3, 4) for pose in poses[1:]]
                fxs = [float(pose.strip().split(' ')[1]) for pose in poses[1:]]
                #print(poses)
        elif cameractrl_poses is not None:
            poses = cameractrl_poses
            w2cs = [np.array(pose[7:]).reshape(3, 4) for pose in cameractrl_poses]
            fxs = [pose[1] for pose in cameractrl_poses]
        else:
            raise ValueError("Please provide either pose_file_path or cameractrl_poses")

        total_frames = len(w2cs)
        transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
        last_row = np.zeros((1, 4))
        last_row[0, -1] = 1.0

        w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
        c2ws = self.get_c2w(w2cs, transform_matrix, relative_c2w)

        for frame_idx, c2w in enumerate(c2ws):
            self.extrinsic2pyramid(c2w, frame_idx / total_frames, hw_ratio=1/1, base_xval=base_xval,
                                    zval=(fxs[frame_idx] if use_exact_fx else zval))

        # Create the colorbar
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=total_frames)
        colorbar = self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical')

        # Change the colorbar label
        colorbar.set_label('Frame', color='#999999') # Change the label and its color

        # Change the tick colors
        colorbar.ax.yaxis.set_tick_params(colors='#999999') # Change the tick color

        # Change the tick frequency
        # Assuming you want to set the ticks at every 10th frame
        ticks = np.arange(0, total_frames, 10)
        colorbar.ax.yaxis.set_ticks(ticks)
        
        plt.title('')
        plt.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        tensor_img = ToTensor()(img)
        buf.close()
        tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
        if use_viewer:
            time.sleep(1)
            plt.show()
        return (tensor_img,)

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=1/1, base_xval=1, zval=3):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        vertex_std = np.array([[0, 0, 0, 1],
                            [base_xval, -base_xval * hw_ratio, zval, 1],
                            [base_xval, base_xval * hw_ratio, zval, 1],
                            [-base_xval, base_xval * hw_ratio, zval, 1],
                            [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.25))

    def customize_legend(self, list_label):
        from matplotlib.patches import Patch
        import matplotlib.pyplot as plt
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def get_c2w(self, w2cs, transform_matrix, relative_c2w):
        if relative_c2w:
            target_cam_c2w = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            abs2rel = target_cam_c2w @ w2cs[0]
            ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
        else:
            ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
        ret_poses = [transform_matrix @ x for x in ret_poses]
        return np.array(ret_poses, dtype=np.float32)
    
    
            
class CheckpointPerturbWeights:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "joint_blocks": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 10.0, "step": 0.001}),
            "final_layer": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 10.0, "step": 0.001}),
            "rest_of_the_blocks": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 10.0, "step": 0.001}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mod"
    OUTPUT_NODE = True

    CATEGORY = "KJNodes/experimental"

    def mod(self, seed, model, joint_blocks, final_layer, rest_of_the_blocks):
        import copy
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = model_management.get_torch_device()
        model_copy = copy.deepcopy(model)
        model_copy.model.to(device)
        keys = model_copy.model.diffusion_model.state_dict().keys()

        dict = {}
        for key in keys:
            dict[key] = model_copy.model.diffusion_model.state_dict()[key]

        pbar = ProgressBar(len(keys))
        for k in keys:
            v = dict[k]
            print(f'{k}: {v.std()}') 
            if k.startswith('joint_blocks'):
                multiplier = joint_blocks
            elif k.startswith('final_layer'):
                multiplier = final_layer
            else:
                multiplier = rest_of_the_blocks
            dict[k] += torch.normal(torch.zeros_like(v) * v.mean(), torch.ones_like(v) * v.std() * multiplier).to(device)
            pbar.update(1)
        model_copy.model.diffusion_model.load_state_dict(dict)
        return model_copy,
    
class DifferentialDiffusionAdvanced():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "samples": ("LATENT",),
                    "mask": ("MASK",),
                    "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                            }}
    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model, samples, mask, multiplier):
        self.multiplier = multiplier
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return (model, s)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to) / self.multiplier

        return (denoise_mask >= threshold).to(denoise_mask.dtype)
    
class FluxBlockLoraSelect:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {}
        argument = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01})

        for i in range(19):
            arg_dict["double_blocks.{}.".format(i)] = argument

        for i in range(38):
            arg_dict["single_blocks.{}.".format(i)] = argument

        return {"required": arg_dict}
    
    RETURN_TYPES = ("SELECTEDDITBLOCKS", )
    RETURN_NAMES = ("blocks", )
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"

    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Select individual block alpha values, value of 0 removes the block altogether"

    def load_lora(self, **kwargs):
        return (kwargs,)
    
class HunyuanVideoBlockLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {}
        argument = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01})

        for i in range(20):
            arg_dict["double_blocks.{}.".format(i)] = argument

        for i in range(40):
            arg_dict["single_blocks.{}.".format(i)] = argument

        return {"required": arg_dict}
    
    RETURN_TYPES = ("SELECTEDDITBLOCKS", )
    RETURN_NAMES = ("blocks", )
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"

    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Select individual block alpha values, value of 0 removes the block altogether"

    def load_lora(self, **kwargs):
        return (kwargs,)

class Wan21BlockLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {}
        argument = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01})

        for i in range(40):
            arg_dict["blocks.{}.".format(i)] = argument

        return {"required": arg_dict}
    
    RETURN_TYPES = ("SELECTEDDITBLOCKS", )
    RETURN_NAMES = ("blocks", )
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"

    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Select individual block alpha values, value of 0 removes the block altogether"

    def load_lora(self, **kwargs):
        return (kwargs,)
    
class DiTBlockLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                
                },
                "optional": {
                    "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                    "opt_lora_path": ("STRING", {"forceInput": True, "tooltip": "Absolute path of the LoRA."}),
                    "blocks": ("SELECTEDDITBLOCKS",),
                }
               }
    
    RETURN_TYPES = ("MODEL", "STRING", )
    RETURN_NAMES = ("model", "rank", )
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "possible rank of the LoRA.")
    FUNCTION = "load_lora"
    CATEGORY = "KJNodes/experimental"

    def load_lora(self, model, strength_model, lora_name=None, opt_lora_path=None, blocks=None):
        
        import comfy.lora

        if opt_lora_path:
            lora_path = opt_lora_path
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
        
        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        # Find the first key that ends with "weight"
        rank = "unknown"
        weight_key = next((key for key in lora.keys() if key.endswith('weight')), None)
        # Print the shape of the value corresponding to the key
        if weight_key:
            print(f"Shape of the first 'weight' key ({weight_key}): {lora[weight_key].shape}")
            rank = str(lora[weight_key].shape[0])
        else:
            print("No key ending with 'weight' found.")
            rank = "Couldn't find rank"
        self.loaded_lora = (lora_path, lora)

        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

        loaded = comfy.lora.load_lora(lora, key_map)

        if blocks is not None:
            keys_to_delete = []

            for block in blocks:
                for key in list(loaded.keys()):
                    match = False
                    if isinstance(key, str) and block in key:
                        match = True
                    elif isinstance(key, tuple):
                        for k in key:
                            if block in k:
                                match = True
                                break

                    if match:
                        ratio = blocks[block]
                        if ratio == 0:
                            keys_to_delete.append(key)
                        else:
                            value = loaded[key].weights
                            weights_list = list(loaded[key].weights)
                            weights_list[2] = ratio
                            loaded[key].weights = tuple(weights_list)

            for key in keys_to_delete:
                del loaded[key]

            print("loading lora keys:")
            for key, value in loaded.items():
                print(f"Key: {key}, Alpha: {value.weights[2]}")


                if model is not None:
                    new_modelpatcher = model.clone()
                    k = new_modelpatcher.add_patches(loaded, strength_model)  
    
        k = set(k)
        for x in loaded:
            if (x not in k):
                print("NOT LOADED {}".format(x))

        return (new_modelpatcher, rank)
    
class CustomControlNetWeightsFluxFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list_of_floats": ("FLOAT", {"forceInput": True}, ),
            },
            "optional": {
                "uncond_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = ("CN_WEIGHTS", "TK_SHORTCUT")
    FUNCTION = "load_weights"
    DESCRIPTION = "Creates controlnet weights from a list of floats for Advanced-ControlNet"

    CATEGORY = "KJNodes/controlnet"

    def load_weights(self, list_of_floats: list[float],
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        
        adv_control = importlib.import_module("ComfyUI-Advanced-ControlNet.adv_control")
        ControlWeights = adv_control.utils.ControlWeights
        TimestepKeyframeGroup = adv_control.utils.TimestepKeyframeGroup
        TimestepKeyframe = adv_control.utils.TimestepKeyframe

        weights = ControlWeights.controlnet(weights_input=list_of_floats, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        print(weights.weights_input)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))
    
SHAKKERLABS_UNION_CONTROLNET_TYPES = {
    "canny": 0,
    "tile": 1,
    "depth": 2,
    "blur": 3,
    "pose": 4,
    "gray": 5,
    "low quality": 6,
}

class SetShakkerLabsUnionControlNetType:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"control_net": ("CONTROL_NET", ),
                             "type": (["auto"] + list(SHAKKERLABS_UNION_CONTROLNET_TYPES.keys()),)
                             }}

    CATEGORY = "conditioning/controlnet"
    RETURN_TYPES = ("CONTROL_NET",)

    FUNCTION = "set_controlnet_type"

    def set_controlnet_type(self, control_net, type):
        control_net = control_net.copy()
        type_number = SHAKKERLABS_UNION_CONTROLNET_TYPES.get(type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        return (control_net,)

class ModelSaveKJ:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "diffusion_models/ComfyUI"}),
                              "model_key_prefix": ("STRING", {"default": "model.diffusion_model."}),
                              },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "advanced/model_merging"

    def save(self, model, filename_prefix, model_key_prefix, prompt=None, extra_pnginfo=None):
        from comfy.utils import save_torch_file
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
    
        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        load_models = [model]

        model_management.load_models_gpu(load_models, force_patch_weights=True)
        default_prefix = "model.diffusion_model."

        sd = model.model.state_dict_for_saving(None, None, None)

        new_sd = {}
        for k in sd:
            if k.startswith(default_prefix):
                new_key = model_key_prefix + k[len(default_prefix):]
            else:
                new_key = k  # In case the key doesn't start with the default prefix, keep it unchanged
            t = sd[k]
            if not t.is_contiguous():
                t = t.contiguous()
            new_sd[new_key] = t
        print(full_output_folder)
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)
        save_torch_file(new_sd, os.path.join(full_output_folder, output_checkpoint))
        return {}
       
class StyleModelApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "StyleModelApply but with strength parameter"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, strength=1.0):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        cond = strength * cond
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )

class AudioConcatenate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio1": ("AUDIO",),
            "audio2": ("AUDIO",),
            "direction": (
            [   'right',
                'left',
            ],
            {
            "default": 'right'
             }),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concanate"
    CATEGORY = "KJNodes/audio"
    DESCRIPTION = """
Concatenates the audio1 to audio2 in the specified direction.
"""

    def concanate(self, audio1, audio2, direction):
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]
        if sample_rate_1 != sample_rate_2:
            raise Exception("Sample rates of the two audios do not match")
        
        waveform_1 = audio1["waveform"]
        print(waveform_1.shape)
        waveform_2 = audio2["waveform"]

        # Concatenate based on the specified direction
        if direction == 'right':
            concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)  # Concatenate along width
        elif direction == 'left':
            concatenated_audio= torch.cat((waveform_2, waveform_1), dim=2)  # Concatenate along width
        return ({"waveform": concatenated_audio, "sample_rate": sample_rate_1},)

class LeapfusionHunyuanI2V:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "index": ("INT", {"default": 0, "min": -1, "max": 1000, "step": 1,"tooltip": "The index of the latent to be replaced. 0 for first frame and -1 for last"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of steps to apply"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of steps to apply"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/experimental"

    def patch(self, model, latent, index, strength, start_percent, end_percent):

        def outer_wrapper(samples, index, start_percent, end_percent):
            def unet_wrapper(apply_model, args):
                steps = args["c"]["transformer_options"]["sample_sigmas"]
                inp, timestep, c = args["input"], args["timestep"], args["c"]
                matched_step_index = (steps == timestep).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    for i in range(len(steps) - 1):
                        # walk from beginning of steps until crossing the timestep
                        if (steps[i] - timestep[0]) * (steps[i + 1] - timestep[0]) <= 0:
                            current_step_index = i
                            break
                    else:
                        current_step_index = 0
                current_percent = current_step_index / (len(steps) - 1)
                if samples is not None:
                    if start_percent <= current_percent <= end_percent:
                        inp[:, :, [index], :, :] = samples[:, :, [0], :, :].to(inp)
                    else:
                        inp[:, :, [index], :, :] = torch.zeros(1)
                return apply_model(inp, timestep, **c)
            return unet_wrapper
        
        samples = latent["samples"] * 0.476986 * strength
        m = model.clone()
        m.set_model_unet_function_wrapper(outer_wrapper(samples, index, start_percent, end_percent))

        return (m,)

class ImageNoiseAugmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_aug_strength": ("FLOAT", {"default": None, "min": 0.0, "max": 100.0, "step": 0.001}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_noise"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
    Add noise to an image.  
    """

    def add_noise(self, image, noise_aug_strength, seed):
        torch.manual_seed(seed)
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * noise_aug_strength
        image_noise = torch.randn_like(image) * sigma[:, None, None, None]
        image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
        image_out = image + image_noise
        return image_out,

class VAELoaderKJ:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "vae_name": (s.vae_list(), ),
                          "device": (["main_device", "cpu"],),
                          "weight_dtype": (["bf16", "fp16", "fp32" ],),
                         }
            }
        
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "KJNodes/vae"

    def load_vae(self, vae_name, device, weight_dtype):
        from comfy.sd import VAE
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[weight_dtype]
        if device == "main_device":
            device = model_management.get_torch_device()
        elif device == "cpu":
            device = torch.device("cpu")
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = load_torch_file(vae_path)
        vae = VAE(sd=sd, device=device, dtype=dtype)
        return (vae,)

from comfy.samplers import sampling_function, CFGGuider
class Guider_ScheduledCFG(CFGGuider):

    def set_cfg(self, cfg, start_percent, end_percent):
        self.cfg = cfg
        self.start_percent = start_percent
        self.end_percent = end_percent

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        steps = model_options["transformer_options"]["sample_sigmas"]
        matched_step_index = (steps == timestep).nonzero()
        assert not (isinstance(self.cfg, list) and len(self.cfg) != (len(steps) - 1)), "cfg list length must match step count"
        if len(matched_step_index) > 0:
            current_step_index = matched_step_index.item()
        else:
            for i in range(len(steps) - 1):
                # walk from beginning of steps until crossing the timestep
                if (steps[i] - timestep[0]) * (steps[i + 1] - timestep[0]) <= 0:
                    current_step_index = i
                    break
            else:
                current_step_index = 0
        current_percent = current_step_index / (len(steps) - 1)

        if self.start_percent <= current_percent <= self.end_percent:
            if isinstance(self.cfg, list):
                cfg = self.cfg[current_step_index]
            else:
                cfg = self.cfg
            uncond = self.conds.get("negative", None)
        else:
            uncond = None
            cfg = 1.0

        return sampling_function(self.inner_model, x, timestep, uncond, self.conds.get("positive", None), cfg, model_options=model_options, seed=seed)            
  
class ScheduledCFGGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step":0.01}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01}),
                    },
                }
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "KJNodes/experimental"
    DESCRiPTION = """
CFG Guider that allows for scheduled CFG changes over steps, the steps outside the range will use CFG 1.0 thus being processed faster.  
cfg input can be a list of floats matching step count, or a single float for all steps.  
"""

    def get_guider(self, model, cfg, positive, negative, start_percent, end_percent):
        guider = Guider_ScheduledCFG(model) 
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg, start_percent, end_percent)
        return (guider, )
    

class ApplyRifleXRoPE_WanVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", {"tooltip": "Only used to get the latent count"}),
                "k": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1, "tooltip": "Index of intrinsic frequency"}),
            } 
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    EXPERIMENTAL = True
    DESCRIPTION = "Extends the potential frame count of HunyuanVideo using this method: https://github.com/thu-ml/RIFLEx"

    def patch(self, model, latent, k):
        model_class = model.model.diffusion_model
        
        model_clone = model.clone()
        num_frames = latent["samples"].shape[2]
        d = model_class.dim // model_class.num_heads

        rope_embedder = EmbedND_RifleX(
            d, 
            10000.0, 
            [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
            num_frames,
            k
            )
        
        model_clone.add_object_patch(f"diffusion_model.rope_embedder", rope_embedder)
                    
        return (model_clone, )
    
class ApplyRifleXRoPE_HunuyanVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", {"tooltip": "Only used to get the latent count"}),
                "k": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, "tooltip": "Index of intrinsic frequency"}),
            } 
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    EXPERIMENTAL = True
    DESCRIPTION = "Extends the potential frame count of HunyuanVideo using this method: https://github.com/thu-ml/RIFLEx"

    def patch(self, model, latent, k):
        model_class = model.model.diffusion_model
        
        model_clone = model.clone()
        num_frames = latent["samples"].shape[2]

        pe_embedder = EmbedND_RifleX(
            model_class.params.hidden_size // model_class.params.num_heads, 
            model_class.params.theta, 
            model_class.params.axes_dim, 
            num_frames,
            k
            )
        
        model_clone.add_object_patch(f"diffusion_model.pe_embedder", pe_embedder)
                    
        return (model_clone, )

def rope_riflex(pos, dim, theta, L_test, k):
    from einops import rearrange
    assert dim % 2 == 0
    if model_management.is_device_mps(pos.device) or model_management.is_intel_xpu() or model_management.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)

    # RIFLEX modification - adjust last frequency component if L_test and k are provided
    if k and L_test:
        omega[k-1] = 0.9 * 2 * torch.pi / L_test

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

class EmbedND_RifleX(nn.Module):
    def __init__(self, dim, theta, axes_dim, num_frames, k):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = num_frames
        self.k = k

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_riflex(ids[..., i], self.axes_dim[i], self.theta, self.num_frames, self.k if i == 0 else 0) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.elapsed = 0

class TimerNodeKJ:
    @classmethod
    
    def INPUT_TYPES(s):
      return {
        "required": {
            "any_input": (IO.ANY, ),
            "mode": (["start", "stop"],),
            "name": ("STRING", {"default": "Timer"}),
        },
        "optional": {
            "timer": ("TIMER",),
        },
	}

    RETURN_TYPES = (IO.ANY, "TIMER", "INT", )
    RETURN_NAMES = ("any_output", "timer", "time")
    FUNCTION = "timer"
    CATEGORY = "KJNodes/misc"

    def timer(self, mode, name, any_input=None, timer=None):
        if timer is None:
            if mode == "start":
                timer = Timer(name=name)            
                timer.start_time = time.time()
                return {"ui": {
                "text": [f"{timer.start_time}"]}, 
                "result": (any_input, timer, 0) 
                 }
        elif mode == "stop" and timer is not None:
            end_time = time.time()
            timer.elapsed = int((end_time - timer.start_time) * 1000)
            timer.start_time = None
            return (any_input, timer, timer.elapsed)

class HunyuanVideoEncodeKeyframesToCond:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "start_frame": ("IMAGE", ),
                    "end_frame": ("IMAGE", ),
                    "num_frames": ("INT", {"default": 33, "min": 2, "max": 4096, "step": 1}),
                    "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                    "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to encode at a time."}),
                    "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                    },
                    "optional": {
                        "negative": ("CONDITIONING", ),
                    }
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "KJNodes/videomodels"

    def encode(self, model, positive, start_frame, end_frame, num_frames, vae, tile_size, overlap, temporal_size, temporal_overlap, negative=None):

        model_clone = model.clone()

        model_clone.add_object_patch("concat_keys", ("concat_image",))

       
        x = (start_frame.shape[1] // 8) * 8
        y = (start_frame.shape[2] // 8) * 8

        if start_frame.shape[1] != x or start_frame.shape[2] != y:
            x_offset = (start_frame.shape[1] % 8) // 2
            y_offset = (start_frame.shape[2] % 8) // 2
            start_frame = start_frame[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
        if end_frame.shape[1] != x or end_frame.shape[2] != y:
            x_offset = (start_frame.shape[1] % 8) // 2
            y_offset = (start_frame.shape[2] % 8) // 2
            end_frame = end_frame[:,x_offset:x + x_offset, y_offset:y + y_offset,:]

        video_frames = torch.zeros(num_frames-2, start_frame.shape[1], start_frame.shape[2], start_frame.shape[3], device=start_frame.device, dtype=start_frame.dtype)
        video_frames = torch.cat([start_frame, video_frames, end_frame], dim=0)

        concat_latent = vae.encode_tiled(video_frames[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)

        out_latent = {}
        out_latent["samples"] = torch.zeros_like(concat_latent)

        out = []
        for conditioning in [positive, negative if negative is not None else []]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent
                n = [t[0], d]
                c.append(n)
            out.append(c)
        if len(out) == 1:
            out.append(out[0])
        return (model_clone, out[0], out[1], out_latent)      