import re

from comfy.graph_utils import GraphBuilder

class InversionDemoAdvancedPromptNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING")
    FUNCTION = "advanced_prompt"

    CATEGORY = "InversionDemo Nodes/Demo"

    def parse_timesteps(self, text):
        text = re.sub(r'<[^<]*>', lambda m: m.group(0).replace(':', '||COLON||'), text)
        text = re.sub(r':\d+\.\d+\)', lambda m: m.group(0).replace(':', '||COLON||'), text)
        def recurse(text, min_value, max_value):
            # First, replace any colons in angle brackets with a placeholder
            pattern = r'\[([^:\[\]]*):([^:\[\]]*):(\d+\.\d+)\]'
            m = re.search(pattern, text)
            if m is None:
                return [{
                    "text": re.sub(r'\|\|COLON\|\|', ':', text),
                    "min": min_value,
                    "max": max_value,
                }]

            # If we have a match, check if the value is in range
            start, end = m.span()
            value = float(m.group(3))
            before = text[:start] + m.group(1) + text[end:]
            after = text[:start] + m.group(2) + text[end:]
            if value <= min_value:
                return recurse(after, min_value, max_value)
            elif value >= max_value:
                return recurse(before, min_value, max_value)
            else:
                return recurse(before, min_value, value) + recurse(after, value, max_value)
        return recurse(text, 0, 1)

    def parse_loras(self, prompt):
        # Get all string pieces matching the pattern "<lora:(name):(strength)(:(clip_strength))?>"
        # where name is a string and strength is a float
        # and clip_strength is an optional float
        pattern = r"<lora:([^:]+):([-0-9.]+)(?::([-0-9.]+))?>"
        loras = re.findall(pattern, prompt)
        if len(loras) == 0:
            return prompt, loras
        cleaned_prompt = re.sub(pattern, "", prompt).strip()
        return cleaned_prompt, loras


    def advanced_prompt(self, prompt, clip, model):
        graph = GraphBuilder()
        cleaned_prompt, loras = self.parse_loras(prompt)
        for lora in loras:
            lora_name = lora[0]
            lora_model_strength = float(lora[1])
            lora_clip_strength = lora_model_strength if lora[2] == "" else float(lora[2])

            loader = graph.node("LoraLoader", model=model, clip=clip, lora_name = lora_name, strength_model = lora_model_strength, strength_clip = lora_clip_strength)
            model = loader.out(0)
            clip = loader.out(1)

        timesteps = self.parse_timesteps(cleaned_prompt)
        prev_output = None
        for timestep in timesteps:
            encoder = graph.node("CLIPTextEncode", clip=clip, text=timestep["text"])
            ranger = graph.node("ConditioningSetTimestepRange", conditioning=encoder.out(0), start=timestep["min"], end=timestep["max"])
            if prev_output is None:
                prev_output = ranger.out(0)
            else:
                prev_output = graph.node("ConditioningCombine", conditioning_1=prev_output, conditioning_2=ranger.out(0)).out(0)

        return {
            "result": (model, clip, prev_output),
            "expand": graph.finalize(),
        }

class InversionDemoFakeAdvancedPromptNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING")
    FUNCTION = "advanced_prompt"

    CATEGORY = "InversionDemo Nodes/Debug"

    def advanced_prompt(self, prompt, clip, model):
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return (model, clip, [[cond, {"pooled_output": pooled}]])

class InversionDemoLazySwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN",),
                "on_false": ("*", {"lazy": True}),
                "on_true": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "switch"

    CATEGORY = "InversionDemo Nodes/Logic"

    def check_lazy_status(self, switch, on_false = None, on_true = None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    def switch(self, switch, on_false = None, on_true = None):
        value = on_true if switch else on_false
        return (value,)

NUM_IF_ELSE_NODES = 10
class InversionDemoLazyConditional:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        args = {
            "value1": ("*", {"lazy": True}),
            "condition1": ("BOOLEAN", {"forceInput": True}),
        }

        for i in range(1,NUM_IF_ELSE_NODES):
            args["value%d" % (i + 1)] = ("*", {"lazy": True})
            args["condition%d" % (i + 1)] = ("BOOLEAN", {"lazy": True, "forceInput": True})

        args["else"] = ("*", {"lazy": True})

        return {
            "required": {},
            "optional": args,
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "conditional"

    CATEGORY = "InversionDemo Nodes/Logic"

    def check_lazy_status(self, **kwargs):
        for i in range(0,NUM_IF_ELSE_NODES):
            cond = "condition%d" % (i + 1)
            if cond not in kwargs:
                return [cond]
            if kwargs[cond]:
                val = "value%d" % (i + 1)
                if val not in kwargs:
                    return [val]
                else:
                    return []

        if "else" not in kwargs:
            return ["else"]

    def conditional(self, **kwargs):
        for i in range(0,NUM_IF_ELSE_NODES):
            cond = "condition%d" % (i + 1)
            if cond not in kwargs:
                return [cond]
            if kwargs.get(cond, False):
                val = "value%d" % (i + 1)
                return (kwargs.get(val, None),)

        return (kwargs.get("else", None),)
    
    
class InversionDemoLazyIndexSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
                "value0": ("*", {"lazy": True}),
            },
            "optional": {
                "value1": ("*", {"lazy": True}),
                "value2": ("*", {"lazy": True}),
                "value3": ("*", {"lazy": True}),
                "value4": ("*", {"lazy": True}),
                "value5": ("*", {"lazy": True}),
                "value6": ("*", {"lazy": True}),
                "value7": ("*", {"lazy": True}),
                "value8": ("*", {"lazy": True}),
                "value9": ("*", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "index_switch"

    CATEGORY = "InversionDemo Nodes/Logic"

    def check_lazy_status(self, index, **kwargs):
        key = "value%d" % index
        if key not in kwargs:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "value%d" % index
        return (kwargs[key],)

class InversionDemoLazyMixImages:
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

    CATEGORY = "InversionDemo Nodes/Demo"

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

        return (image1 * (1. - mask) + image2 * mask,)

GENERAL_NODE_CLASS_MAPPINGS = {
    "InversionDemoAdvancedPromptNode": InversionDemoAdvancedPromptNode,
    "InversionDemoFakeAdvancedPromptNode": InversionDemoFakeAdvancedPromptNode,
    "InversionDemoLazySwitch": InversionDemoLazySwitch,
    "InversionDemoLazyIndexSwitch": InversionDemoLazyIndexSwitch,
    "InversionDemoLazyMixImages": InversionDemoLazyMixImages,
    "InversionDemoLazyConditional": InversionDemoLazyConditional,
}

GENERAL_NODE_DISPLAY_NAME_MAPPINGS = {
    "InversionDemoAdvancedPromptNode": "Advanced Prompt",
    "InversionDemoFakeAdvancedPromptNode": "Fake Advanced Prompt",
    "InversionDemoLazySwitch": "Lazy Switch",
    "InversionDemoLazyIndexSwitch": "Lazy Index Switch",
    "InversionDemoLazyMixImages": "Lazy Mix Images",
    "InversionDemoLazyConditional": "Lazy Conditional",
}
