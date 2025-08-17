import re

class StringFunction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "action": (["append", "replace"], {}),
                "tidy_tags": (["yes", "no"], {}),
            },
            "optional": {
                "text_a": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "text_b": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "text_c": ("STRING", {"multiline": True, "dynamicPrompts": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "exec"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def exec(self, action, tidy_tags, text_a="", text_b="", text_c=""):     
        tidy_tags = tidy_tags == "yes"
        out = ""
        if action == "append":
            out = (", " if tidy_tags else "").join(filter(None, [text_a, text_b, text_c]))
        else:
           if text_c is None:
               text_c = ""
           if text_b.startswith("/") and text_b.endswith("/"):
               regex = text_b[1:-1]
               out = re.sub(regex, text_c, text_a)
           else:
               out = text_a.replace(text_b, text_c)
        if tidy_tags:
            out = re.sub(r"\s{2,}", " ", out)
            out = out.replace(" ,", ",")
            out = re.sub(r",{2,}", ",", out)
            out = out.strip()
        return {"ui": {"text": (out,)}, "result": (out,)}
            
NODE_CLASS_MAPPINGS = {
    "StringFunction|pysssss": StringFunction,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringFunction|pysssss": "String Function 🐍",
}
