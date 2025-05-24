import re
from comfy.comfy_types.node_typing import IO

# Portions of this code is derived from Itdrdata/ComfyUI-Impact-Pack,
# specifically the implementation of ConvertDataType.

class ConvertDataType:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*"),
                             "int_mode": (["truncate","round","bankers_rounding"],),
                             "string_round_value": ("BOOLEAN", {"default": True}),
                             "string_round_to": ("INT", {"default": 2,   "min": 0})
                             }}

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN", IO.ANY,)
    RETURN_NAMES = ("STRING", "FLOAT", "INT", "BOOLEAN", "COMBO",)
    FUNCTION = "convert_type"

    CATEGORY = "utils/primitive"

    @staticmethod
    def is_number(string):
        pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+$')
        return bool(pattern.match(string))

    def convert_type(self, value, int_mode, string_round_value, string_round_to):
        # Initialize all outputs, necessary in the case of converting a string,
        # and the string is not a number.
        float_out  = None
        int_out    = None
        string_out = None

        if isinstance(value, str):
            if self.is_number(value):
                float_out = float(value)
            else:
                string_out = value
        elif isinstance(value, (int, float)):
            float_out = float(value)
        else:
            string_out = str(value)

        if float_out is not None:
            v = float_out
            if int_mode == "truncate":
                int_out = int(v)
            elif int_mode == "bankers_rounding":
                int_out = int(round(v))
            elif int_mode == "round":
                int_out = int(v + 0.5) if v >= 0 else int(v - 0.5)
            else:
                raise ValueError(f"Unknown mode: {int_mode}")

            if string_round_value:
                string_out = f"{v:.{string_round_to}f}"
            else:
                string_out = str(v)

        if float_out is not None:
            boolean_out = bool(float_out)
        elif string_out is not None:
            boolean_out = bool(string_out)
        else:
            boolean_out = False

        items   = (
            [t.strip() for t in string_out.split(",") if t.strip()]
            if isinstance(string_out, str) else
            []
        )
        combo_out = items[0] if items else ""

        return (string_out, float_out, int_out, boolean_out, combo_out)

NODE_CLASS_MAPPINGS = {
    "Convert Data Type": ConvertDataType
}
