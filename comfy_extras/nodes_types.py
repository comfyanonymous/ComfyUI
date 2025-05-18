import re
from comfy.comfy_types.node_typing import IO

ROUND_MODES = ["truncate","round","bankers_rounding"]

def float_to_int(value: float, mode: str) -> int:
    if mode == "truncate":
        return int(value)
    elif mode == "bankers_rounding":
        # Python’s round implements “banker’s” (tie-to-even)
        return int(round(value))
    elif mode == "round":
        # half-away-from-zero
        return int(value + 0.5) if value >= 0 else int(value - 0.5)
    else:
        raise ValueError(f"Unknown mode: {mode}")

class IntToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0})}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value) -> float:
        return (float(value),)

class FloatToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "value": ("FLOAT", {"default": 0.0, "round": False}),
                    "mode": (ROUND_MODES,)
                    },
               }

    RETURN_TYPES = ("INT",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value, mode) -> int:
        return (float_to_int(value, mode),)

class IntToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value) -> str:
        return (str(value),)

class FloatToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "value": ("FLOAT", {"default": 0.0, "round": False}),
                    "round_value": ("BOOLEAN", {"default": True}),
                    "round_to": ("INT", {"default": 2,   "min": 0})
               }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value, round_value, round_to) -> str:
        if round_value:
            out = f"{value:.{round_to}f}"
        else:
            out = str(value)
        return (out,)

class StringToNum:

    # Regex that recognises *numeric literals Python will accept as floats*:
    #   • optional sign ([+-]?)
    #   • one of the three mantissa forms:
    #       – digits '.' optional-digits  (123. or 123.456)
    #       – '.' digits                 (.456)
    #       – digits                     (123)
    #   • optional exponent part with its own optional sign: ([eE][+-]?\d+)?
    #   • \Z anchors the match at the absolute end of the string so
    #     trailing whitespace or characters invalidate the match.
    _FLOAT_RE = re.compile(
        r"""
            [+-]?                  # optional sign
            (?:\d+\.\d*|\.\d+|\d+) # mantissa
            (?:[eE][+-]?\d+)?      # optional exponent
            \Z                     # end of string
        """,
        re.VERBOSE
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "value": ("STRING", {"default": "0", "multiline": False}),
                "int_mode": (ROUND_MODES,)
               }}

    RETURN_TYPES = ("INT","FLOAT",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value, int_mode):
        s = value.strip()
        if not self._FLOAT_RE.fullmatch(s):
            raise ValueError(f"StringToNum: cannot parse '{value}' as a number.")

        float_out = float(s)
        int_out = float_to_int(float_out, int_mode)

        return (int_out, float_out)

class BoolToAll:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("INT","FLOAT","STRING",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value):
        return (int(value), float(value), str(value),)

class IntToBool:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0})}}

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value) -> bool:
        return (bool(value),)

class FloatToBool:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"default": 0})}}

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value) -> bool:
        return (bool(value),)

class StringToBool:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "value": ("STRING", {"default": "False", "multiline": False}),
                "true_text": ("STRING", {"default": "True", "multiline": False}),
                "case_sensitive": ("BOOLEAN", {"default": True})
               }}

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "convert_type"
    CATEGORY = "utils/type_convert"

    def convert_type(self, value, true_text, case_sensitive) -> bool:
        if case_sensitive:
            match = (value == true_text)
        else:
            match = (value.casefold() == true_text.casefold())
        return (match,)

class StringToCombo:
    '''Converts a string into a combo input that may be used with any
       node with a list widget.
    '''
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": (IO.STRING, {"multiline": False})}}

    RETURN_TYPES  = (IO.ANY,)
    RETURN_NAMES  = ("COMBO",)
    FUNCTION      = "convert_type"
    CATEGORY      = "utils/type_convert"

    def convert_type(self, text: str):
        items = [t.strip() for t in text.split(",") if t.strip()]
        return (items[0] if items else "",)


NODE_CLASS_MAPPINGS = {
    "Boolean to All": BoolToAll,
    "Integer to Boolean": IntToBool,
    "Integer to Float": IntToFloat,
    "Integer to String": IntToString,
    "Float to Boolean": FloatToBool,
    "Float to Integer": FloatToInt,
    "Float to String": FloatToString,
    "String to Boolean": StringToBool,
    "String to Number": StringToNum,
    "String to Combo": StringToCombo
}
