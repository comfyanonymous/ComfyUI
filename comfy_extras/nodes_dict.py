import contextlib
import json
from typing import Optional

from comfy.comfy_types.node_typing import IO


class DictionaryNew:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_1": (IO.STRING, {"default": "", "multiline": False}),
                "value_1": (IO.STRING, {"default": "", "multiline": False}),
            },
            "optional": {
                "key_2": (IO.STRING, {"default": "", "multiline": False}),
                "value_2": (IO.STRING, {"default": "", "multiline": False}),
                "key_3": (IO.STRING, {"default": "", "multiline": False}),
                "value_3": (IO.STRING, {"default": "", "multiline": False}),
                "key_4": (IO.STRING, {"default": "", "multiline": False}),
                "value_4": (IO.STRING, {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (IO.DICT,)
    FUNCTION = "execute"
    CATEGORY = "utils/dict"

    @classmethod
    def execute(
        cls, key_1: str, value_1: str, key_2: str, value_2: str, key_3: str, value_3: str, key_4: str, value_4: str,
    ):
        return (
            {
                k: v
                for k, v in [
                    (key_1, value_1),
                    (key_2, value_2),
                    (key_3, value_3),
                    (key_4, value_4),
                ]
                if k
            },
        )


class DictionaryConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"dictionary_text": (IO.STRING, {"forceInput": True})},
            "optional": {"fallback_dict": (IO.DICT,)},
        }

    DESCRIPTION = "Parses a string into a dictionary"
    RETURN_TYPES = (IO.DICT,)
    FUNCTION = "execute"
    CATEGORY = "utils/dict"

    @classmethod
    def execute(cls, dictionary_text: str, fallback_dict: Optional[dict] = None):
        with contextlib.suppress(Exception):
            return json.loads(dictionary_text),
        return (fallback_dict,) if fallback_dict is not None else ({},)


class DictionaryGet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dictionary": (IO.DICT,),
                "key": (IO.STRING, {"default": "", "multiline": False}),
            },
            "optional": {
                "default_value": (IO.STRING, {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (IO.DICT,)
    FUNCTION = "execute"
    CATEGORY = "utils/dict"

    @classmethod
    def execute(cls, dictionary: dict, key: str, default_value=""):
        return (str(dictionary.get(key, default_value)),)


class DictionaryUpdate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_1": (IO.DICT,),
                "dict_2": (IO.DICT,),
            },
            "optional": {
                "dict_3": (IO.DICT,),
                "dict_4": (IO.DICT,),
            },
        }

    RETURN_TYPES = (IO.DICT,)
    FUNCTION = "execute"
    CATEGORY = "utils/dict"

    @classmethod
    def execute(cls, dict_1: dict, dict_2: dict, dict_3: Optional[dict] = None, dict_4: Optional[dict] = None):
        return ({**dict_1, **dict_2, **(dict_3 or {}), **(dict_4 or {})},)


NODE_CLASS_MAPPINGS = {
    "DictionaryNew": DictionaryNew,
    "DictionaryConvert": DictionaryConvert,
    "DictionaryGet": DictionaryGet,
    "DictionaryUpdate": DictionaryUpdate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DictionaryNew": "Dictionary New",
    "DictionaryConvert": "Convert to Dictionary",
    "DictionaryGet": "Dictionary Get",
    "DictionaryUpdate": "Dictionary Update",
}
