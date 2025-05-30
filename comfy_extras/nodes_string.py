import re

from comfy.comfy_types.node_typing import IO

class StringConcatenate():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_a": (IO.STRING, {"multiline": True}),
                "string_b": (IO.STRING, {"multiline": True}),
                "delimiter": (IO.STRING, {"multiline": False, "default": ""})
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string_a, string_b, delimiter, **kwargs):
        return delimiter.join((string_a, string_b)),

class StringSubstring():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "start": (IO.INT, {}),
                "end": (IO.INT, {}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, start, end, **kwargs):
        return string[start:end],

class StringLength():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True})
            }
        }

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("length",)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, **kwargs):
        length = len(string)

        return length,

class CaseConverter():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "mode": (IO.COMBO, {"options": ["UPPERCASE", "lowercase", "Capitalize", "Title Case"]})
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, mode, **kwargs):
        if mode == "UPPERCASE":
            result = string.upper()
        elif mode == "lowercase":
            result = string.lower()
        elif mode == "Capitalize":
            result = string.capitalize()
        elif mode == "Title Case":
            result = string.title()
        else:
            result = string

        return result,


class StringTrim():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "mode": (IO.COMBO, {"options": ["Both", "Left", "Right"]})
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, mode, **kwargs):
        if mode == "Both":
            result = string.strip()
        elif mode == "Left":
            result = string.lstrip()
        elif mode == "Right":
            result = string.rstrip()
        else:
            result = string

        return result,

class StringReplace():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "find": (IO.STRING, {"multiline": True}),
                "replace": (IO.STRING, {"multiline": True})
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, find, replace, **kwargs):
        result = string.replace(find, replace)
        return result,


class StringContains():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "substring": (IO.STRING, {"multiline": True}),
                "case_sensitive": (IO.BOOLEAN, {"default": True})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    RETURN_NAMES = ("contains",)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, substring, case_sensitive, **kwargs):
        if case_sensitive:
            contains = substring in string
        else:
            contains = substring.lower() in string.lower()

        return contains,


class StringCompare():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_a": (IO.STRING, {"multiline": True}),
                "string_b": (IO.STRING, {"multiline": True}),
                "mode": (IO.COMBO, {"options": ["Starts With", "Ends With", "Equal"]}),
                "case_sensitive": (IO.BOOLEAN, {"default": True})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string_a, string_b, mode, case_sensitive, **kwargs):
        if case_sensitive:
            a = string_a
            b = string_b
        else:
            a = string_a.lower()
            b = string_b.lower()

        if mode == "Equal":
            return a == b,
        elif mode == "Starts With":
            return a.startswith(b),
        elif mode == "Ends With":
            return a.endswith(b),

class RegexMatch():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "regex_pattern": (IO.STRING, {"multiline": True}),
                "case_insensitive": (IO.BOOLEAN, {"default": True}),
                "multiline": (IO.BOOLEAN, {"default": False}),
                "dotall": (IO.BOOLEAN, {"default": False})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    RETURN_NAMES = ("matches",)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, regex_pattern, case_insensitive, multiline, dotall, **kwargs):
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        try:
            match = re.search(regex_pattern, string, flags)
            result = match is not None

        except re.error:
            result = False

        return result,


class RegexExtract():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "regex_pattern": (IO.STRING, {"multiline": True}),
                "mode": (IO.COMBO, {"options": ["First Match", "All Matches", "First Group", "All Groups"]}),
                "case_insensitive": (IO.BOOLEAN, {"default": True}),
                "multiline": (IO.BOOLEAN, {"default": False}),
                "dotall": (IO.BOOLEAN, {"default": False}),
                "group_index": (IO.INT, {"default": 1, "min": 0, "max": 100})
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, regex_pattern, mode, case_insensitive, multiline, dotall, group_index, **kwargs):
        join_delimiter = "\n"

        flags = 0
        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        try:
            if mode == "First Match":
                match = re.search(regex_pattern, string, flags)
                if match:
                    result = match.group(0)
                else:
                    result = ""

            elif mode == "All Matches":
                matches = re.findall(regex_pattern, string, flags)
                if matches:
                    if isinstance(matches[0], tuple):
                        result = join_delimiter.join([m[0] for m in matches])
                    else:
                        result = join_delimiter.join(matches)
                else:
                    result = ""

            elif mode == "First Group":
                match = re.search(regex_pattern, string, flags)
                if match and len(match.groups()) >= group_index:
                    result = match.group(group_index)
                else:
                    result = ""

            elif mode == "All Groups":
                matches = re.finditer(regex_pattern, string, flags)
                results = []
                for match in matches:
                    if match.groups() and len(match.groups()) >= group_index:
                        results.append(match.group(group_index))
                result = join_delimiter.join(results)
            else:
                result = ""

        except re.error:
            result = ""

        return result,


class RegexReplace():
    DESCRIPTION = "Find and replace text using regex patterns."
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": True}),
                "regex_pattern": (IO.STRING, {"multiline": True}),
                "replace": (IO.STRING, {"multiline": True}),
            },
            "optional": {
                "case_insensitive": (IO.BOOLEAN, {"default": True}),
                "multiline": (IO.BOOLEAN, {"default": False}),
                "dotall": (IO.BOOLEAN, {"default": False, "tooltip": "When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines."}),
                "count": (IO.INT, {"default": 0, "min": 0, "max": 100, "tooltip": "Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc."}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/string"

    def execute(self, string, regex_pattern, replace, case_insensitive=True, multiline=False, dotall=False, count=0, **kwargs):
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        result = re.sub(regex_pattern, replace, string, count=count, flags=flags)
        return result,

NODE_CLASS_MAPPINGS = {
    "StringConcatenate": StringConcatenate,
    "StringSubstring": StringSubstring,
    "StringLength": StringLength,
    "CaseConverter": CaseConverter,
    "StringTrim": StringTrim,
    "StringReplace": StringReplace,
    "StringContains": StringContains,
    "StringCompare": StringCompare,
    "RegexMatch": RegexMatch,
    "RegexExtract": RegexExtract,
    "RegexReplace": RegexReplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringConcatenate": "Concatenate",
    "StringSubstring": "Substring",
    "StringLength": "Length",
    "CaseConverter": "Case Converter",
    "StringTrim": "Trim",
    "StringReplace": "Replace",
    "StringContains": "Contains",
    "StringCompare": "Compare",
    "RegexMatch": "Regex Match",
    "RegexExtract": "Regex Extract",
    "RegexReplace": "Regex Replace",
}
