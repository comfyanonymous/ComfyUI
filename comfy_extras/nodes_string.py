"""
String manipulation nodes converted to ComfyUI v3 format.

This module contains v3 conversions of all string manipulation nodes from nodes_string.py.
The v3 implementations provide type safety, better documentation, and cleaner APIs
while maintaining full backward compatibility with v1 through the automatic
compatibility layer.
"""

import re
from comfy_api.v3 import io


class StringConcatenate(io.ComfyNodeV3):
    """Concatenates two strings with an optional delimiter between them."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StringConcatenate",
            display_name="String Concatenate",
            category="utils/string",
            description="Concatenates two strings together with an optional delimiter between them.",
            inputs=[
                io.String.Input(
                    "string_a",
                    display_name="String A",
                    multiline=True,
                    tooltip="The first string to concatenate",
                ),
                io.String.Input(
                    "string_b",
                    display_name="String B",
                    multiline=True,
                    tooltip="The second string to concatenate",
                ),
                io.String.Input(
                    "delimiter",
                    display_name="Delimiter",
                    default="",
                    multiline=False,
                    tooltip="The delimiter to insert between the two strings (empty by default)",
                ),
            ],
            outputs=[
                io.String.Output(
                    "concatenated",
                    display_name="Concatenated String",
                    tooltip="The result of concatenating string_a and string_b with the delimiter",
                ),
            ],
        )

    @classmethod
    def execute(cls, string_a: str, string_b: str, delimiter: str) -> io.NodeOutput:
        """Concatenates two strings with an optional delimiter."""
        result = delimiter.join((string_a, string_b))
        return io.NodeOutput(result)


class StringSubstring(io.ComfyNodeV3):
    """Extracts a substring from a string using start and end indices."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StringSubstring",
            display_name="String Substring",
            category="utils/string",
            description="Extracts a portion of a string using Python slice notation [start:end].",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to extract a substring from",
                ),
                io.Int.Input(
                    "start",
                    display_name="Start Index",
                    tooltip="Starting position (inclusive). Negative values count from the end",
                ),
                io.Int.Input(
                    "end",
                    display_name="End Index",
                    tooltip="Ending position (exclusive). Negative values count from the end",
                ),
            ],
            outputs=[
                io.String.Output(
                    "substring",
                    display_name="Substring",
                    tooltip="The extracted substring",
                ),
            ],
        )

    @classmethod
    def execute(cls, string: str, start: int, end: int) -> io.NodeOutput:
        """Extracts substring using Python slice notation."""
        return io.NodeOutput(string[start:end])


class StringLength(io.ComfyNodeV3):
    """Returns the length of a string."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StringLength",
            display_name="String Length",
            category="utils/string",
            description="Calculates the number of characters in a string.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to measure",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "length",
                    display_name="Length",
                    tooltip="The number of characters in the string",
                ),
            ],
        )

    @classmethod
    def execute(cls, string: str) -> io.NodeOutput:
        """Returns the length of the input string."""
        return io.NodeOutput(len(string))


class CaseConverter(io.ComfyNodeV3):
    """Converts string case to uppercase, lowercase, capitalize, or title case."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="CaseConverter",
            display_name="Case Converter",
            category="utils/string",
            description="Converts text to different case formats.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to convert",
                ),
                io.Combo.Input(
                    "mode",
                    display_name="Mode",
                    options=["UPPERCASE", "lowercase", "Capitalize", "Title Case"],
                    tooltip="The case conversion mode to apply",
                ),
            ],
            outputs=[
                io.String.Output(
                    "converted",
                    display_name="Converted String",
                    tooltip="The string with the selected case conversion applied",
                ),
            ],
        )

    @classmethod
    def execute(cls, string: str, mode: str) -> io.NodeOutput:
        """Converts string to the selected case format."""
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

        return io.NodeOutput(result)


class StringTrim(io.ComfyNodeV3):
    """Removes whitespace from the beginning, end, or both sides of a string."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StringTrim",
            display_name="String Trim",
            category="utils/string",
            description="Removes leading and/or trailing whitespace from a string.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to trim",
                ),
                io.Combo.Input(
                    "mode",
                    display_name="Mode",
                    options=["Both", "Left", "Right"],
                    tooltip="Which side(s) to trim whitespace from",
                ),
            ],
            outputs=[
                io.String.Output(
                    "trimmed",
                    display_name="Trimmed String",
                    tooltip="The string with whitespace removed",
                ),
            ],
        )

    @classmethod
    def execute(cls, string: str, mode: str) -> io.NodeOutput:
        """Removes whitespace based on the selected mode."""
        if mode == "Both":
            result = string.strip()
        elif mode == "Left":
            result = string.lstrip()
        elif mode == "Right":
            result = string.rstrip()
        else:
            result = string

        return io.NodeOutput(result)


class StringReplace(io.ComfyNodeV3):
    """Replaces all occurrences of a substring with another string."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StringReplace",
            display_name="String Replace",
            category="utils/string",
            description="Replaces all occurrences of a substring within a string.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to search in",
                ),
                io.String.Input(
                    "find",
                    display_name="Find",
                    multiline=True,
                    tooltip="The substring to search for",
                ),
                io.String.Input(
                    "replace",
                    display_name="Replace",
                    multiline=True,
                    tooltip="The string to replace matches with",
                ),
            ],
            outputs=[
                io.String.Output(
                    "result",
                    display_name="Result",
                    tooltip="The string with all replacements made",
                ),
            ],
        )

    @classmethod
    def execute(cls, string: str, find: str, replace: str) -> io.NodeOutput:
        """Replaces all occurrences of find with replace."""
        result = string.replace(find, replace)
        return io.NodeOutput(result)


class StringContains(io.ComfyNodeV3):
    """Checks if a string contains a substring."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StringContains",
            display_name="String Contains",
            category="utils/string",
            description="Checks whether a string contains a specific substring.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to search in",
                ),
                io.String.Input(
                    "substring",
                    display_name="Substring",
                    multiline=True,
                    tooltip="The substring to search for",
                ),
                io.Boolean.Input(
                    "case_sensitive",
                    display_name="Case Sensitive",
                    default=True,
                    tooltip="Whether the search should be case sensitive",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    "contains",
                    display_name="Contains",
                    tooltip="True if the substring is found, False otherwise",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, string: str, substring: str, case_sensitive: bool
    ) -> io.NodeOutput:
        """Checks if string contains substring with optional case sensitivity."""
        if case_sensitive:
            contains = substring in string
        else:
            contains = substring.lower() in string.lower()

        return io.NodeOutput(contains)


class StringCompare(io.ComfyNodeV3):
    """Compares two strings with various comparison modes."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StringCompare",
            display_name="String Compare",
            category="utils/string",
            description="Compares two strings using different comparison modes.",
            inputs=[
                io.String.Input(
                    "string_a",
                    display_name="String A",
                    multiline=True,
                    tooltip="The first string to compare",
                ),
                io.String.Input(
                    "string_b",
                    display_name="String B",
                    multiline=True,
                    tooltip="The second string to compare",
                ),
                io.Combo.Input(
                    "mode",
                    display_name="Mode",
                    options=["Starts With", "Ends With", "Equal"],
                    tooltip="The comparison mode to use",
                ),
                io.Boolean.Input(
                    "case_sensitive",
                    display_name="Case Sensitive",
                    default=True,
                    tooltip="Whether the comparison should be case sensitive",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    "result",
                    display_name="Result",
                    tooltip="True if the comparison succeeds, False otherwise",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, string_a: str, string_b: str, mode: str, case_sensitive: bool
    ) -> io.NodeOutput:
        """Compares two strings based on the selected mode and case sensitivity."""
        if case_sensitive:
            a = string_a
            b = string_b
        else:
            a = string_a.lower()
            b = string_b.lower()

        if mode == "Equal":
            result = a == b
        elif mode == "Starts With":
            result = a.startswith(b)
        elif mode == "Ends With":
            result = a.endswith(b)
        else:
            result = False

        return io.NodeOutput(result)


class RegexMatch(io.ComfyNodeV3):
    """Tests if a string matches a regular expression pattern."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="RegexMatch",
            display_name="Regex Match",
            category="utils/string",
            description="Tests whether a string matches a regular expression pattern.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to test",
                ),
                io.String.Input(
                    "regex_pattern",
                    display_name="Regex Pattern",
                    multiline=True,
                    tooltip="The regular expression pattern to match against",
                ),
                io.Boolean.Input(
                    "case_insensitive",
                    display_name="Case Insensitive",
                    default=True,
                    tooltip="Whether to ignore case when matching",
                ),
                io.Boolean.Input(
                    "multiline",
                    display_name="Multiline",
                    default=False,
                    tooltip="Whether ^ and $ match line boundaries",
                ),
                io.Boolean.Input(
                    "dotall",
                    display_name="Dot All",
                    default=False,
                    tooltip="Whether . matches newline characters",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    "matches",
                    display_name="Matches",
                    tooltip="True if the pattern matches, False otherwise",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        string: str,
        regex_pattern: str,
        case_insensitive: bool,
        multiline: bool,
        dotall: bool,
    ) -> io.NodeOutput:
        """Tests if string matches the regex pattern."""
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

        return io.NodeOutput(result)


class RegexExtract(io.ComfyNodeV3):
    """Extracts text from a string using regular expression patterns."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="RegexExtract",
            display_name="Regex Extract",
            category="utils/string",
            description="Extracts text from a string using regular expression patterns and groups.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to extract from",
                ),
                io.String.Input(
                    "regex_pattern",
                    display_name="Regex Pattern",
                    multiline=True,
                    tooltip="The regular expression pattern with optional groups",
                ),
                io.Combo.Input(
                    "mode",
                    display_name="Mode",
                    options=["First Match", "All Matches", "First Group", "All Groups"],
                    tooltip="What to extract from the matches",
                ),
                io.Boolean.Input(
                    "case_insensitive",
                    display_name="Case Insensitive",
                    default=True,
                    tooltip="Whether to ignore case when matching",
                ),
                io.Boolean.Input(
                    "multiline",
                    display_name="Multiline",
                    default=False,
                    tooltip="Whether ^ and $ match line boundaries",
                ),
                io.Boolean.Input(
                    "dotall",
                    display_name="Dot All",
                    default=False,
                    tooltip="Whether . matches newline characters",
                ),
                io.Int.Input(
                    "group_index",
                    display_name="Group Index",
                    default=1,
                    min=0,
                    max=100,
                    tooltip="Which capture group to extract (0 = entire match)",
                ),
            ],
            outputs=[
                io.String.Output(
                    "extracted",
                    display_name="Extracted",
                    tooltip="The extracted text (multiple matches joined with newlines)",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        string: str,
        regex_pattern: str,
        mode: str,
        case_insensitive: bool,
        multiline: bool,
        dotall: bool,
        group_index: int,
    ) -> io.NodeOutput:
        """Extracts text based on regex pattern and mode."""
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

        return io.NodeOutput(result)


class RegexReplace(io.ComfyNodeV3):
    """Find and replace text using regex patterns."""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="RegexReplace",
            display_name="Regex Replace",
            category="utils/string",
            description="Find and replace text using regular expression patterns.",
            inputs=[
                io.String.Input(
                    "string",
                    display_name="String",
                    multiline=True,
                    tooltip="The string to perform replacements on",
                ),
                io.String.Input(
                    "regex_pattern",
                    display_name="Regex Pattern",
                    multiline=True,
                    tooltip="The regular expression pattern to match",
                ),
                io.String.Input(
                    "replace",
                    display_name="Replace",
                    multiline=True,
                    tooltip="The replacement text (can use \\1, \\2 for capture groups)",
                ),
                io.Boolean.Input(
                    "case_insensitive",
                    display_name="Case Insensitive",
                    default=True,
                    optional=True,
                    tooltip="Whether to ignore case when matching",
                ),
                io.Boolean.Input(
                    "multiline",
                    display_name="Multiline",
                    default=False,
                    optional=True,
                    tooltip="Whether ^ and $ match line boundaries",
                ),
                io.Boolean.Input(
                    "dotall",
                    display_name="Dot All",
                    default=False,
                    optional=True,
                    tooltip="When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines.",
                ),
                io.Int.Input(
                    "count",
                    display_name="Count",
                    default=0,
                    min=0,
                    max=100,
                    optional=True,
                    tooltip="Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "result",
                    display_name="Result",
                    tooltip="The string with replacements made",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        string: str,
        regex_pattern: str,
        replace: str,
        case_insensitive: bool = True,
        multiline: bool = False,
        dotall: bool = False,
        count: int = 0,
    ) -> io.NodeOutput:
        """Replaces text matching regex pattern."""
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        result = re.sub(regex_pattern, replace, string, count=count, flags=flags)
        return io.NodeOutput(result)


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

