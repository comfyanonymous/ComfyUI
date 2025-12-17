import re
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class StringConcatenate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringConcatenate",
            display_name="Concatenate",
            category="utils/string",
            inputs=[
                io.String.Input("string_a", multiline=True),
                io.String.Input("string_b", multiline=True),
                io.String.Input("delimiter", multiline=False, default=""),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string_a, string_b, delimiter):
        return io.NodeOutput(delimiter.join((string_a, string_b)))


class StringSubstring(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringSubstring",
            display_name="Substring",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
                io.Int.Input("start"),
                io.Int.Input("end"),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, start, end):
        return io.NodeOutput(string[start:end])


class StringLength(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringLength",
            display_name="Length",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
            ],
            outputs=[
                io.Int.Output(display_name="length"),
            ]
        )

    @classmethod
    def execute(cls, string):
        return io.NodeOutput(len(string))


class CaseConverter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CaseConverter",
            display_name="Case Converter",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
                io.Combo.Input("mode", options=["UPPERCASE", "lowercase", "Capitalize", "Title Case"]),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, mode):
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


class StringTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringTrim",
            display_name="Trim",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
                io.Combo.Input("mode", options=["Both", "Left", "Right"]),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, mode):
        if mode == "Both":
            result = string.strip()
        elif mode == "Left":
            result = string.lstrip()
        elif mode == "Right":
            result = string.rstrip()
        else:
            result = string

        return io.NodeOutput(result)


class StringReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringReplace",
            display_name="Replace",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("find", multiline=True),
                io.String.Input("replace", multiline=True),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, find, replace):
        return io.NodeOutput(string.replace(find, replace))


class StringContains(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringContains",
            display_name="Contains",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("substring", multiline=True),
                io.Boolean.Input("case_sensitive", default=True),
            ],
            outputs=[
                io.Boolean.Output(display_name="contains"),
            ]
        )

    @classmethod
    def execute(cls, string, substring, case_sensitive):
        if case_sensitive:
            contains = substring in string
        else:
            contains = substring.lower() in string.lower()

        return io.NodeOutput(contains)


class StringCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringCompare",
            display_name="Compare",
            category="utils/string",
            inputs=[
                io.String.Input("string_a", multiline=True),
                io.String.Input("string_b", multiline=True),
                io.Combo.Input("mode", options=["Starts With", "Ends With", "Equal"]),
                io.Boolean.Input("case_sensitive", default=True),
            ],
            outputs=[
                io.Boolean.Output(),
            ]
        )

    @classmethod
    def execute(cls, string_a, string_b, mode, case_sensitive):
        if case_sensitive:
            a = string_a
            b = string_b
        else:
            a = string_a.lower()
            b = string_b.lower()

        if mode == "Equal":
            return io.NodeOutput(a == b)
        elif mode == "Starts With":
            return io.NodeOutput(a.startswith(b))
        elif mode == "Ends With":
            return io.NodeOutput(a.endswith(b))


class RegexMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexMatch",
            display_name="Regex Match",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("regex_pattern", multiline=True),
                io.Boolean.Input("case_insensitive", default=True),
                io.Boolean.Input("multiline", default=False),
                io.Boolean.Input("dotall", default=False),
            ],
            outputs=[
                io.Boolean.Output(display_name="matches"),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, case_insensitive, multiline, dotall):
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


class RegexExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexExtract",
            display_name="Regex Extract",
            category="utils/string",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("regex_pattern", multiline=True),
                io.Combo.Input("mode", options=["First Match", "All Matches", "First Group", "All Groups"]),
                io.Boolean.Input("case_insensitive", default=True),
                io.Boolean.Input("multiline", default=False),
                io.Boolean.Input("dotall", default=False),
                io.Int.Input("group_index", default=1, min=0, max=100),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, mode, case_insensitive, multiline, dotall, group_index):
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


class RegexReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexReplace",
            display_name="Regex Replace",
            category="utils/string",
            description="Find and replace text using regex patterns.",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("regex_pattern", multiline=True),
                io.String.Input("replace", multiline=True),
                io.Boolean.Input("case_insensitive", default=True, optional=True),
                io.Boolean.Input("multiline", default=False, optional=True),
                io.Boolean.Input("dotall", default=False, optional=True, tooltip="When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines."),
                io.Int.Input("count", default=0, min=0, max=100, optional=True, tooltip="Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, replace, case_insensitive=True, multiline=False, dotall=False, count=0):
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        result = re.sub(regex_pattern, replace, string, count=count, flags=flags)
        return io.NodeOutput(result)


class StringExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            StringConcatenate,
            StringSubstring,
            StringLength,
            CaseConverter,
            StringTrim,
            StringReplace,
            StringContains,
            StringCompare,
            RegexMatch,
            RegexExtract,
            RegexReplace,
        ]

async def comfy_entrypoint() -> StringExtension:
    return StringExtension()
