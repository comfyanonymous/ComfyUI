from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import importlib
import sys


@dataclass
class _Schema:
    node_id: str
    display_name: str | None = None
    category: str = "sd"
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    description: str = ""
    search_aliases: list[str] = field(default_factory=list)


class _FieldFactory:
    @staticmethod
    def Input(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    @staticmethod
    def Output(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def _import_nodes_string():
    fake_io = SimpleNamespace(
        Schema=_Schema,
        String=_FieldFactory,
        Int=_FieldFactory,
        Boolean=_FieldFactory,
        Combo=_FieldFactory,
        NodeOutput=lambda value: (value,),
        ComfyNode=object,
    )
    fake_latest = ModuleType("comfy_api.latest")
    fake_latest.ComfyExtension = object
    fake_latest.io = fake_io

    fake_comfy_api = ModuleType("comfy_api")
    fake_comfy_api.latest = fake_latest
    fake_typing_extensions = ModuleType("typing_extensions")
    fake_typing_extensions.override = lambda func: func

    with patch.dict(
        sys.modules,
        {
            "comfy_api": fake_comfy_api,
            "comfy_api.latest": fake_latest,
            "typing_extensions": fake_typing_extensions,
        },
    ):
        sys.modules.pop("comfy_extras.nodes_string", None)
        return importlib.import_module("comfy_extras.nodes_string")


nodes_string = _import_nodes_string()


class TestStringNodeSchemaRenames:
    def test_text_prefix_display_names_are_exposed(self):
        cases = [
            (nodes_string.StringConcatenate, "Text Concatenate"),
            (nodes_string.StringSubstring, "Text Substring"),
            (nodes_string.StringLength, "Text Length"),
            (nodes_string.CaseConverter, "Text Case Converter"),
            (nodes_string.StringTrim, "Text Trim"),
            (nodes_string.StringReplace, "Text Replace"),
            (nodes_string.StringContains, "Text Contains"),
            (nodes_string.StringCompare, "Text Compare"),
            (nodes_string.RegexMatch, "Text Match"),
            (nodes_string.RegexExtract, "Text Extract Substring"),
            (nodes_string.RegexReplace, "Text Replace (Regex)"),
        ]

        for node_cls, expected_name in cases:
            assert node_cls.define_schema().display_name == expected_name

    def test_old_display_names_remain_searchable(self):
        cases = [
            (nodes_string.StringConcatenate, "Concatenate"),
            (nodes_string.StringSubstring, "Substring"),
            (nodes_string.StringLength, "Length"),
            (nodes_string.CaseConverter, "Case Converter"),
            (nodes_string.StringTrim, "Trim"),
            (nodes_string.StringReplace, "Replace"),
            (nodes_string.StringContains, "Contains"),
            (nodes_string.StringCompare, "Compare"),
            (nodes_string.RegexMatch, "Regex Match"),
            (nodes_string.RegexExtract, "Regex Extract"),
            (nodes_string.RegexReplace, "Regex Replace"),
        ]

        for node_cls, expected_alias in cases:
            assert expected_alias in node_cls.define_schema().search_aliases

    def test_regex_nodes_keep_regex_search_keyword(self):
        regex_nodes = [
            nodes_string.RegexMatch,
            nodes_string.RegexExtract,
            nodes_string.RegexReplace,
        ]

        for node_cls in regex_nodes:
            assert "regex" in node_cls.define_schema().search_aliases
