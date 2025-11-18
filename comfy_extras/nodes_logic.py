from typing import TypedDict
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io



class SwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySwitchNode",
            display_name="Switch",
            category="logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_false", template=template, lazy=True),
                io.MatchType.Input("on_true", template=template, lazy=True),
            ],
            outputs=[
                io.MatchType.Output("output", template=template, display_name="output"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, switch, on_false=None, on_true=None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    @classmethod
    def execute(cls, switch, on_true, on_false) -> io.NodeOutput:
        return io.NodeOutput(on_true if switch else on_false)


class DCTestNode(io.ComfyNode):
    class DCValues(TypedDict):
        combo: str
        string: str
        integer: int
        image: io.Image.Type

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DCTestNode",
            display_name="DCTest",
            category="logic",
            is_output_node=True,
            inputs=[io.DynamicCombo.Input("combo", options=[
                io.DynamicCombo.Option("option1", [io.String.Input("string")]),
                io.DynamicCombo.Option("option2", [io.Int.Input("integer")]),
                io.DynamicCombo.Option("option3", [io.Image.Input("image")]),
                ]
            )],
            outputs=[io.AnyType.Output()],
        )

    @classmethod
    def execute(cls, combo: DCValues) -> io.NodeOutput:
        combo_val = combo["combo"]
        if combo_val == "option1":
            return io.NodeOutput(combo["string"])
        elif combo_val == "option2":
            return io.NodeOutput(combo["integer"])
        elif combo_val == "option3":
            return io.NodeOutput(combo["image"])
        else:
            raise ValueError(f"Invalid combo: {combo_val}")


class AutogrowNamesTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplateNames(input=io.String.Input("string"), names=["a", "b", "c"])
        return io.Schema(
            node_id="AutogrowNamesTestNode",
            display_name="AutogrowNamesTest",
            category="logic",
            inputs=[
                io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output("output")],
        )

    @classmethod
    def execute(cls, autogrow: io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = "".join(vals)
        return io.NodeOutput(combined)

class AutogrowPrefixTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(input=io.String.Input("string"), prefix="string", min=1, max=10)
        return io.Schema(
            node_id="AutogrowPrefixTestNode",
            display_name="AutogrowPrefixTest",
            category="logic",
            inputs=[
                io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output("output")],
        )

    @classmethod
    def execute(cls, autogrow: io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = "".join(vals)
        return io.NodeOutput(combined)

class LogicExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SwitchNode,
            DCTestNode,
            AutogrowNamesTestNode,
            AutogrowPrefixTestNode,
        ]

async def comfy_entrypoint() -> LogicExtension:
    return LogicExtension()
