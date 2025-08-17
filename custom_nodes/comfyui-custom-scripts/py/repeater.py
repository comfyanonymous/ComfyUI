# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")


class Repeater:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "source": (any, {}),
            "repeats": ("INT", {"min": 0, "max": 5000, "default": 2}),
            "output": (["single", "multi"], {}),
            "node_mode": (["reuse", "create"], {}),
        }}

    RETURN_TYPES = (any,)
    FUNCTION = "repeat"
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def repeat(self, repeats, output, node_mode, **kwargs):
        if output == "multi":
            # Multi outputs are split to indiviual nodes on the frontend when serializing
            return ([kwargs["source"]],)
        elif node_mode == "reuse":
            # When reusing we have a single input node, repeat that N times
            return ([kwargs["source"]] * repeats,)
        else:
            # When creating new nodes, they'll be added dynamically when the graph is serialized
            return ((list(kwargs.values())),)


NODE_CLASS_MAPPINGS = {
    "Repeater|pysssss": Repeater,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Repeater|pysssss": "Repeater 🐍",
}
