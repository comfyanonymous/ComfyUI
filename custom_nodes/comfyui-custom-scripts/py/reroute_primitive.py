# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")


class ReroutePrimitive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (any, )},
        }

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    RETURN_TYPES = (any,)
    FUNCTION = "route"
    CATEGORY = "__hidden__"

    def route(self, value):
        return (value,)


class MultiPrimitive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"value": (any, )},
        }

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    RETURN_TYPES = (any,)
    FUNCTION = "listify"
    CATEGORY = "utils"
    OUTPUT_IS_LIST = (True,)

    def listify(self, **kwargs):
        return (list(kwargs.values()),)


NODE_CLASS_MAPPINGS = {
    "ReroutePrimitive|pysssss": ReroutePrimitive,
    # "MultiPrimitive|pysssss": MultiPrimitive,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReroutePrimitive|pysssss": "Reroute Primitive 🐍",
    # "MultiPrimitive|pysssss": "Multi Primitive 🐍",
}
