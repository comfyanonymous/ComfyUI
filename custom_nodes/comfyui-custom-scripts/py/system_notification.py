# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")


class SystemNotification:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "message": ("STRING", {"default": "Your notification has triggered."}),
            "any": (any, {}),
            "mode": (["always", "on empty queue"], {}),
        }}

    FUNCTION = "nop"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    RETURN_TYPES = (any,)

    CATEGORY = "utils"

    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def nop(self, any, message, mode):
        return {"ui": {"message": message, "mode": mode}, "result": (any,)}


NODE_CLASS_MAPPINGS = {
    "SystemNotification|pysssss": SystemNotification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SystemNotification|pysssss": "SystemNotification 🐍",
}
