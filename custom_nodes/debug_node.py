class DebugNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_input": ("CONDITIONING",),
                "text": ("STRING", { "default": "" }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "debug_node"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    def debug_node(self, cond_input, text):
        return { "ui": { "texts": ["ABC"] } }


NODE_CLASS_MAPPINGS = {
    "DebugNode": DebugNode
}