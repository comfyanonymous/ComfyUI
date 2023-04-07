class Note:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ()
    FUNCTION = "Note"

    OUTPUT_NODE = False

    CATEGORY = "silver_custom"

NODE_CLASS_MAPPINGS = {
    "Note": Note
}
