class FileSubflow:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    RETURN_TYPES = ()
    FUNCTION = ""

    CATEGORY = "loaders"
    
NODE_CLASS_MAPPINGS = {
    "FileSubflow": FileSubflow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FileSubflow": "Load Subflow",
}
