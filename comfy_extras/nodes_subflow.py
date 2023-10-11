import folder_paths

class Subflow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "subflow_name": (folder_paths.get_filename_list("subflows"), ), }}
    RETURN_TYPES = ()
    FUNCTION = ""

    CATEGORY = "loaders"
    
NODE_CLASS_MAPPINGS = {
    "Subflow": Subflow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Subflow": "Load Subflow"
}
