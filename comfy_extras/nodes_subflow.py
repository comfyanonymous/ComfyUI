import folder_paths

class LoadSubflow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "subflow_name": (folder_paths.get_filename_list("subflows"), ), }}
    RETURN_TYPES = ()
    FUNCTION = ""

    CATEGORY = "loaders"

class FileSubflow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "subflow_name": (folder_paths.get_filename_list("subflows"), )} }
    RETURN_TYPES = ()
    FUNCTION = ""

    CATEGORY = "utils"

class InMemorySubflow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {} }
    RETURN_TYPES = ()
    FUNCTION = ""

    CATEGORY = ""
    
NODE_CLASS_MAPPINGS = {
    "LoadSubflow": LoadSubflow,
    "FileSubflow": FileSubflow,
    "InMemorySubflow": InMemorySubflow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Subflow": "Load Subflow",
    "FileSubflow": "File Subflow",
    "InMemorySubflow": "In Memory Subflow"
}
