import folder_paths
import json
import os.path as osp

class Subflow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "subflow_name": (folder_paths.get_filename_list("subflows"), ), }}
    RETURN_TYPES = ()
    FUNCTION = "exec_subflow"

    CATEGORY = "loaders"

    def exec_subflow(self, subflow_name):
        subflow_path = folder_paths.get_full_path("subflows", subflow_name)
        with open(subflow_path) as f:
            if osp.splitext(subflow_path)[1] == ".json":
                subflow_data = json.load(f)
                return subflow_data

        return None
    
NODE_CLASS_MAPPINGS = {
    "Subflow": Subflow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Subflow": "Load Subflow"
}
