from comfy.comfy_types import IO, ComfyNodeABC

class FetchApi(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "type": (IO.COMBO, {"options": ["input", "output"]}),
            "subfolder": (IO.STRING, {}),
            "filename": (IO.STRING, {}),
            "auto_download": (IO.BOOLEAN, {}),
        }}

    FUNCTION = "process"
    OUTPUT_NODE = True

    RETURN_TYPES = ()

    CATEGORY = "utils/api"

    def process(self, type, subfolder, filename, auto_download, **kwargs):
        return {
            "ui": {
                "result": [type, subfolder, filename]
            }
        }

NODE_CLASS_MAPPINGS = {
    "FetchApi": FetchApi
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FetchApi": "Fetch"
}
