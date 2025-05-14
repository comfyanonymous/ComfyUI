from inspect import cleandoc
from google import genai
from google.genai.types import HttpOptions

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy_api_nodes.apinode_utils import validate_string
from server import PromptServer


class VertexGeminiAPI(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's GPT Image 1 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for GPT Image 1",
                    },
                ),
                "model": (
                    IO.STRING,
                    {
                        "default": "gemini-2.0-flash-001",
                        "tooltip": "The gemini model to use"
                    }
                )
            },
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "api_call"
    CATEGORY = "api node/text/gemini/vertex"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(
        self,
        prompt,
        model="gemini-2.0-flash-001",
        **kwargs
    ):
        validate_string(prompt, strip_whitespace=False)
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        print(response.text)
        PromptServer.instance.send_progress_text(response.text)
        return response.text

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VertexGeminiAPI": VertexGeminiAPI,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VertexGeminiAPI": "VertexGeminiAPI",
}