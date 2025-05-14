import os
import uuid
from inspect import cleandoc
from google import genai
from google.genai.types import HttpOptions, Part
from google.cloud import storage

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy_api_nodes.apinode_utils import validate_string
from server import PromptServer

### Documentation ###
'''
export GOOGLE_CLOUD_PROJECT="<NAME OF GCP PROJECT>"
export GOOGLE_CLOUD_LOCATION="<NAME OF REGION e.g. us-central1>"
export GOOGLE_GENAI_USE_VERTEXAI=True Use the vertex api

How to create a service account:
1. On GCP console go to IAM & Admin
2. Select Service accounts
3. on the top select + create service account
4. put in the name of service account and select Create and continue
5. In "Grant this service account access to project" select "Vertex AI User" & "Storage Object Admin"
6. Once created go to the list of service account and select your service account
7. Click on keys and click Add key (Json)
8. download the key and keep it in secure place on your system
9. use this key to auth by exporting var below:
export GOOGLE_APPLICATION_CREDENTIALS="/Path/to/Key.json"
'''

BUCKET_NAME = "comfyui-interview-temp"

def get_model_list():
    return list([
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-04-17"
    ])


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
                    get_model_list(),
                    {
                        "default": "gemini-2.0-flash-001",
                        "tooltip": "Select the model you would like to use"
                    }
                )
            },
            "optional": {
                "image_path": (
                    IO.STRING,
                    {
                        "default": None,
                        "tooltip": "Optional reference path to an image for inference.",
                    }
                )
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "api_call"
    CATEGORY = "api node/text/gemini/vertex"
    DESCRIPTION = cleandoc(__doc__ or "")
    OUTPUT_NODE = True
    API_NODE = True

    def api_call(
        self,
        prompt,
        model="gemini-2.0-flash-001",
        image_path=None,
        unique_id=None,
        **kwargs
    ):
        validate_string(prompt, strip_whitespace=False)
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        contents = [prompt]
        if image_path:
            storage_client = storage.Client()

            # Define bucket and file info
            bucket_name = "comfyui-interview-temp"
            source_file = image_path              # local path
            file_name = os.path.basename(source_file)
            destination_blob = f"{uuid.uuid4()}/{file_name}"         # name in bucket
            # Upload
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob)
            blob.upload_from_filename(source_file)

            print(f"Uploaded {source_file} to gs://{bucket_name}/{destination_blob}")
            image_part = Part.from_uri(
                file_uri=f"gs://{bucket_name}/{destination_blob}",
                mime_type="image/jpeg"
            )
            contents.append(image_part)
        response = client.models.generate_content(
            model=model,
            contents=contents,
        )
        print(response.text)
        PromptServer.instance.send_progress_text(response.text, node_id=unique_id)
        return (response.text,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VertexGeminiAPI": VertexGeminiAPI,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VertexGeminiAPI": "VertexGeminiAPI",
}