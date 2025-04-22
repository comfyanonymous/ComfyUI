import io
from inspect import cleandoc

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy_api_nodes.apis import (
    IdeogramGenerateRequest,
    IdeogramGenerateResponse,
    ImageRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse
)
from comfy_api_nodes.apis.client import ApiEndpoint, HttpMethod, SynchronousOperation


class IdeogramTextToImage(ComfyNodeABC):
    """
    Generates images synchronously based on a given prompt and optional parameters.

    Images links are available for a limited period of time; if you would like to keep the image, you must download it.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for the image generation",
                }),
                "model": (IO.COMBO, { "options": ["V_2", "V_2_TURBO", "V_1", "V_1_TURBO"], "default": "V_2", "tooltip": "Model to use for image generation"}),
            },
            "optional": {
                "aspect_ratio": (IO.COMBO, { "options": ["ASPECT_1_1", "ASPECT_4_3", "ASPECT_3_4", "ASPECT_16_9", "ASPECT_9_16", "ASPECT_2_1", "ASPECT_1_2", "ASPECT_3_2", "ASPECT_2_3", "ASPECT_4_5", "ASPECT_5_4"], "default": "ASPECT_1_1", "tooltip": "The aspect ratio for image generation. Cannot be used with resolution"
                }),
                "resolution": (IO.COMBO, { "options": ["1024x1024", "1024x1792", "1792x1024"],
                    "default": "1024x1024",
                    "tooltip": "The resolution for image generation (V2 only). Cannot be used with aspect_ratio"
                }),
                "magic_prompt_option": (IO.COMBO, { "options": ["AUTO", "ON", "OFF"],
                    "default": "AUTO",
                    "tooltip": "Determine if MagicPrompt should be used in generation"
                }),
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
                "style_type": (IO.COMBO, { "options": ["NONE", "ANIME", "CINEMATIC", "CREATIVE", "DIGITAL_ART", "PHOTOGRAPHIC"],
                    "default": "NONE",
                    "tooltip": "Style type for generation (V2+ only)"
                }),
                "negative_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of what to exclude from the image (V1/V2 only)"
                }),
                "num_images": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number"
                }),
                "color_palette": (IO.STRING, {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Color palette preset name or hex colors with weights (V2/V2_TURBO only)"
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG"
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "Example"

    def api_call(self, prompt, model, aspect_ratio=None, resolution=None,
                 magic_prompt_option="AUTO", seed=0, style_type="NONE",
                 negative_prompt="", num_images=1, color_palette="", auth_token=None):
        import io

        import numpy as np
        import requests
        import torch
        from PIL import Image

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/ideogram/generate",
                method=HttpMethod.POST,
                request_model=IdeogramGenerateRequest,
                response_model=IdeogramGenerateResponse
            ),
            request=IdeogramGenerateRequest(
                image_request=ImageRequest(
                  prompt=prompt,
                  model=model,
                  num_images=num_images,
                  seed=seed,
                  aspect_ratio=aspect_ratio if aspect_ratio != "ASPECT_1_1" else None,
                  resolution=resolution if resolution != "1024x1024" else None,
                  magic_prompt_option=magic_prompt_option if magic_prompt_option != "AUTO" else None,
                  style_type=style_type if style_type != "NONE" else None,
                  negative_prompt=negative_prompt if negative_prompt else None,
                  color_palette=None
                )
            ),
            auth_token=auth_token
        )

        response = operation.execute()

        if not response.data or len(response.data) == 0:
            raise Exception("No images were generated in the response")
        image_url = response.data[0].url

        if not image_url:
            raise Exception("No image URL was generated in the response")
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            raise Exception("Failed to download the image")

        img = Image.open(io.BytesIO(img_response.content))
        img = img.convert("RGB")  # Ensure RGB format

        # Convert to numpy array, normalize to float32 between 0 and 1
        img_array = np.array(img).astype(np.float32) / 255.0

        # Convert to torch tensor and add batch dimension
        img_tensor = torch.from_numpy(img_array)[None,]

        return (img_tensor,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

class OpenAITextToImage(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's DALL·E 3 endpoint.

    Uses the proxy at /proxy/dalle-3/generate. Returned URLs are short‑lived,
    so download or cache results if you need to keep them.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt for DALL·E",
                }),
                # TODO: add NEW MODEL
                "model": (IO.COMBO, {
                    "options": ["dall-e-3", "dall-e-2"],
                    "default": "dall-e-3",
                    "tooltip": "OpenAI model name",
                }),
            },
            "optional": {
                "n": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                    "tooltip": "How many images to generate",
                }),
                "size": (IO.COMBO, {
                    "options": ["256x256", "512x512", "1024x1792", "1792x1024", "1024x1024", "1536x1024", "1024x1536", "auto"],
                    "default": "auto",
                    "tooltip": "Image size",
                }),
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 2**31-1,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Optional random seed",
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG"
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "Example"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(self, prompt, model, n=1, size="1024x1024", seed=0, auth_token=None):
        # Validate size based on model
        if model == "dall-e-2":
            if size == "auto":
                size = "1024x1024"
            valid_sizes = ["256x256", "512x512", "1024x1024"]
            if size not in valid_sizes:
                raise ValueError(f"Size {size} not valid for dall-e-2. Must be one of: {', '.join(valid_sizes)}")
        elif model == "dall-e-3":
            if size == "auto":
                size = "1024x1024"
            valid_sizes = ["1024x1024", "1792x1024", "1024x1792"] 
            if size not in valid_sizes:
                raise ValueError(f"Size {size} not valid for dall-e-3. Must be one of: {', '.join(valid_sizes)}")
        # TODO: add NEW MODEL


        import io

        import numpy as np
        import torch
        from PIL import Image
        import requests

        # build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/openai/images/generations",
                method=HttpMethod.POST,
                request_model=OpenAIImageGenerationRequest,
                response_model=OpenAIImageGenerationResponse
            ),
            request=OpenAIImageGenerationRequest(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                seed=seed if seed != 0 else None
            ),
            auth_token=auth_token
        )

        response = operation.execute()

        # validate raw JSON response
        
        data = response.data
        if not data or len(data) == 0:
            raise Exception("No images returned from OpenAI endpoint")

        # Get base64 image data
        image_url = data[0].url
        if not image_url:
            raise Exception("No image URL was generated in the response")
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            raise Exception("Failed to download the image")

        img = Image.open(io.BytesIO(img_response.content))
        img = img.convert("RGB")  # Ensure RGB format

        # Convert to numpy array, normalize to float32 between 0 and 1
        img_array = np.array(img).astype(np.float32) / 255.0

        # Convert to torch tensor and add batch dimension
        img_tensor = torch.from_numpy(img_array)[None,]

        return (img_tensor,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "IdeogramTextToImage": IdeogramTextToImage,
    "OpenAIDalleTextToImage": OpenAITextToImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTextToImage": "Ideogram Text to Image",
    "OpenAIDalleTextToImage": "OpenAI DALL·E 3 Text to Image",
}
