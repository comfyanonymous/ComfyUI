from inspect import cleandoc
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apis.stability_api import (
    StabilityStableUltraRequest,
    StabilityStableUltraResponse,
    StabilityAspectRatio,
    get_stability_style_presets,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    bytesio_to_image_tensor,
    tensor_to_bytesio,
)

import torch
import base64
from io import BytesIO



class StabilityStableImageUltraNode:
    """
    Generates images synchronously based on prompt and resolution.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/stability"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What you wish to see in the output image. A strong, descriptive prompt that clearly defines" +
                                    "What you wish to see in the output image. A strong, descriptive prompt that clearly defines" +
                                    "elements, colors, and subjects will lead to better results. " +
                                    "To control the weight of a given word use the format `(word:weight)`," +
                                    "where `word` is the word you'd like to control the weight of and `weight`" +
                                    "is a value between 0 and 1. For example: `The sky was a crisp (blue:0.3) and (green:0.8)`" +
                                    "would convey a sky that was blue and green, but more green than blue."
                    },
                ),
                "aspect_ratio": ([x.value for x in StabilityAspectRatio],
                    {
                        "default": StabilityAspectRatio.ratio_1_1,
                        "tooltip": "Aspect ratio of generated image.",
                    },
                ),
                "style_preset": (get_stability_style_presets(),
                    {
                        "tooltip": "Optional desired style of generated image.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967294,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "optional": {
                "image": (IO.IMAGE,),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "A blurb of text describing what you do not wish to see in the output image. This is an advanced feature."
                    },
                ),
                "image_denoise": (
                    IO.FLOAT,
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoise of input image; 0.0 yields image identical to input, 1.0 is as if no image was provided at all.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            },
        }

    def api_call(self, prompt: str, aspect_ratio: str, style_preset: str, seed: int,
                 negative_prompt: str=None, image: torch.Tensor = None, image_denoise: float=None,
                 auth_token=None):
        # prepare image binary if image present
        image_binary = None
        if image is not None:
            image_binary = tensor_to_bytesio(image, 1504 * 1504).read()
        else:
            image_denoise = None

        if not negative_prompt:
            negative_prompt = None
        if style_preset == "None":
            style_preset = None

        files = {
            "image": image_binary
        }

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/stability/v2beta/stable-image/generate/ultra",
                method=HttpMethod.POST,
                request_model=StabilityStableUltraRequest,
                response_model=StabilityStableUltraResponse,
            ),
            request=StabilityStableUltraRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                seed=seed,
                strength=image_denoise,
                style_preset=style_preset,
            ),
            files=files,
            content_type="multipart/form-data",
            auth_token=auth_token,
        )
        response_api = operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stable Image Ultra generation failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return (returned_image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "StabilityStableImageUltraNode": StabilityStableImageUltraNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StabilityStableImageUltraNode": "Stabiliy Stable Image Ultra",
}
