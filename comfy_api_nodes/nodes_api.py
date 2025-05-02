import base64
import io
import math
from inspect import cleandoc

import numpy as np
import requests
import torch
from PIL import Image

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.utils import common_upscale
from comfy_api_nodes.apis import (
    OpenAIImageEditRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse,
)
from comfy_api_nodes.apis.client import ApiEndpoint, HttpMethod, SynchronousOperation


def downscale_input(image):
    samples = image.movedim(-1,1)
    #downscaling input images to roughly the same size as the outputs
    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s

def validate_and_cast_response(response):
    # validate raw JSON response
    data = response.data
    if not data or len(data) == 0:
        raise Exception("No images returned from API endpoint")

    # Initialize list to store image tensors
    image_tensors = []

    # Process each image in the data array
    for image_data in data:
        image_url = image_data.url
        b64_data = image_data.b64_json

        if not image_url and not b64_data:
            raise Exception("No image was generated in the response")

        if b64_data:
            img_data = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_data))

        elif image_url:
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                raise Exception("Failed to download the image")
            img = Image.open(io.BytesIO(img_response.content))

        img = img.convert("RGBA")

        # Convert to numpy array, normalize to float32 between 0 and 1
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)

        # Add to list of tensors
        image_tensors.append(img_tensor)

    return torch.stack(image_tensors, dim=0)

class OpenAIDalle2(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's DALL·E 2 endpoint.

    Uses the proxy at /proxy/openai/images/generations. Returned URLs are short‑lived,
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
            },
            "optional": {
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 2**31-1,
                    "step": 1,
                    "display": "number",
                    "tooltip": "not implemented yet in backend",
                }),
                "size": (IO.COMBO, {
                    "options": ["256x256", "512x512", "1024x1024"],
                    "default": "1024x1024",
                    "tooltip": "Image size",
                }),
                "n": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                    "tooltip": "How many images to generate",
                }),
                "image": (IO.IMAGE, {
                    "default": None,
                    "tooltip": "Optional reference image for image editing.",
                }),
                "mask": (IO.MASK, {
                    "default": None,
                    "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG"
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(self, prompt, seed=0, image=None, mask=None, n=1, size="1024x1024", auth_token=None):
        model = "dall-e-2"
        path = "/proxy/openai/images/generations"
        request_class = OpenAIImageGenerationRequest
        img_binary = None

        if image is not None and mask is not None:
            path = "/proxy/openai/images/edits"
            request_class = OpenAIImageEditRequest

            input_tensor = image.squeeze().cpu()
            height, width, channels = input_tensor.shape
            rgba_tensor = torch.ones(height, width, 4, device="cpu")
            rgba_tensor[:, :, :channels] = input_tensor

            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            rgba_tensor[:,:,3] = (1-mask.squeeze().cpu())

            rgba_tensor = downscale_input(rgba_tensor.unsqueeze(0)).squeeze()

            image_np = (rgba_tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_binary = img_byte_arr#.getvalue()
            img_binary.name = "image.png"
        elif image is not None or mask is not None:
            raise Exception("Dall-E 2 image editing requires an image AND a mask")

        # Build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                seed=seed,
            ),
            files={
                "image": img_binary,
            } if img_binary else None,
            auth_token=auth_token
        )

        response = operation.execute()

        img_tensor = validate_and_cast_response(response)
        return (img_tensor,)

class OpenAIDalle3(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's DALL·E 3 endpoint.

    Uses the proxy at /proxy/openai/images/generations. Returned URLs are short‑lived,
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
            },
            "optional": {
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 2**31-1,
                    "step": 1,
                    "display": "number",
                    "tooltip": "not implemented yet in backend",
                }),
                "quality" : (IO.COMBO, {
                    "options": ["standard","hd"],
                    "default": "standard",
                    "tooltip": "Image quality",
                }),
                "style": (IO.COMBO, {
                    "options": ["natural","vivid"],
                    "default": "natural",
                    "tooltip": "Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.",
                }),
                "size": (IO.COMBO, {
                    "options": ["1024x1024", "1024x1792", "1792x1024"],
                    "default": "1024x1024",
                    "tooltip": "Image size",
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG"
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(self, prompt, seed=0, style="natural", quality="standard", size="1024x1024", auth_token=None):
        model = "dall-e-3"

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
                quality=quality,
                size=size,
                style=style,
                seed=seed,
            ),
            auth_token=auth_token
        )

        response = operation.execute()

        img_tensor = validate_and_cast_response(response)
        return (img_tensor,)

class OpenAIGPTImage1(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's GPT Image 1 endpoint.

    Uses the proxy at /proxy/openai/images/generations. Returned URLs are short‑lived,
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
                    "tooltip": "Text prompt for GPT Image 1",
                }),
            },
            "optional": {
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 2**31-1,
                    "step": 1,
                    "display": "number",
                    "tooltip": "not implemented yet in backend",
                }),
                "quality": (IO.COMBO, {
                    "options": ["low","medium","high"],
                    "default": "low",
                    "tooltip": "Image quality, affects cost and generation time.",
                }),
                "background": (IO.COMBO, {
                    "options": ["opaque","transparent"],
                    "default": "opaque",
                    "tooltip": "Return image with or without background",
                }),
                "size": (IO.COMBO, {
                    "options": ["auto", "1024x1024", "1024x1536", "1536x1024"],
                    "default": "auto",
                    "tooltip": "Image size",
                }),
                "n": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                    "tooltip": "How many images to generate",
                }),
                "image": (IO.IMAGE, {
                    "default": None,
                    "tooltip": "Optional reference image for image editing.",
                }),
                "mask": (IO.MASK, {
                    "default": None,
                    "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                }),
                "moderation": (IO.COMBO, {
                    "options": ["low","auto"],
                    "default": "low",
                    "tooltip": "Moderation level",
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG"
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(self, prompt, seed=0, quality="low", background="opaque", image=None, mask=None, n=1, size="1024x1024", auth_token=None, moderation="low"):
        model = "gpt-image-1"
        path = "/proxy/openai/images/generations"
        request_class = OpenAIImageGenerationRequest
        img_binaries = []
        mask_binary = None
        files = []

        if image is not None:
            path = "/proxy/openai/images/edits"
            request_class = OpenAIImageEditRequest

            batch_size = image.shape[0]


            for i in range(batch_size):
                single_image = image[i:i+1]
                scaled_image = downscale_input(single_image).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                img_binary = img_byte_arr
                img_binary.name = f"image_{i}.png"

                img_binaries.append(img_binary)
                if batch_size == 1:
                    files.append(("image", img_binary))
                else:
                    files.append(("image[]", img_binary))

        if mask is not None:
            if image.shape[0] != 1:
                raise Exception("Cannot use a mask with multiple image")
            if image is None:
                raise Exception("Cannot use a mask without an input image")
            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:,:,3] = (1-mask.squeeze().cpu())

            scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()

            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_img_byte_arr = io.BytesIO()
            mask_img.save(mask_img_byte_arr, format='PNG')
            mask_img_byte_arr.seek(0)
            mask_binary = mask_img_byte_arr
            mask_binary.name = "mask.png"
            files.append(("mask", mask_binary))


        # Build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                quality=quality,
                background=background,
                n=n,
                seed=seed,
                size=size,
                moderation=moderation,
            ),
            files=files if files else None,
            auth_token=auth_token
        )

        response = operation.execute()

        img_tensor = validate_and_cast_response(response)
        return (img_tensor,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OpenAIDalle2": OpenAIDalle2,
    "OpenAIDalle3": OpenAIDalle3,
    "OpenAIGPTImage1": OpenAIGPTImage1,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIDalle2": "OpenAI DALL·E 2",
    "OpenAIDalle3": "OpenAI DALL·E 3",
    "OpenAIGPTImage1": "OpenAI GPT Image 1",
}
