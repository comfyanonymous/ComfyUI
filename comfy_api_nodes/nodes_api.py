import base64
import io
import math
from inspect import cleandoc
from comfy.comfy_types.node_typing import FileLocator
from typing import Literal, Optional
from comfy.utils import common_upscale
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api_nodes.apis import (
    OpenAIImageEditRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageEditRequest,
    OpenAIImageGenerationResponse,
    MinimaxVideoGenerationRequest,
    MinimaxVideoGenerationResponse,
    MinimaxFileRetrieveResponse,
    MinimaxTaskResultResponse,
    IdeogramGenerateRequest,
    IdeogramGenerateResponse,
    BFLFluxProGenerateRequest,
    BFLFluxProGenerateResponse,
    ImageRequest,
    Model
)
from comfy_api_nodes.apis.BFLPolling import BFLStatus
from comfy_api_nodes.apis.luma_api import (
    LumaImageModel,
    LumaVideoModel,
    LumaVideoOutputResolution,
    LumaVideoModelOutputDuration,
    LumaAspectRatio,
    LumaState,
    LumaImageGenerationRequest,
    LumaGenerationRequest,
    LumaGeneration,
    LumaCharacterRef,
    LumaModifyImageRef,
    LumaImageIdentity,
    LumaReference,
    LumaReferenceChain,
    LumaImageReference,
    LumaKeyframes,
    LumaIO,
)
from comfy_api_nodes.apis.client import ApiClient, ApiEndpoint, HttpMethod, SynchronousOperation, PollingOperation, EmptyRequest, UploadRequest, UploadResponse

import numpy as np
from PIL import Image
import requests
import torch
import math
import base64
import logging
import json
import av
import os
import time
import uuid
import folder_paths
from io import BytesIO

def downscale_input(image, total_pixels=1536*1024):
    samples = image.movedim(-1,1)
    #downscaling input images to roughly the same size as the outputs
    total = int(total_pixels)
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

def validate_aspect_ratio(aspect_ratio: str, minimum_ratio: float, maximum_ratio: float, minimum_ratio_str: str, maximum_ratio_str: str):
    # get ratio values
    numbers = aspect_ratio.split(':')
    if len(numbers) != 2:
        raise Exception(f"Aspect ratio must be in the format X:Y, such as 16:9, but was {aspect_ratio}.")
    try:
        numerator = int(numbers[0])
        denominator = int(numbers[1])
    except ValueError:
        raise Exception(f"Aspect ratio must contain numbers separated by ':', such as 16:9, but was {aspect_ratio}.")
    calculated_ratio = numerator/denominator
    # if not close to minimum and maximum, check bounds
    if not math.isclose(calculated_ratio, minimum_ratio) or not math.isclose(calculated_ratio, maximum_ratio):
        if calculated_ratio < minimum_ratio:
            raise Exception(f"Aspect ratio cannot reduce to any less than {minimum_ratio_str} ({minimum_ratio}), but was {aspect_ratio} ({calculated_ratio}).")
        elif calculated_ratio > maximum_ratio:
            raise Exception(f"Aspect ratio cannot reduce to any greater than {maximum_ratio_str} ({maximum_ratio}), but was {aspect_ratio} ({calculated_ratio}).")
    return aspect_ratio


def mimetype_to_extension(mime_type: str) -> str:
    """Converts a MIME type to a file extension."""
    return mime_type.split('/')[-1].lower()


def download_url_to_bytesio(url: str, timeout: int = None) -> BytesIO:
    """Downloads content from a URL using requests and returns it as BytesIO.

    Args:
        url: The URL to download.
        timeout: Request timeout in seconds. Defaults to None (no timeout).

    Returns:
        BytesIO object containing the downloaded content.
    """
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
    return BytesIO(response.content)


def bytesio_to_image_tensor(image_bytesio: BytesIO, mode: str = "RGBA") -> torch.Tensor:
    """Converts image data from BytesIO to a torch.Tensor.

    Args:
        image_bytesio: BytesIO object containing the image data.
        mode: The PIL mode to convert the image to (e.g., "RGB", "RGBA").

    Returns:
        A torch.Tensor representing the image (1, H, W, C).

    Raises:
        PIL.UnidentifiedImageError: If the image data cannot be identified.
        ValueError: If the specified mode is invalid.
    """
    image = Image.open(image_bytesio)
    image = image.convert(mode)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def process_image_response(response: requests.Response):
    '''Uses content from a Response object and converts it to a torch.Tensor'''
    return bytesio_to_image_tensor(BytesIO(response.content))


def _tensor_to_pil(image: torch.Tensor, total_pixels: int = 2048*2048) -> Image.Image:
    """Converts a single torch.Tensor image [H, W, C] to a PIL Image, optionally downscaling."""
    if len(image.shape) > 3:
        image = image[0]
    # TODO: remove alpha if not allowed and present
    input_tensor = image.cpu()
    input_tensor = downscale_input(input_tensor.unsqueeze(0), total_pixels=total_pixels).squeeze()
    image_np = (input_tensor.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    return img


def _pil_to_bytesio(img: Image.Image, mime_type: str = 'image/png') -> BytesIO:
    """Converts a PIL Image to a BytesIO object."""
    img_byte_arr = io.BytesIO()
    # Derive PIL format from MIME type (e.g., 'image/png' -> 'PNG')
    pil_format = mime_type.split('/')[-1].upper()
    if pil_format == 'JPG':
        pil_format = 'JPEG'
    img.save(img_byte_arr, format=pil_format)
    img_byte_arr.seek(0)
    return img_byte_arr


def tensor_to_bytesio(image: torch.Tensor, name: Optional[str] = None, total_pixels: int = 2048*2048, mime_type: str = 'image/png') -> BytesIO:
    """Converts a torch.Tensor image to a named BytesIO object.

    Args:
        image: Input torch.Tensor image.
        name: Optional filename for the BytesIO object.
        total_pixels: Maximum total pixels for potential downscaling.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp', 'video/mp4').

    Returns:
        Named BytesIO object containing the image data.
    """
    pil_image = _tensor_to_pil(image, total_pixels=total_pixels)
    img_binary = _pil_to_bytesio(pil_image, mime_type=mime_type)
    img_binary.name = f"{name if name else uuid.uuid4()}.{mimetype_to_extension(mime_type)}"
    return img_binary


def tensor_to_base64_string(image_tensor: torch.Tensor, total_pixels: int = 2048*2048, mime_type: str = 'image/png') -> str:
    """Convert [B, H, W, C] or [H, W, C] tensor to a base64 string.

    Args:
        image_tensor: Input torch.Tensor image.
        total_pixels: Maximum total pixels for potential downscaling.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp', 'video/mp4').

    Returns:
        Base64 encoded string of the image.
    """
    pil_image = _tensor_to_pil(image_tensor, total_pixels=total_pixels)
    img_byte_arr = _pil_to_bytesio(pil_image, mime_type=mime_type)
    img_bytes = img_byte_arr.getvalue()
    # Encode bytes to base64 string
    base64_encoded_string = base64.b64encode(img_bytes).decode("utf-8")
    return base64_encoded_string


def tensor_to_data_uri(image_tensor: torch.Tensor, total_pixels: int = 2048 * 2048, mime_type: str = 'image/png') -> str:
    """Converts a tensor image to a Data URI string.

    Args:
        image_tensor: Input torch.Tensor image.
        total_pixels: Maximum total pixels for potential downscaling.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp').

    Returns:
        Data URI string (e.g., 'data:image/png;base64,...').
    """
    base64_string = tensor_to_base64_string(image_tensor, total_pixels, mime_type)
    return f"data:{mime_type};base64,{base64_string}"


def upload_images_to_comfyapi(image: torch.Tensor, max_images=8, auth_token=None) -> list[str]:
    # if batch, try to upload each file if max_images is greater than 0
    idx_image = 0
    download_urls: list[str] = []
    is_batch = len(image.shape) > 3
    batch_length = 1
    if is_batch:
        batch_length = image.shape[0]
    while True:
        curr_image = image
        if len(image.shape) > 3:
            curr_image = image[idx_image]
        # get BytesIO version of image
        img_binary = tensor_to_bytesio(curr_image)
        # first, request upload/download urls from comfy API
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/customers/storage",
                method=HttpMethod.POST,
                request_model=UploadRequest,
                response_model=UploadResponse
            ),
            request=UploadRequest(
                filename=img_binary.name
            ),
            auth_token=auth_token
        )
        response = operation.execute()

        upload_response = ApiClient.upload_file(response.upload_url, img_binary)
        # verify success
        try:
            upload_response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Could not upload one or more images: {e}")
        # add download_url to list
        download_urls.append(response.download_url)

        idx_image += 1
        # stop uploading additional files if done
        if is_batch and max_images > 0:
            if idx_image >= max_images:
                break
            if idx_image >= batch_length:
                break
    return download_urls

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
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

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
        import torch
        from PIL import Image
        import io
        import numpy as np
        import requests

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

class FluxProUltraImageNode(ComfyNodeABC):
    """
    Generates images synchronously based on prompt and resolution.
    """
    MINIMUM_RATIO = 1/4
    MAXIMUM_RATIO = 4/1
    MINIMUM_RATIO_STR = "1:4"
    MAXIMUM_RATIO_STR = "4:1"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for the image generation",
                }),
                "prompt_upsampling": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result)."
                }),
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
                }),
                "aspect_ratio": (IO.STRING, {
                    "default": "16:9",
                    "tooltip": "Aspect ratio of image; must be between 1:4 and 4:1.",
                }),
                "raw": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "When True, generate less processed, more natural-looking images."
                }),
            },
            "optional": {
                "image_prompt": (IO.IMAGE, ),
                "image_prompt_strength": (IO.FLOAT, {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend between the prompt and the image prompt.",
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, aspect_ratio: str):
        try:
            validate_aspect_ratio(aspect_ratio, minimum_ratio=cls.MINIMUM_RATIO, maximum_ratio=cls.MAXIMUM_RATIO,
                                  minimum_ratio_str=cls.MINIMUM_RATIO_STR, maximum_ratio_str=cls.MAXIMUM_RATIO_STR)
        except Exception as e:
            return str(e)
        return True

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node"

    def api_call(self, prompt: str, aspect_ratio: str, prompt_upsampling=False, raw=False, seed=0, image_prompt=None, image_prompt_strength=0.1, auth_token=None, **kwargs):
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.1-ultra/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxProGenerateRequest,
                response_model=BFLFluxProGenerateResponse
            ),
            request=BFLFluxProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                seed=seed,
                aspect_ratio=validate_aspect_ratio(aspect_ratio, minimum_ratio=self.MINIMUM_RATIO, maximum_ratio=self.MAXIMUM_RATIO,
                                                   minimum_ratio_str=self.MINIMUM_RATIO_STR, maximum_ratio_str=self.MAXIMUM_RATIO_STR),
                raw=raw,
                image_prompt=image_prompt if image_prompt is None else self._convert_image_to_base64(image_prompt),
                image_prompt_strength=None if image_prompt is None else round(image_prompt_strength, 2),
            ),
            auth_token=auth_token
        )
        output_image = self._handle_bfl_synchronous_operation(operation)
        return (output_image,)

    def _handle_bfl_synchronous_operation(self, operation: SynchronousOperation, timeout_bfl_calls=360):
        response_api: BFLFluxProGenerateResponse = operation.execute()
        return self._poll_until_generated(response_api.polling_url, timeout=timeout_bfl_calls)

    def _poll_until_generated(self, polling_url: str, timeout=360):
        # used bfl-comfy-nodes to verify code implementation:
        # https://github.com/black-forest-labs/bfl-comfy-nodes/tree/main
        start_time = time.time()
        retries_404 = 0
        max_retries_404 = 5
        retry_404_seconds = 2
        retry_202_seconds = 2
        retry_pending_seconds = 1
        request = requests.Request(method=HttpMethod.GET, url=polling_url)
        # NOTE: should True loop be replaced with checking if workflow has been interrupted?
        while True:
            response = requests.Session().send(request.prepare())
            if response.status_code == 200:
                result = response.json()
                if result["status"] == BFLStatus.ready:
                    img_url = result["result"]["sample"]
                    img_response = requests.get(img_url)
                    return process_image_response(img_response)
                elif result["status"] in [BFLStatus.request_moderated, BFLStatus.content_moderated]:
                    status = result["status"]
                    raise Exception(f"BFL API did not return an image due to: {status}.")
                elif result["status"] == BFLStatus.error:
                    raise Exception(f"BFL API encountered an error: {result}.")
                elif result["status"] == BFLStatus.pending:
                    time.sleep(retry_pending_seconds)
                    continue
            elif response.status_code == 404:
                if retries_404 < max_retries_404:
                    retries_404 += 1
                    time.sleep(retry_404_seconds)
                    continue
                raise Exception(f"BFL API could not find task after {max_retries_404} tries.")
            elif response.status_code == 202:
                time.sleep(retry_202_seconds)
            elif time.time() - start_time > timeout:
                raise Exception(f"BFL API experienced a timeout; could not return request under {timeout} seconds.")
            else:
                raise Exception(f"BFL API encountered an error: {response.json()}")

    def _convert_image_to_base64(self, image: torch.Tensor):
        scaled_image = downscale_input(image, total_pixels=2048*2048)
        # remove batch dimension if present
        if len(scaled_image.shape) > 3:
            scaled_image = scaled_image[0]
        image_np = (scaled_image.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return base64.b64encode(img_byte_arr.getvalue()).decode()

class LumaReferenceNode:
    """
    Holds an image and weight for use with Luma Generate Image node.
    """
    RETURN_TYPES = (LumaIO.LUMA_REF,)
    RETURN_NAMES = ("luma_ref",)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "create_luma_reference"
    CATEGORY = "api node/Luma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, {
                    "tooltip": "Image to use as reference.",
                }),
                "weight": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Weight of image reference.",
                }),
            },
            "optional": {
                "luma_ref": (LumaIO.LUMA_REF,)
            }
        }

    def create_luma_reference(self, image: torch.Tensor, weight: float, luma_ref: LumaReferenceChain=None):
        if luma_ref is not None:
            luma_ref = luma_ref.clone()
        else:
            luma_ref = LumaReferenceChain()
        luma_ref.add(LumaReference(image=image, weight=round(weight, 2)))
        return (luma_ref, )

class LumaImageGenerationNode:
    """
    Generates images synchronously based on prompt and aspect ratio.
    """
    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for the image generation",
                }),
                "model": ([model.value for model in LumaImageModel],),
                "aspect_ratio": ([ratio.value for ratio in LumaAspectRatio], {
                    "default": LumaAspectRatio.ratio_16_9,
                }),
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                }),
                "style_image_weight": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Weight of style image. Ignored if no style_image provided.",
                }),
            },
            "optional": {
                "image_luma_ref": (LumaIO.LUMA_REF, {
                    "tooltip": "Luma Reference node connection to influence generation with input images; up to 4 images can be considered."
                }),
                "style_image": (IO.IMAGE, {
                    "tooltip": "Style reference image; only 1 image will be used."
                }),
                "character_image": (IO.IMAGE, {
                    "tooltip": "Character reference images; can be a batch of multiple, up to 4 images can be considered."
                })
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            }
        }

    def api_call(self, prompt: str, model: str, aspect_ratio: str, seed, style_image_weight: float,
                 image_luma_ref: LumaReferenceChain=None, style_image: torch.Tensor=None, character_image: torch.Tensor=None,
                 auth_token=None, **kwargs):
        # handle image_luma_ref
        api_image_ref = None
        if image_luma_ref is not None:
            api_image_ref = self._convert_luma_refs(image_luma_ref, auth_token=auth_token)
        # handle style_luma_ref
        api_style_ref = None
        if style_image is not None:
            api_style_ref = self._convert_style_image(style_image, weight=style_image_weight, auth_token=auth_token)
        # handle character_ref images
        character_ref = None
        if character_image is not None:
            download_urls = upload_images_to_comfyapi(character_image, max_images=4, auth_token=auth_token)
            character_ref = LumaCharacterRef(identity0=LumaImageIdentity(images=download_urls))

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations/image",
                method=HttpMethod.POST,
                request_model=LumaImageGenerationRequest,
                response_model=LumaGeneration
            ),
            request=LumaImageGenerationRequest(
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                image_ref=api_image_ref,
                style_ref=api_style_ref,
                character_ref=character_ref,
            ),
            auth_token=auth_token
        )
        response_api: LumaGeneration = operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            auth_token=auth_token,
        )
        response_poll = operation.execute()

        img_response = requests.get(response_poll.assets.image)
        img = process_image_response(img_response)
        return (img,)

    def _convert_luma_refs(self, luma_ref: LumaReferenceChain, max_refs: int, auth_token=None):
        luma_urls = []
        ref_count = 0
        for ref in luma_ref.refs:
            download_urls = upload_images_to_comfyapi(ref.image, max_images=1, auth_token=auth_token)
            luma_urls.append(download_urls[0])
            ref_count += 1
            if ref_count >= max_refs:
                break
        return luma_ref.create_api_model(download_urls=luma_urls, max_refs=max_refs)

    def _convert_style_image(self, style_image: torch.Tensor, weight: float, auth_token=None):
        chain = LumaReferenceChain(first_ref=LumaReference(image=style_image, weight=weight))
        return self._convert_luma_refs(chain, max_refs=1, auth_token=auth_token)

class LumaImageModifyNode:
    """
    Modifies images synchronously based on prompt and aspect ratio.
    """
    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for the image generation",
                }),
                "image_weight": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Weight of the image; the closer to 0.0, the less the image will be modified."
                }),
                "model": ([model.value for model in LumaImageModel],),
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                }),
            },
            "optional": {
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            }
        }

    def api_call(self, prompt: str, model: str, image: torch.Tensor, image_weight: float, seed, auth_token=None, **kwargs):
        # first, upload image
        download_urls = upload_images_to_comfyapi(image, max_images=1, auth_token=auth_token)
        image_url = download_urls[0]
        # next, make Luma call with download url provided
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations/image",
                method=HttpMethod.POST,
                request_model=LumaImageGenerationRequest,
                response_model=LumaGeneration
            ),
            request=LumaImageGenerationRequest(
                prompt=prompt,
                model=model,
                modify_image_ref=LumaModifyImageRef(
                    url=image_url,
                    weight=round(image_weight, 2)
                ),
            ),
            auth_token=auth_token
        )
        response_api: LumaGeneration = operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            auth_token=auth_token,
        )
        response_poll = operation.execute()

        img_response = requests.get(response_poll.assets.image)
        img = process_image_response(img_response)
        return (img,)

class LumaTextToVideoGenerationNode:
    """
    Generates videos synchronously based on prompt and output_size.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"

    RETURN_TYPES = (IO.VIDEO,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for the video generation",
                }),
                "model": ([model.value for model in LumaVideoModel],),
                "aspect_ratio": ([ratio.value for ratio in LumaAspectRatio], {
                    "default": LumaAspectRatio.ratio_16_9,
                }),
                "resolution": ([resolution.value for resolution in LumaVideoOutputResolution], {
                    "default": LumaVideoOutputResolution.res_540p,
                }),
                "duration": ([dur.value for dur in LumaVideoModelOutputDuration],),
                "loop": (IO.BOOLEAN, {
                    "default": False,
                }),
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                }),
            },
            "optional": {
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            }
        }

    def api_call(self, prompt: str, model: str, aspect_ratio: str, resolution: str, duration: str, loop: bool, seed,
                 auth_token=None, **kwargs):
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations",
                method=HttpMethod.POST,
                request_model=LumaGenerationRequest,
                response_model=LumaGeneration
            ),
            request=LumaGenerationRequest(
                prompt=prompt,
                model=model,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                loop=loop,
            ),
            auth_token=auth_token
        )
        response_api: LumaGeneration = operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            auth_token=auth_token,
        )
        response_poll = operation.execute()

        vid_response = requests.get(response_poll.assets.video)
        return (VideoFromFile(BytesIO(vid_response.content)), )

class LumaImageToVideoGenerationNode:
    """
    Generates videos synchronously based on prompt, input images, and output_size.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"

    RETURN_TYPES = (IO.VIDEO,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for the video generation",
                }),
                "model": ([model.value for model in LumaVideoModel],),
                # "aspect_ratio": ([ratio.value for ratio in LumaAspectRatio], {
                #     "default": LumaAspectRatio.ratio_16_9,
                # }),
                "resolution": ([resolution.value for resolution in LumaVideoOutputResolution], {
                    "default": LumaVideoOutputResolution.res_540p,
                }),
                "duration": ([dur.value for dur in LumaVideoModelOutputDuration],),
                "loop": (IO.BOOLEAN, {
                    "default": False,
                }),
                "seed": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                }),
            },
            "optional": {
                "first_image": (IO.IMAGE, {
                    "tooltip": "First frame of generated video."
                }),
                "last_image": (IO.IMAGE, {
                    "tooltip": "Last frame of generated video."
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            }
        }

    def api_call(self, prompt: str, model: str, resolution: str, duration: str, loop: bool, seed,
                 first_image: torch.Tensor=None, last_image: torch.Tensor=None,
                 auth_token=None, **kwargs):
        if first_image is None and last_image is None:
            raise Exception("At least one of first_image and last_image requires an input.")
        keyframes = self._convert_to_keyframes(first_image, last_image, auth_token)

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations",
                method=HttpMethod.POST,
                request_model=LumaGenerationRequest,
                response_model=LumaGeneration
            ),
            request=LumaGenerationRequest(
                prompt=prompt,
                model=model,
                aspect_ratio=LumaAspectRatio.ratio_16_9,
                resolution=resolution,
                duration=duration,
                loop=loop,
                keyframes=keyframes
            ),
            auth_token=auth_token
        )
        response_api: LumaGeneration = operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            auth_token=auth_token,
        )
        response_poll = operation.execute()

        vid_response = requests.get(response_poll.assets.video)
        return (VideoFromFile(BytesIO(vid_response.content)), )

    def _convert_to_keyframes(self, first_image: torch.Tensor=None, last_image: torch.Tensor=None, auth_token=None):
        if first_image is None and last_image is None:
            return None
        frame0 = None
        frame1 = None
        if first_image is not None:
            download_urls = upload_images_to_comfyapi(first_image, max_images=1, auth_token=auth_token)
            frame0 = LumaImageReference(type='image', url=download_urls[0])
        if last_image is not None:
            download_urls = upload_images_to_comfyapi(last_image, max_images=1, auth_token=auth_token)
            frame1 = LumaImageReference(type='image', url=download_urls[0])
        return LumaKeyframes(frame0=frame0, frame1=frame1)

class MinimaxTextToVideoNode:
    """
    Generates videos synchronously based on a prompt, and optional parameters using Minimax's API.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt to guide the video generation",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "model": (
                    [
                        "T2V-01",
                        "I2V-01-Director",
                        "S2V-01",
                        "I2V-01",
                        "I2V-01-live",
                        "T2V-01",
                    ],
                    {
                        "default": "T2V-01",
                        "tooltip": "Model to use for video generation",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Generates videos from prompts using Minimax's API"
    FUNCTION = "generate_video"
    CATEGORY = "video"
    API_NODE = True
    OUTPUT_NODE = True

    def generate_video(
        self,
        prompt_text,
        filename_prefix,
        seed=0,
        model="T2V-01",
        prompt=None,
        extra_pnginfo=None,
        auth_token=None,
    ):
        video_generate_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/minimax/video_generation",
                method=HttpMethod.POST,
                request_model=MinimaxVideoGenerationRequest,
                response_model=MinimaxVideoGenerationResponse,
            ),
            request=MinimaxVideoGenerationRequest(
                model=Model(model),
                prompt=prompt_text,
                callback_url=None,
                first_frame_image=None,
                subject_reference=None,
                prompt_optimizer=None,
            ),
            auth_token=auth_token,
        )
        response = video_generate_operation.execute()

        task_id = response.task_id

        video_generate_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path="/proxy/minimax/query/video_generation",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=MinimaxTaskResultResponse,
                query_params={"task_id": task_id},
            ),
            completed_statuses=["Success"],
            failed_statuses=["Fail"],
            status_extractor=lambda x: x.status.value,
            auth_token=auth_token,
        )
        task_result = video_generate_operation.execute()

        file_id = task_result.file_id
        if file_id is None:
            raise Exception("Request was not successful. Missing file ID.")
        file_retrieve_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/minimax/files/retrieve",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=MinimaxFileRetrieveResponse,
                query_params={"file_id": int(file_id)},
            ),
            request=EmptyRequest(),
            auth_token=auth_token,
        )
        file_result = file_retrieve_operation.execute()

        file_url = file_result.file.download_url
        if file_url is None:
            raise Exception(f"No video was found in the response. Full response: {file_result.model_dump()}")
        logging.info(f"Generated video URL: {file_url}")

        # Construct the save path
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )
        file_basename = f"{filename}_{counter:05}_.mp4"
        save_path = os.path.join(full_output_folder, file_basename)

        # Download the video data
        video_response = requests.get(file_url)
        video_data = video_response.content

        # Save the video data to a file
        with open(save_path, "wb") as video_file:
            video_file.write(video_data)

        # Add workflow metadata to the video container
        if prompt is not None or extra_pnginfo is not None:
            try:
                container = av.open(save_path, mode="r+")
                if prompt is not None:
                    container.metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        container.metadata[x] = json.dumps(extra_pnginfo[x])
                container.close()
            except Exception as e:
                logging.warning(f"Failed to add metadata to video: {e}")

        # Create a FileLocator for the frontend to use for the preview
        results: list[FileLocator] = [
            {
                "filename": file_basename,
                "subfolder": subfolder,
                "type": self.type,
            }
        ]

        return {"ui": {"images": results, "animated": (True,)}}

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OpenAIDalle2": OpenAIDalle2,
    "OpenAIDalle3": OpenAIDalle3,
    "OpenAIGPTImage1": OpenAIGPTImage1,
    "IdeogramTextToImage": IdeogramTextToImage,
    "FluxProUltraImageNode": FluxProUltraImageNode,
    "LumaImageNode": LumaImageGenerationNode,
    "LumaImageModifyNode": LumaImageModifyNode,
    "LumaReferenceNode": LumaReferenceNode,
    "LumaVideoNode": LumaTextToVideoGenerationNode,
    "LumaImageToVideoNode": LumaImageToVideoGenerationNode,
    "MinimaxTextToVideoNode": MinimaxTextToVideoNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIDalle2": "OpenAI DALL·E 2",
    "OpenAIDalle3": "OpenAI DALL·E 3",
    "OpenAIGPTImage1": "OpenAI GPT Image 1",
    "IdeogramTextToImage": "Ideogram Text to Image",
    "FluxProUltraImageNode": "Flux 1.1 [pro] Ultra Image",
    "LumaImageNode": "Luma Text to Image",
    "LumaImageModifyNode": "Luma Image to Image",
    "LumaReferenceNode": "Luma Reference",
    "LumaVideoNode": "Luma Text to Video",
    "LumaImageToVideoNode": "Luma Image to Video",
    "MinimaxTextToVideoNode": "Minimax Text to Video",
}
