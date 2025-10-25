from __future__ import annotations
import aiohttp
import mimetypes
from typing import Optional, Union
from comfy.utils import common_upscale
from comfy_api_nodes.apis.client import (
    ApiClient,
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    UploadRequest,
    UploadResponse,
)
from server import PromptServer
from comfy.cli_args import args

import numpy as np
from PIL import Image
import torch
import math
import base64
from .util import tensor_to_bytesio, bytesio_to_image_tensor
from io import BytesIO


async def validate_and_cast_response(
    response, timeout: int = None, node_id: Union[str, None] = None
) -> torch.Tensor:
    """Validates and casts a response to a torch.Tensor.

    Args:
        response: The response to validate and cast.
        timeout: Request timeout in seconds. Defaults to None (no timeout).

    Returns:
        A torch.Tensor representing the image (1, H, W, C).

    Raises:
        ValueError: If the response is not valid.
    """
    # validate raw JSON response
    data = response.data
    if not data or len(data) == 0:
        raise ValueError("No images returned from API endpoint")

    # Initialize list to store image tensors
    image_tensors: list[torch.Tensor] = []

    # Process each image in the data array
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        for img_data in data:
            img_bytes: bytes
            if img_data.b64_json:
                img_bytes = base64.b64decode(img_data.b64_json)
            elif img_data.url:
                if node_id:
                    PromptServer.instance.send_progress_text(f"Result URL: {img_data.url}", node_id)
                async with session.get(img_data.url) as resp:
                    if resp.status != 200:
                        raise ValueError("Failed to download generated image")
                    img_bytes = await resp.read()
            else:
                raise ValueError("Invalid image payload – neither URL nor base64 data present.")

            pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
            arr = np.asarray(pil_img).astype(np.float32) / 255.0
            image_tensors.append(torch.from_numpy(arr))

    return torch.stack(image_tensors, dim=0)


def validate_aspect_ratio(
    aspect_ratio: str,
    minimum_ratio: float,
    maximum_ratio: float,
    minimum_ratio_str: str,
    maximum_ratio_str: str,
) -> float:
    """Validates and casts an aspect ratio string to a float.

    Args:
        aspect_ratio: The aspect ratio string to validate.
        minimum_ratio: The minimum aspect ratio.
        maximum_ratio: The maximum aspect ratio.
        minimum_ratio_str: The minimum aspect ratio string.
        maximum_ratio_str: The maximum aspect ratio string.

    Returns:
        The validated and cast aspect ratio.

    Raises:
        Exception: If the aspect ratio is not valid.
    """
    # get ratio values
    numbers = aspect_ratio.split(":")
    if len(numbers) != 2:
        raise TypeError(
            f"Aspect ratio must be in the format X:Y, such as 16:9, but was {aspect_ratio}."
        )
    try:
        numerator = int(numbers[0])
        denominator = int(numbers[1])
    except ValueError as exc:
        raise TypeError(
            f"Aspect ratio must contain numbers separated by ':', such as 16:9, but was {aspect_ratio}."
        ) from exc
    calculated_ratio = numerator / denominator
    # if not close to minimum and maximum, check bounds
    if not math.isclose(calculated_ratio, minimum_ratio) or not math.isclose(
        calculated_ratio, maximum_ratio
    ):
        if calculated_ratio < minimum_ratio:
            raise TypeError(
                f"Aspect ratio cannot reduce to any less than {minimum_ratio_str} ({minimum_ratio}), but was {aspect_ratio} ({calculated_ratio})."
            )
        if calculated_ratio > maximum_ratio:
            raise TypeError(
                f"Aspect ratio cannot reduce to any greater than {maximum_ratio_str} ({maximum_ratio}), but was {aspect_ratio} ({calculated_ratio})."
            )
    return aspect_ratio


async def download_url_to_bytesio(
    url: str, timeout: int = None, auth_kwargs: Optional[dict[str, str]] = None
) -> BytesIO:
    """Downloads content from a URL using requests and returns it as BytesIO.

    Args:
        url: The URL to download.
        timeout: Request timeout in seconds. Defaults to None (no timeout).

    Returns:
        BytesIO object containing the downloaded content.
    """
    headers = {}
    if url.startswith("/proxy/"):
        url = str(args.comfy_api_base).rstrip("/") + url
        auth_token = auth_kwargs.get("auth_token")
        comfy_api_key = auth_kwargs.get("comfy_api_key")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        elif comfy_api_key:
            headers["X-API-KEY"] = comfy_api_key
    timeout_cfg = aiohttp.ClientTimeout(total=timeout) if timeout else None
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            return BytesIO(await resp.read())


def process_image_response(response_content: bytes | str) -> torch.Tensor:
    """Uses content from a Response object and converts it to a torch.Tensor"""
    return bytesio_to_image_tensor(BytesIO(response_content))


def text_filepath_to_base64_string(filepath: str) -> str:
    """Converts a text file to a base64 string."""
    with open(filepath, "rb") as f:
        file_content = f.read()
    return base64.b64encode(file_content).decode("utf-8")


def text_filepath_to_data_uri(filepath: str) -> str:
    """Converts a text file to a data URI."""
    base64_string = text_filepath_to_base64_string(filepath)
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return f"data:{mime_type};base64,{base64_string}"


async def upload_file_to_comfyapi(
    file_bytes_io: BytesIO,
    filename: str,
    upload_mime_type: Optional[str],
    auth_kwargs: Optional[dict[str, str]] = None,
) -> str:
    """
    Uploads a single file to ComfyUI API and returns its download URL.

    Args:
        file_bytes_io: BytesIO object containing the file data.
        filename: The filename of the file.
        upload_mime_type: MIME type of the file.
        auth_kwargs: Optional authentication token(s).

    Returns:
        The download URL for the uploaded file.
    """
    if upload_mime_type is None:
        request_object = UploadRequest(file_name=filename)
    else:
        request_object = UploadRequest(file_name=filename, content_type=upload_mime_type)
    operation = SynchronousOperation(
        endpoint=ApiEndpoint(
            path="/customers/storage",
            method=HttpMethod.POST,
            request_model=UploadRequest,
            response_model=UploadResponse,
        ),
        request=request_object,
        auth_kwargs=auth_kwargs,
    )

    response: UploadResponse = await operation.execute()
    await ApiClient.upload_file(response.upload_url, file_bytes_io, content_type=upload_mime_type)
    return response.download_url


async def upload_images_to_comfyapi(
    image: torch.Tensor,
    max_images=8,
    auth_kwargs: Optional[dict[str, str]] = None,
    mime_type: Optional[str] = None,
) -> list[str]:
    """
    Uploads images to ComfyUI API and returns download URLs.
    To upload multiple images, stack them in the batch dimension first.

    Args:
        image: Input torch.Tensor image.
        max_images: Maximum number of images to upload.
        auth_kwargs: Optional authentication token(s).
        mime_type: Optional MIME type for the image.
    """
    # if batch, try to upload each file if max_images is greater than 0
    download_urls: list[str] = []
    is_batch = len(image.shape) > 3
    batch_len = image.shape[0] if is_batch else 1

    for idx in range(min(batch_len, max_images)):
        tensor = image[idx] if is_batch else image
        img_io = tensor_to_bytesio(tensor, mime_type=mime_type)
        url = await upload_file_to_comfyapi(img_io, img_io.name, mime_type, auth_kwargs)
        download_urls.append(url)
    return download_urls


def resize_mask_to_image(
    mask: torch.Tensor,
    image: torch.Tensor,
    upscale_method="nearest-exact",
    crop="disabled",
    allow_gradient=True,
    add_channel_dim=False,
):
    """
    Resize mask to be the same dimensions as an image, while maintaining proper format for API calls.
    """
    _, H, W, _ = image.shape
    mask = mask.unsqueeze(-1)
    mask = mask.movedim(-1, 1)
    mask = common_upscale(
        mask, width=W, height=H, upscale_method=upscale_method, crop=crop
    )
    mask = mask.movedim(1, -1)
    if not add_channel_dim:
        mask = mask.squeeze(-1)
    if not allow_gradient:
        mask = (mask > 0.5).float()
    return mask
