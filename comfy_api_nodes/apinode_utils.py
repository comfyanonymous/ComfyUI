from __future__ import annotations
import aiohttp
import mimetypes
from typing import Union
from server import PromptServer

import numpy as np
from PIL import Image
import torch
import base64
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
