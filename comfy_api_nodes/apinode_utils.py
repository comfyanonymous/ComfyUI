from __future__ import annotations
import aiohttp
import io
import logging
import mimetypes
from typing import Optional, Union
from comfy.utils import common_upscale
from comfy_api.input_impl import VideoFromFile
from comfy_api.util import VideoContainer, VideoCodec
from comfy_api.input.video_types import VideoInput
from comfy_api.input.basic_types import AudioInput
from comfy_api_nodes.apis.client import (
    ApiClient,
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    UploadRequest,
    UploadResponse,
)
from server import PromptServer


import numpy as np
from PIL import Image
import torch
import math
import base64
import uuid
from io import BytesIO
import av


async def download_url_to_video_output(video_url: str, timeout: int = None) -> VideoFromFile:
    """Downloads a video from a URL and returns a `VIDEO` output.

    Args:
        video_url: The URL of the video to download.

    Returns:
        A Comfy node `VIDEO` output.
    """
    video_io = await download_url_to_bytesio(video_url, timeout)
    if video_io is None:
        error_msg = f"Failed to download video from {video_url}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    return VideoFromFile(video_io)


def downscale_image_tensor(image, total_pixels=1536 * 1024) -> torch.Tensor:
    """Downscale input image tensor to roughly the specified total pixels."""
    samples = image.movedim(-1, 1)
    total = int(total_pixels)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1, -1)
    return s


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
        elif calculated_ratio > maximum_ratio:
            raise TypeError(
                f"Aspect ratio cannot reduce to any greater than {maximum_ratio_str} ({maximum_ratio}), but was {aspect_ratio} ({calculated_ratio})."
            )
    return aspect_ratio


def mimetype_to_extension(mime_type: str) -> str:
    """Converts a MIME type to a file extension."""
    return mime_type.split("/")[-1].lower()


async def download_url_to_bytesio(url: str, timeout: int = None) -> BytesIO:
    """Downloads content from a URL using requests and returns it as BytesIO.

    Args:
        url: The URL to download.
        timeout: Request timeout in seconds. Defaults to None (no timeout).

    Returns:
        BytesIO object containing the downloaded content.
    """
    timeout_cfg = aiohttp.ClientTimeout(total=timeout) if timeout else None
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            return BytesIO(await resp.read())


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


async def download_url_to_image_tensor(url: str, timeout: int = None) -> torch.Tensor:
    """Downloads an image from a URL and returns a [B, H, W, C] tensor."""
    image_bytesio = await download_url_to_bytesio(url, timeout)
    return bytesio_to_image_tensor(image_bytesio)


def process_image_response(response_content: bytes | str) -> torch.Tensor:
    """Uses content from a Response object and converts it to a torch.Tensor"""
    return bytesio_to_image_tensor(BytesIO(response_content))


def _tensor_to_pil(image: torch.Tensor, total_pixels: int = 2048 * 2048) -> Image.Image:
    """Converts a single torch.Tensor image [H, W, C] to a PIL Image, optionally downscaling."""
    if len(image.shape) > 3:
        image = image[0]
    # TODO: remove alpha if not allowed and present
    input_tensor = image.cpu()
    input_tensor = downscale_image_tensor(
        input_tensor.unsqueeze(0), total_pixels=total_pixels
    ).squeeze()
    image_np = (input_tensor.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    return img


def _pil_to_bytesio(img: Image.Image, mime_type: str = "image/png") -> BytesIO:
    """Converts a PIL Image to a BytesIO object."""
    if not mime_type:
        mime_type = "image/png"

    img_byte_arr = io.BytesIO()
    # Derive PIL format from MIME type (e.g., 'image/png' -> 'PNG')
    pil_format = mime_type.split("/")[-1].upper()
    if pil_format == "JPG":
        pil_format = "JPEG"
    img.save(img_byte_arr, format=pil_format)
    img_byte_arr.seek(0)
    return img_byte_arr


def tensor_to_bytesio(
    image: torch.Tensor,
    name: Optional[str] = None,
    total_pixels: int = 2048 * 2048,
    mime_type: str = "image/png",
) -> BytesIO:
    """Converts a torch.Tensor image to a named BytesIO object.

    Args:
        image: Input torch.Tensor image.
        name: Optional filename for the BytesIO object.
        total_pixels: Maximum total pixels for potential downscaling.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp', 'video/mp4').

    Returns:
        Named BytesIO object containing the image data.
    """
    if not mime_type:
        mime_type = "image/png"

    pil_image = _tensor_to_pil(image, total_pixels=total_pixels)
    img_binary = _pil_to_bytesio(pil_image, mime_type=mime_type)
    img_binary.name = (
        f"{name if name else uuid.uuid4()}.{mimetype_to_extension(mime_type)}"
    )
    return img_binary


def tensor_to_base64_string(
    image_tensor: torch.Tensor,
    total_pixels: int = 2048 * 2048,
    mime_type: str = "image/png",
) -> str:
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


def tensor_to_data_uri(
    image_tensor: torch.Tensor,
    total_pixels: int = 2048 * 2048,
    mime_type: str = "image/png",
) -> str:
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


def video_to_base64_string(
    video: VideoInput,
    container_format: VideoContainer = None,
    codec: VideoCodec = None
) -> str:
    """
    Converts a video input to a base64 string.

    Args:
        video: The video input to convert
        container_format: Optional container format to use (defaults to video.container if available)
        codec: Optional codec to use (defaults to video.codec if available)
    """
    video_bytes_io = io.BytesIO()

    # Use provided format/codec if specified, otherwise use video's own if available
    format_to_use = container_format if container_format is not None else getattr(video, 'container', VideoContainer.MP4)
    codec_to_use = codec if codec is not None else getattr(video, 'codec', VideoCodec.H264)

    video.save_to(video_bytes_io, format=format_to_use, codec=codec_to_use)
    video_bytes_io.seek(0)
    return base64.b64encode(video_bytes_io.getvalue()).decode("utf-8")


async def upload_video_to_comfyapi(
    video: VideoInput,
    auth_kwargs: Optional[dict[str, str]] = None,
    container: VideoContainer = VideoContainer.MP4,
    codec: VideoCodec = VideoCodec.H264,
    max_duration: Optional[int] = None,
) -> str:
    """
    Uploads a single video to ComfyUI API and returns its download URL.
    Uses the specified container and codec for saving the video before upload.

    Args:
        video: VideoInput object (Comfy VIDEO type).
        auth_kwargs: Optional authentication token(s).
        container: The video container format to use (default: MP4).
        codec: The video codec to use (default: H264).
        max_duration: Optional maximum duration of the video in seconds. If the video is longer than this, an error will be raised.

    Returns:
        The download URL for the uploaded video file.
    """
    if max_duration is not None:
        try:
            actual_duration = video.duration_seconds
            if actual_duration is not None and actual_duration > max_duration:
                raise ValueError(
                    f"Video duration ({actual_duration:.2f}s) exceeds the maximum allowed ({max_duration}s)."
                )
        except Exception as e:
            logging.error(f"Error getting video duration: {e}")
            raise ValueError(f"Could not verify video duration from source: {e}") from e

    upload_mime_type = f"video/{container.value.lower()}"
    filename = f"uploaded_video.{container.value.lower()}"

    # Convert VideoInput to BytesIO using specified container/codec
    video_bytes_io = io.BytesIO()
    video.save_to(video_bytes_io, format=container, codec=codec)
    video_bytes_io.seek(0)

    return await upload_file_to_comfyapi(video_bytes_io, filename, upload_mime_type, auth_kwargs)


def audio_tensor_to_contiguous_ndarray(waveform: torch.Tensor) -> np.ndarray:
    """
    Prepares audio waveform for av library by converting to a contiguous numpy array.

    Args:
        waveform: a tensor of shape (1, channels, samples) derived from a Comfy `AUDIO` type.

    Returns:
        Contiguous numpy array of the audio waveform. If the audio was batched,
            the first item is taken.
    """
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Expected waveform tensor shape (1, channels, samples)")

    # If batch is > 1, take first item
    if waveform.shape[0] > 1:
        waveform = waveform[0]

    # Prepare for av: remove batch dim, move to CPU, make contiguous, convert to numpy array
    audio_data_np = waveform.squeeze(0).cpu().contiguous().numpy()
    if audio_data_np.dtype != np.float32:
        audio_data_np = audio_data_np.astype(np.float32)

    return audio_data_np


def audio_ndarray_to_bytesio(
    audio_data_np: np.ndarray,
    sample_rate: int,
    container_format: str = "mp4",
    codec_name: str = "aac",
) -> BytesIO:
    """
    Encodes a numpy array of audio data into a BytesIO object.
    """
    audio_bytes_io = io.BytesIO()
    with av.open(audio_bytes_io, mode="w", format=container_format) as output_container:
        audio_stream = output_container.add_stream(codec_name, rate=sample_rate)
        frame = av.AudioFrame.from_ndarray(
            audio_data_np,
            format="fltp",
            layout="stereo" if audio_data_np.shape[0] > 1 else "mono",
        )
        frame.sample_rate = sample_rate
        frame.pts = 0

        for packet in audio_stream.encode(frame):
            output_container.mux(packet)

        # Flush stream
        for packet in audio_stream.encode(None):
            output_container.mux(packet)

    audio_bytes_io.seek(0)
    return audio_bytes_io


async def upload_audio_to_comfyapi(
    audio: AudioInput,
    auth_kwargs: Optional[dict[str, str]] = None,
    container_format: str = "mp4",
    codec_name: str = "aac",
    mime_type: str = "audio/mp4",
    filename: str = "uploaded_audio.mp4",
) -> str:
    """
    Uploads a single audio input to ComfyUI API and returns its download URL.
    Encodes the raw waveform into the specified format before uploading.

    Args:
        audio: a Comfy `AUDIO` type (contains waveform tensor and sample_rate)
        auth_kwargs: Optional authentication token(s).

    Returns:
        The download URL for the uploaded audio file.
    """
    sample_rate: int = audio["sample_rate"]
    waveform: torch.Tensor = audio["waveform"]
    audio_data_np = audio_tensor_to_contiguous_ndarray(waveform)
    audio_bytes_io = audio_ndarray_to_bytesio(
        audio_data_np, sample_rate, container_format, codec_name
    )

    return await upload_file_to_comfyapi(audio_bytes_io, filename, mime_type, auth_kwargs)


def audio_to_base64_string(
    audio: AudioInput, container_format: str = "mp4", codec_name: str = "aac"
) -> str:
    """Converts an audio input to a base64 string."""
    sample_rate: int = audio["sample_rate"]
    waveform: torch.Tensor = audio["waveform"]
    audio_data_np = audio_tensor_to_contiguous_ndarray(waveform)
    audio_bytes_io = audio_ndarray_to_bytesio(
        audio_data_np, sample_rate, container_format, codec_name
    )
    audio_bytes = audio_bytes_io.getvalue()
    return base64.b64encode(audio_bytes).decode("utf-8")


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


def validate_string(
    string: str,
    strip_whitespace=True,
    field_name="prompt",
    min_length=None,
    max_length=None,
):
    if string is None:
        raise Exception(f"Field '{field_name}' cannot be empty.")
    if strip_whitespace:
        string = string.strip()
    if min_length and len(string) < min_length:
        raise Exception(
            f"Field '{field_name}' cannot be shorter than {min_length} characters; was {len(string)} characters long."
        )
    if max_length and len(string) > max_length:
        raise Exception(
            f" Field '{field_name} cannot be longer than {max_length} characters; was {len(string)} characters long."
        )


def image_tensor_pair_to_batch(
    image1: torch.Tensor, image2: torch.Tensor
) -> torch.Tensor:
    """
    Converts a pair of image tensors to a batch tensor.
    If the images are not the same size, the smaller image is resized to
    match the larger image.
    """
    if image1.shape[1:] != image2.shape[1:]:
        image2 = common_upscale(
            image2.movedim(-1, 1),
            image1.shape[2],
            image1.shape[1],
            "bilinear",
            "center",
        ).movedim(1, -1)
    return torch.cat((image1, image2), dim=0)
