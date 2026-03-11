import logging
import uuid
from io import BytesIO
from urllib.parse import urlparse

import aiohttp
import torch

from .conversions import (
    audio_ndarray_to_bytesio,
    audio_tensor_to_contiguous_ndarray,
    tensor_to_bytesio,
)


# ---------------------------------------------------------------------------
# BYOK provider-specific upload helpers
# ---------------------------------------------------------------------------

_FAL_UPLOAD_DOMAINS = (".fal.ai", ".fal.run", ".fal.media")


async def upload_file_to_fal(file_bytes: BytesIO, mime_type: str) -> str:
    """Upload a file to fal.ai CDN and return the public CDN URL.

    Uses the presigned URL flow: POST to initiate, then PUT the file bytes.
    """
    from ._helpers import get_fal_auth_header

    headers = get_fal_auth_header()
    headers["Content-Type"] = "application/json"

    file_bytes.seek(0)
    data = file_bytes.read()

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        # Step 1: Initiate upload
        async with sess.post(
            "https://rest.fal.ai/storage/upload/initiate",
            params={"storage_type": "fal-cdn-v3"},
            headers=headers,
            json={"content_type": mime_type, "file_name": f"{uuid.uuid4().hex[:12]}"},
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise Exception(f"fal.ai upload initiate failed ({resp.status}): {body[:300]}")
            result = await resp.json()

        upload_url = result.get("upload_url", "")
        file_url = result.get("file_url", "")

        # Validate returned URL domains for safety
        upload_host = urlparse(upload_url).hostname or ""
        if not any(upload_host.endswith(d) for d in _FAL_UPLOAD_DOMAINS) and "amazonaws.com" not in upload_host:
            raise ValueError(f"fal.ai returned unexpected upload domain: {upload_host}")

        # Step 2: PUT the file bytes to presigned URL
        put_headers = {"Content-Type": mime_type}
        async with sess.put(upload_url, data=data, headers=put_headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise Exception(f"fal.ai file upload failed ({resp.status}): {body[:300]}")

    return file_url


async def upload_file_to_google(file_bytes: BytesIO, mime_type: str, display_name: str) -> str:
    """Upload a file to Google's Files API and return the file URI (e.g., 'files/abc123').

    Uses the resumable upload protocol.
    """
    from ._helpers import get_google_auth_header

    auth = get_google_auth_header()
    file_bytes.seek(0)
    data = file_bytes.read()
    num_bytes = len(data)

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        # Step 1: Start resumable upload
        start_headers = {
            **auth,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        async with sess.post(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            headers=start_headers,
            json={"file": {"display_name": display_name}},
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise Exception(f"Google upload initiate failed ({resp.status}): {body[:300]}")
            upload_url = resp.headers.get("X-Goog-Upload-URL", "")
            if not upload_url:
                raise Exception("Google upload initiate did not return an upload URL")

        # Step 2: Upload bytes
        upload_headers = {
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Length": str(num_bytes),
        }
        async with sess.put(upload_url, data=data, headers=upload_headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise Exception(f"Google file upload failed ({resp.status}): {body[:300]}")
            result = await resp.json()

    # Extract file URI from response
    file_name = result.get("file", {}).get("name", "")
    if not file_name:
        file_name = result.get("name", "")
    return file_name


async def upload_image_to_fal(image_tensor: torch.Tensor, mime_type: str = "image/png") -> str:
    """Convert an image tensor to bytes and upload to fal.ai CDN. Returns the CDN URL."""
    bio = tensor_to_bytesio(image_tensor, mime_type=mime_type)
    return await upload_file_to_fal(bio, mime_type)


async def upload_images_to_fal(
    image: torch.Tensor | list[torch.Tensor],
    *,
    max_images: int = 8,
    mime_type: str = "image/png",
    total_pixels: int | None = 2048 * 2048,
) -> list[str]:
    """Upload multiple images to fal.ai CDN and return CDN URLs."""
    tensors: list[torch.Tensor] = []
    if isinstance(image, list):
        for img in image:
            is_batch = len(img.shape) > 3
            if is_batch:
                tensors.extend(img[i] for i in range(img.shape[0]))
            else:
                tensors.append(img)
    else:
        is_batch = len(image.shape) > 3
        if is_batch:
            tensors.extend(image[i] for i in range(image.shape[0]))
        else:
            tensors.append(image)

    download_urls: list[str] = []
    num_to_upload = min(len(tensors), max_images)
    for idx in range(num_to_upload):
        bio = tensor_to_bytesio(tensors[idx], total_pixels=total_pixels, mime_type=mime_type)
        url = await upload_file_to_fal(bio, mime_type or "image/png")
        download_urls.append(url)
    return download_urls


async def upload_video_to_fal(
    video,
    *,
    container=None,
    codec=None,
    max_duration: int | None = None,
) -> str:
    """Convert a video to bytes and upload to fal.ai CDN. Returns the CDN URL."""
    from comfy_api.latest import Types

    if container is None:
        container = Types.VideoContainer.MP4
    if codec is None:
        codec = Types.VideoCodec.H264

    if max_duration is not None:
        try:
            actual_duration = video.get_duration()
            if actual_duration > max_duration:
                raise ValueError(
                    f"Video duration ({actual_duration:.2f}s) exceeds the maximum allowed ({max_duration}s)."
                )
        except Exception as e:
            logging.error("Error getting video duration: %s", str(e))
            raise ValueError(f"Could not verify video duration from source: {e}") from e

    upload_mime_type = f"video/{container.value.lower()}"
    video_bytes_io = BytesIO()
    video.save_to(video_bytes_io, format=container, codec=codec)
    video_bytes_io.seek(0)
    return await upload_file_to_fal(video_bytes_io, upload_mime_type)


async def upload_audio_to_fal(
    audio,
    *,
    container_format: str = "mp4",
    codec_name: str = "aac",
    mime_type: str = "audio/mp4",
) -> str:
    """Convert audio to bytes and upload to fal.ai CDN. Returns the CDN URL."""
    sample_rate: int = audio["sample_rate"]
    waveform: torch.Tensor = audio["waveform"]
    audio_data_np = audio_tensor_to_contiguous_ndarray(waveform)
    audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, sample_rate, container_format, codec_name)
    return await upload_file_to_fal(audio_bytes_io, mime_type)


async def upload_3d_model_to_fal(
    model_3d,
    file_format: str,
) -> str:
    """Upload a 3D model to fal.ai CDN. Returns the CDN URL."""
    _3d_mime_types = {
        "glb": "model/gltf-binary",
        "obj": "model/obj",
        "fbx": "application/octet-stream",
    }
    data = model_3d.get_data()
    if not isinstance(data, BytesIO):
        data = BytesIO(data)
    return await upload_file_to_fal(data, _3d_mime_types.get(file_format, "application/octet-stream"))
