import asyncio
import contextlib
import logging
import time
import uuid
from io import BytesIO
from typing import Optional, Union
from urllib.parse import urlparse

import aiohttp
import torch
from pydantic import BaseModel, Field

from comfy_api.latest import IO, Input
from comfy_api.util import VideoCodec, VideoContainer

from . import request_logger
from ._helpers import is_processing_interrupted, sleep_with_interrupt
from .client import (
    ApiEndpoint,
    _diagnose_connectivity,
    _display_time_progress,
    sync_op,
)
from .common_exceptions import ApiServerError, LocalNetworkError, ProcessingInterrupted
from .conversions import (
    audio_ndarray_to_bytesio,
    audio_tensor_to_contiguous_ndarray,
    tensor_to_bytesio,
)


class UploadRequest(BaseModel):
    file_name: str = Field(..., description="Filename to upload")
    content_type: Optional[str] = Field(
        None,
        description="Mime type of the file. For example: image/png, image/jpeg, video/mp4, etc.",
    )


class UploadResponse(BaseModel):
    download_url: str = Field(..., description="URL to GET uploaded file")
    upload_url: str = Field(..., description="URL to PUT file to upload")


async def upload_images_to_comfyapi(
    cls: type[IO.ComfyNode],
    image: torch.Tensor,
    *,
    max_images: int = 8,
    mime_type: Optional[str] = None,
    wait_label: Optional[str] = "Uploading",
) -> list[str]:
    """
    Uploads images to ComfyUI API and returns download URLs.
    To upload multiple images, stack them in the batch dimension first.
    """
    # if batch, try to upload each file if max_images is greater than 0
    download_urls: list[str] = []
    is_batch = len(image.shape) > 3
    batch_len = image.shape[0] if is_batch else 1

    for idx in range(min(batch_len, max_images)):
        tensor = image[idx] if is_batch else image
        img_io = tensor_to_bytesio(tensor, mime_type=mime_type)
        url = await upload_file_to_comfyapi(cls, img_io, img_io.name, mime_type, wait_label)
        download_urls.append(url)
    return download_urls


async def upload_audio_to_comfyapi(
    cls: type[IO.ComfyNode],
    audio: Input.Audio,
    *,
    container_format: str = "mp4",
    codec_name: str = "aac",
    mime_type: str = "audio/mp4",
    filename: str = "uploaded_audio.mp4",
) -> str:
    """
    Uploads a single audio input to ComfyUI API and returns its download URL.
    Encodes the raw waveform into the specified format before uploading.
    """
    sample_rate: int = audio["sample_rate"]
    waveform: torch.Tensor = audio["waveform"]
    audio_data_np = audio_tensor_to_contiguous_ndarray(waveform)
    audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, sample_rate, container_format, codec_name)
    return await upload_file_to_comfyapi(cls, audio_bytes_io, filename, mime_type)


async def upload_video_to_comfyapi(
    cls: type[IO.ComfyNode],
    video: Input.Video,
    *,
    container: VideoContainer = VideoContainer.MP4,
    codec: VideoCodec = VideoCodec.H264,
    max_duration: Optional[int] = None,
) -> str:
    """
    Uploads a single video to ComfyUI API and returns its download URL.
    Uses the specified container and codec for saving the video before upload.
    """
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
    filename = f"uploaded_video.{container.value.lower()}"

    # Convert VideoInput to BytesIO using specified container/codec
    video_bytes_io = BytesIO()
    video.save_to(video_bytes_io, format=container, codec=codec)
    video_bytes_io.seek(0)

    return await upload_file_to_comfyapi(cls, video_bytes_io, filename, upload_mime_type)


async def upload_file_to_comfyapi(
    cls: type[IO.ComfyNode],
    file_bytes_io: BytesIO,
    filename: str,
    upload_mime_type: Optional[str],
    wait_label: Optional[str] = "Uploading",
) -> str:
    """Uploads a single file to ComfyUI API and returns its download URL."""
    if upload_mime_type is None:
        request_object = UploadRequest(file_name=filename)
    else:
        request_object = UploadRequest(file_name=filename, content_type=upload_mime_type)
    create_resp = await sync_op(
        cls,
        endpoint=ApiEndpoint(path="/customers/storage", method="POST"),
        data=request_object,
        response_model=UploadResponse,
        final_label_on_success=None,
        monitor_progress=False,
    )
    await upload_file(
        cls,
        create_resp.upload_url,
        file_bytes_io,
        content_type=upload_mime_type,
        wait_label=wait_label,
    )
    return create_resp.download_url


async def upload_file(
    cls: type[IO.ComfyNode],
    upload_url: str,
    file: Union[BytesIO, str],
    *,
    content_type: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    wait_label: Optional[str] = None,
) -> None:
    """
    Upload a file to a signed URL (e.g., S3 pre-signed PUT) with retries, Comfy progress display, and interruption.

    Args:
        cls: Node class (provides auth context + UI progress hooks).
        upload_url: Pre-signed PUT URL.
        file: BytesIO or path string.
        content_type: Explicit MIME type. If None, we *suppress* Content-Type.
        max_retries: Maximum retry attempts.
        retry_delay: Initial delay in seconds.
        retry_backoff: Exponential backoff factor.
        wait_label: Progress label shown in Comfy UI.

    Raises:
        ProcessingInterrupted, LocalNetworkError, ApiServerError, Exception
    """
    if isinstance(file, BytesIO):
        with contextlib.suppress(Exception):
            file.seek(0)
        data = file.read()
    elif isinstance(file, str):
        with open(file, "rb") as f:
            data = f.read()
    else:
        raise ValueError("file must be a BytesIO or a filesystem path string")

    headers: dict[str, str] = {}
    skip_auto_headers: set[str] = set()
    if content_type:
        headers["Content-Type"] = content_type
    else:
        skip_auto_headers.add("Content-Type")  # Don't let aiohttp add Content-Type, it can break the signed request

    attempt = 0
    delay = retry_delay
    start_ts = time.monotonic()
    op_uuid = uuid.uuid4().hex[:8]
    while True:
        attempt += 1
        operation_id = _generate_operation_id("PUT", upload_url, attempt, op_uuid)
        timeout = aiohttp.ClientTimeout(total=None)
        stop_evt = asyncio.Event()

        async def _monitor():
            try:
                while not stop_evt.is_set():
                    if is_processing_interrupted():
                        return
                    if wait_label:
                        _display_time_progress(cls, wait_label, int(time.monotonic() - start_ts), None)
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return

        monitor_task = asyncio.create_task(_monitor())
        sess: Optional[aiohttp.ClientSession] = None
        try:
            try:
                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method="PUT",
                    request_url=upload_url,
                    request_headers=headers or None,
                    request_params=None,
                    request_data=f"[File data {len(data)} bytes]",
                )
            except Exception as e:
                logging.debug("[DEBUG] upload request logging failed: %s", e)

            sess = aiohttp.ClientSession(timeout=timeout)
            req = sess.put(upload_url, data=data, headers=headers, skip_auto_headers=skip_auto_headers)
            req_task = asyncio.create_task(req)

            done, pending = await asyncio.wait({req_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

            if monitor_task in done and req_task in pending:
                req_task.cancel()
                raise ProcessingInterrupted("Upload cancelled")

            try:
                resp = await req_task
            except asyncio.CancelledError:
                raise ProcessingInterrupted("Upload cancelled") from None

            async with resp:
                if resp.status >= 400:
                    with contextlib.suppress(Exception):
                        try:
                            body = await resp.json()
                        except Exception:
                            body = await resp.text()
                        msg = f"Upload failed with status {resp.status}"
                        request_logger.log_request_response(
                            operation_id=operation_id,
                            request_method="PUT",
                            request_url=upload_url,
                            response_status_code=resp.status,
                            response_headers=dict(resp.headers),
                            response_content=body,
                            error_message=msg,
                        )
                    if resp.status in {408, 429, 500, 502, 503, 504} and attempt <= max_retries:
                        await sleep_with_interrupt(
                            delay,
                            cls,
                            wait_label,
                            start_ts,
                            None,
                            display_callback=_display_time_progress if wait_label else None,
                        )
                        delay *= retry_backoff
                        continue
                    raise Exception(f"Failed to upload (HTTP {resp.status}).")
                try:
                    request_logger.log_request_response(
                        operation_id=operation_id,
                        request_method="PUT",
                        request_url=upload_url,
                        response_status_code=resp.status,
                        response_headers=dict(resp.headers),
                        response_content="File uploaded successfully.",
                    )
                except Exception as e:
                    logging.debug("[DEBUG] upload response logging failed: %s", e)
                return
        except asyncio.CancelledError:
            raise ProcessingInterrupted("Task cancelled") from None
        except (aiohttp.ClientError, OSError) as e:
            if attempt <= max_retries:
                with contextlib.suppress(Exception):
                    request_logger.log_request_response(
                        operation_id=operation_id,
                        request_method="PUT",
                        request_url=upload_url,
                        request_headers=headers or None,
                        request_data=f"[File data {len(data)} bytes]",
                        error_message=f"{type(e).__name__}: {str(e)} (will retry)",
                    )
                await sleep_with_interrupt(
                    delay,
                    cls,
                    wait_label,
                    start_ts,
                    None,
                    display_callback=_display_time_progress if wait_label else None,
                )
                delay *= retry_backoff
                continue

            diag = await _diagnose_connectivity()
            if not diag["internet_accessible"]:
                raise LocalNetworkError(
                    "Unable to connect to the network. Please check your internet connection and try again."
                ) from e
            raise ApiServerError("The API service appears unreachable at this time.") from e
        finally:
            stop_evt.set()
            if monitor_task:
                monitor_task.cancel()
                with contextlib.suppress(Exception):
                    await monitor_task
            if sess:
                with contextlib.suppress(Exception):
                    await sess.close()


def _generate_operation_id(method: str, url: str, attempt: int, op_uuid: str) -> str:
    try:
        parsed = urlparse(url)
        slug = (parsed.path.rsplit("/", 1)[-1] or parsed.netloc or "upload").strip("/").replace("/", "_")
    except Exception:
        slug = "upload"
    return f"{method}_{slug}_{op_uuid}_try{attempt}"
