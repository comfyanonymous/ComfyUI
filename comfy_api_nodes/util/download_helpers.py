import asyncio
import contextlib
import logging
import time
import uuid
from io import BytesIO
from typing import Optional, Union, IO
from pathlib import Path

import aiohttp
import torch
from aiohttp.client_exceptions import ClientError, ContentTypeError
from urllib.parse import urlparse

from comfy_api_nodes.apis import request_logger

from ._helpers import _is_processing_interrupted
from .common_exceptions import ProcessingInterrupted, LocalNetworkError, ApiServerError
from .api_client import _diagnose_connectivity
from .conversions import bytesio_to_image_tensor


_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


async def download_url_to_bytesio(
    url: str,
    timeout: Optional[float] = None,
    *,
    dest: Optional[Union[BytesIO, IO[bytes], str, Path]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
) -> None:
    """Stream-download a URL into memory or to a provided destination.

    Raises:
        ProcessingInterrupted, LocalNetworkError, ApiServerError, Exception (HTTP and other errors)
    """
    attempt = 0
    delay = retry_delay

    while True:
        attempt += 1
        op_id = _generate_operation_id("GET", url, attempt)
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        stop_evt = asyncio.Event()

        async def _monitor():
            try:
                while not stop_evt.is_set():
                    if _is_processing_interrupted():
                        return
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return

        monitor_task: Optional[asyncio.Task] = None
        sess: Optional[aiohttp.ClientSession] = None

        # Open file path if a path was provided
        is_path_sink = isinstance(dest, (str, Path))
        fhandle = None
        try:
            try:
                request_logger.log_request_response(
                    operation_id=op_id,
                    request_method="GET",
                    request_url=url,
                )
            except Exception as e:
                logging.debug("[DEBUG] download request logging failed: %s", e)

            monitor_task = asyncio.create_task(_monitor())
            sess = aiohttp.ClientSession(timeout=timeout_cfg)
            req_task = asyncio.create_task(sess.get(url))

            done, pending = await asyncio.wait({req_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

            # Interruption wins the race
            if monitor_task in done and req_task in pending:
                req_task.cancel()
                raise ProcessingInterrupted("Task cancelled")

            resp = await req_task
            async with resp:
                if resp.status >= 400:
                    # Attempt to capture body for logging (do not log huge binaries)
                    with contextlib.suppress(Exception):
                        try:
                            body = await resp.json()
                        except (ContentTypeError, ValueError):
                            text = await resp.text()
                            body = text if len(text) <= 4096 else f"[text {len(text)} bytes]"
                        request_logger.log_request_response(
                            operation_id=op_id,
                            request_method="GET",
                            request_url=url,
                            response_status_code=resp.status,
                            response_headers=dict(resp.headers),
                            response_content=body,
                            error_message=f"HTTP {resp.status}",
                        )

                    if resp.status in _RETRY_STATUS and attempt <= max_retries:
                        await _sleep_with_cancel(delay)
                        delay *= retry_backoff
                        continue
                    raise Exception(f"Failed to download (HTTP {resp.status}).")

                # Prepare path sink if needed
                if is_path_sink:
                    p = Path(str(dest))
                    with contextlib.suppress(Exception):
                        p.parent.mkdir(parents=True, exist_ok=True)
                    fhandle = open(p, "wb")
                    sink = fhandle
                else:
                    sink = dest  # BytesIO or file-like

                # Stream body in chunks to sink with cancellation checks
                written = 0
                last_tick = time.monotonic()
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    sink.write(chunk)
                    written += len(chunk)
                    now = time.monotonic()
                    if now - last_tick >= 1.0:
                        last_tick = now
                        if _is_processing_interrupted():
                            raise ProcessingInterrupted("Task cancelled")

                if isinstance(dest, BytesIO):
                    dest.seek(0)

                try:
                    request_logger.log_request_response(
                        operation_id=op_id,
                        request_method="GET",
                        request_url=url,
                        response_status_code=resp.status,
                        response_headers=dict(resp.headers),
                        response_content=f"[streamed {written} bytes to dest]",
                    )
                except Exception as e:
                    logging.debug("[DEBUG] download response logging failed: %s", e)
                return
        except ProcessingInterrupted:
            logging.debug("Download was interrupted by user")
            raise
        except (ClientError, asyncio.TimeoutError) as e:
            if attempt <= max_retries:
                with contextlib.suppress(Exception):
                    request_logger.log_request_response(
                        operation_id=op_id,
                        request_method="GET",
                        request_url=url,
                        error_message=f"{type(e).__name__}: {str(e)} (will retry)",
                    )
                await _sleep_with_cancel(delay)
                delay *= retry_backoff
                continue

            diag = await _diagnose_connectivity()
            if diag.get("is_local_issue"):
                raise LocalNetworkError(
                    "Unable to connect to the network. Please check your internet connection and try again."
                ) from e
            raise ApiServerError("The remote service appears unreachable at this time.") from e
        finally:
            with contextlib.suppress(Exception):
                if fhandle:
                    fhandle.flush()
                    fhandle.close()
            stop_evt.set()
            if monitor_task:
                monitor_task.cancel()
                with contextlib.suppress(Exception):
                    await monitor_task
            if sess:
                with contextlib.suppress(Exception):
                    await sess.close()


async def download_url_to_image_tensor(
    url: str,
    timeout: int = None,
    auth_kwargs: Optional[dict[str, str]] = None,
    *,
    dest: Optional[Union[BytesIO, IO[bytes], str, Path]] = None,
    mode: str = "RGBA",
) -> torch.Tensor:
    """
    Download image and decode to tensor. Supports streaming `dest` like util version.
    """
    if dest is None:
        bio = await download_url_to_bytesio(url, timeout, auth_kwargs, dest=None)
        return bytesio_to_image_tensor(bio, mode=mode)  # type: ignore[arg-type]

    await download_url_to_bytesio(url, timeout, auth_kwargs, dest=dest)

    if isinstance(dest, BytesIO):
        with contextlib.suppress(Exception):
            dest.seek(0)
        return bytesio_to_image_tensor(dest, mode=mode)

    if hasattr(dest, "read") and hasattr(dest, "seek"):
        try:
            with contextlib.suppress(Exception):
                dest.flush()
            dest.seek(0)
            data = dest.read()
            return bytesio_to_image_tensor(BytesIO(data), mode=mode)
        except Exception:
            pass

    if isinstance(dest, (str, Path)) or getattr(dest, "name", None):
        path_str = str(dest if isinstance(dest, (str, Path)) else getattr(dest, "name"))
        with open(path_str, "rb") as f:
            return bytesio_to_image_tensor(BytesIO(f.read()), mode=mode)

    raise ValueError(
        "Destination is not readable and no path is available to decode the image. "
        "Pass dest=None to decode from memory, or provide a readable handle / path."
    )


def _generate_operation_id(method: str, url: str, attempt: int) -> str:
    try:
        parsed = urlparse(url)
        slug = (parsed.path.rsplit("/", 1)[-1] or parsed.netloc or "download").strip("/").replace("/", "_")
    except Exception:
        slug = "download"
    return f"{method}_{slug}_try{attempt}_{uuid.uuid4().hex[:8]}"


async def _sleep_with_cancel(seconds: float) -> None:
    """Sleep in 1s slices while checking for interruption."""
    end = time.monotonic() + seconds
    while True:
        if _is_processing_interrupted():
            raise ProcessingInterrupted("Task cancelled")
        now = time.monotonic()
        if now >= end:
            return
        await asyncio.sleep(min(1.0, end - now))
