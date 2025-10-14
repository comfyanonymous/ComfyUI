import asyncio
import contextlib
import time
from typing import Optional, Callable

from comfy_api.latest import IO
from comfy.cli_args import args
from comfy.model_management import processing_interrupted

from .common_exceptions import ProcessingInterrupted


def _is_processing_interrupted() -> bool:
    """Return True if user/runtime requested interruption."""
    return processing_interrupted()


def _get_node_id(node_cls: type[IO.ComfyNode]) -> str:
    return node_cls.hidden.unique_id


def _get_auth_header(node_cls: type[IO.ComfyNode]) -> dict[str, str]:
    if node_cls.hidden.auth_token_comfy_org:
        return {"Authorization": f"Bearer {node_cls.hidden.auth_token_comfy_org}"}
    if node_cls.hidden.api_key_comfy_org:
        return {"X-API-KEY": node_cls.hidden.api_key_comfy_org}
    return {}


def _default_base_url() -> str:
    return getattr(args, "comfy_api_base", "https://api.comfy.org")


async def _sleep_with_interrupt(
    seconds: float,
    node_cls: type[IO.ComfyNode],
    label: Optional[str] = None,
    start_ts: Optional[float] = None,
    estimated_total: Optional[int] = None,
    *,
    display_callback: Optional[Callable[[type[IO.ComfyNode], str, int, Optional[int]], None]] = None,
):
    """
    Sleep in 1s slices while:
      - Checking for interruption (raises ProcessingInterrupted).
      - Optionally emitting time progress via display_callback (if provided).
    """
    end = time.monotonic() + seconds
    while True:
        if _is_processing_interrupted():
            raise ProcessingInterrupted("Task cancelled")
        now = time.monotonic()
        if start_ts is not None and label and display_callback:
            with contextlib.suppress(Exception):
                display_callback(node_cls, label, int(now - start_ts), estimated_total)
        if now >= end:
            break
        await asyncio.sleep(min(1.0, end - now))
