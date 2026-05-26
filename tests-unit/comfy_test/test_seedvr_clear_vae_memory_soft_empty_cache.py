"""Regression test for ``comfy_extras.nodes_seedvr.clear_vae_memory`` —
must dispatch its cache clear via ``comfy.model_management.soft_empty_cache``
rather than calling ``torch.cuda.empty_cache()`` directly. The canonical helper
at ``comfy/model_management.py:1780`` short-circuits via ``cpu_mode()`` and
dispatches per-backend (MPS / XPU / NPU / MLU / CUDA), so it is the only
correct call shape on non-CUDA hosts and on managed-device hosts where
``comfy.cli_args.args.cpu`` is True.
"""

from unittest.mock import patch

import torch

# CPU-only CI fix: ``comfy_extras.nodes_seedvr`` transitively imports
# ``comfy.model_management``, whose module-level
# ``cpu_state = CPUState.CPU if args.cpu`` initialiser
# (``comfy/model_management.py:152-153``) reads ``comfy.cli_args.args.cpu``
# at import time. Match the pattern at
# ``tests-unit/comfy_test/test_seedvr_vae_decode_unpadded_t.py:33-44``: flip
# ``args.cpu`` BEFORE importing any ``comfy.ldm.*`` or ``comfy_extras.*``
# symbol. This module forces ``args.cpu = True`` unconditionally (rather
# than only when ``torch.cuda.is_available()`` is False) so ``cpu_mode()``
# returns True at call time regardless of host CUDA availability — the
# path under test is ``soft_empty_cache``'s CPU-mode short-circuit at
# ``comfy/model_management.py:1781``.
from comfy.cli_args import args as _cli_args

_cli_args.cpu = True

import comfy.model_management  # noqa: E402
import comfy_extras.nodes_seedvr as nodes_seedvr  # noqa: E402


def test_clear_vae_memory_uses_soft_empty_cache():
    """``clear_vae_memory(stub)`` must invoke
    ``comfy.model_management.soft_empty_cache`` exactly once and
    ``torch.cuda.empty_cache`` zero times when ``args.cpu`` is True.
    """
    stub = torch.nn.Module()

    with patch.object(
        comfy.model_management, "soft_empty_cache"
    ) as soft_empty_spy, patch.object(
        torch.cuda, "empty_cache"
    ) as cuda_empty_spy:
        nodes_seedvr.clear_vae_memory(stub)

    assert cuda_empty_spy.call_count == 0, (
        f"torch.cuda.empty_cache was called {cuda_empty_spy.call_count} "
        f"times; expected 0. clear_vae_memory must dispatch via "
        f"comfy.model_management.soft_empty_cache, which short-circuits in "
        f"CPU mode (cpu_mode() check at comfy/model_management.py:1781). "
        f"The unguarded torch.cuda.empty_cache() call at "
        f"comfy_extras/nodes_seedvr.py:84 is the regression this test locks."
    )
    assert soft_empty_spy.call_count == 1, (
        f"comfy.model_management.soft_empty_cache was called "
        f"{soft_empty_spy.call_count} times; expected exactly 1. "
        f"clear_vae_memory must dispatch its cache clear via the canonical "
        f"per-backend helper at comfy/model_management.py:1780."
    )
