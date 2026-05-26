"""Regression tests for the diffusers-format guard inside ``comfy.sd.VAE.__init__``.

The guard previously indexed ``metadata["keep_diffusers_format"]`` directly,
raising ``KeyError`` when ``metadata`` was non-``None`` but lacked that key. The
fixed guard uses ``metadata.get("keep_diffusers_format") != "true"``: a missing
key flows through to ``convert_vae_state_dict``; the explicit ``"true"`` value
bypasses it.

Five cells exercise every reachable shape of the guard input — missing key,
explicit ``"true"``, ``None``, explicit non-``"true"``, empty dict — and halt
the constructor at the first post-guard call (``model_management.is_amd``).
``_make_standin`` borrows ``__init__`` onto a bare class, mirroring
``seedvr_model_test.py::_make_standin`` (#109). ``_exercise_guard`` single-
sources the patched-constructor harness so the cells stay synchronised.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import contextlib  # noqa: E402
import unittest.mock  # noqa: E402

import comfy.sd  # noqa: E402


_DIFFUSERS_TRIGGER_KEY = "decoder.up_blocks.0.resnets.0.norm1.weight"


class _PostGuardReached(Exception):
    """Sentinel raised by the patched ``is_amd`` to halt ``__init__`` at the first post-guard statement."""


def _make_standin():
    class _StandIn:
        __init__ = comfy.sd.VAE.__init__

    return _StandIn


def _exercise_guard(metadata):
    """Drive ``VAE.__init__`` with the diffusers trigger key and the supplied
    ``metadata``; halt at ``is_amd``. Returns ``(mock_convert, mock_is_amd)``
    for branch (call_count) + reach (called) assertions per cell.
    """
    StandIn = _make_standin()
    sd = {_DIFFUSERS_TRIGGER_KEY: torch.zeros(1)}

    with unittest.mock.patch.object(
        comfy.sd.diffusers_convert,
        "convert_vae_state_dict",
        autospec=True,
        side_effect=lambda state_dict: state_dict,
    ) as mock_convert, unittest.mock.patch.object(
        comfy.sd.model_management,
        "is_amd",
        autospec=True,
        side_effect=_PostGuardReached("post-guard reached"),
    ) as mock_is_amd:
        with contextlib.suppress(_PostGuardReached):
            StandIn(sd=sd, metadata=metadata)

    return mock_convert, mock_is_amd


def test_diffusers_guard_invokes_convert_when_metadata_missing_key():
    """AC1: metadata is non-None but lacks ``keep_diffusers_format`` → convert is invoked."""
    mock_convert, mock_is_amd = _exercise_guard({"unrelated_key": "value"})

    assert mock_convert.call_count == 1
    assert mock_is_amd.called


def test_diffusers_guard_skips_convert_when_metadata_pins_keep_true():
    """AC2: metadata pins ``keep_diffusers_format == "true"`` → convert is skipped."""
    mock_convert, mock_is_amd = _exercise_guard({"keep_diffusers_format": "true"})

    assert mock_convert.call_count == 0
    assert mock_is_amd.called


def test_diffusers_guard_invokes_convert_when_metadata_is_none():
    """AC3: metadata is ``None`` → first disjunct fires, convert is invoked."""
    mock_convert, mock_is_amd = _exercise_guard(None)

    assert mock_convert.call_count == 1
    assert mock_is_amd.called


def test_diffusers_guard_invokes_convert_when_metadata_pins_keep_false():
    """AC4: metadata pins a non-``"true"`` value → second disjunct fires, convert is invoked."""
    mock_convert, mock_is_amd = _exercise_guard({"keep_diffusers_format": "false"})

    assert mock_convert.call_count == 1
    assert mock_is_amd.called


def test_diffusers_guard_invokes_convert_when_metadata_is_empty_dict():
    """AC5: metadata is ``{}`` (the ``convert_old_quants`` None→{} normalization shape) → convert is invoked."""
    mock_convert, mock_is_amd = _exercise_guard({})

    assert mock_convert.call_count == 1
    assert mock_is_amd.called
