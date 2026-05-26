"""Regression: ``comfy.ldm.seedvr.vae.causal_norm_wrapper`` 5D GroupNorm
gate at ``vae.py:509`` must compare ``memory_occupy`` against the configured
``get_norm_limit()`` accessor, not against a hardcoded ``float('inf')``.

The original code path was ``... > float('inf')`` which is unreachable at any
finite ``memory_occupy`` value, so SeedVR2's ``norm_max_mem`` setting (wired
through ``set_norm_limit``) had no effect.

This module locks in two complementary cases against any future regression,
parametrized over both ``ops.GroupNorm`` subclasses (``disable_weight_init`` and
``manual_cast``) since the production gate ``isinstance(norm_layer, ops.GroupNorm)``
matches both.

* ``test_seedvr_groupnorm_default_limit_uses_full_groupnorm_path`` — with
  the limit at its default ``inf``, the full GroupNorm forward must run and
  the chunked branch must NOT run, regardless of input tensor size.
* ``test_seedvr_groupnorm_low_limit_uses_chunked_groupnorm_path`` — with a
  deliberately low limit (``1e-9 GiB``), the chunked branch must run and
  the full GroupNorm forward must NOT run.

Each case discriminates the two branches with two independent observers:

1. ``nn.Module.register_forward_hook`` on the GroupNorm — fires only on the
   full-path branch ``norm_layer(x)``; the chunked branch bypasses the
   module ``__call__`` and goes through ``F.group_norm`` directly.
2. ``unittest.mock.patch.object(vae.F, 'group_norm', ...)`` spy with
   ``side_effect`` delegating to the real ``torch.nn.functional.group_norm``
   — captures every direct ``F.group_norm`` call's ``num_groups`` argument.
   Calls with ``num_groups < gn.num_groups`` come from the chunked branch
   (``num_groups_per_chunk = gn.num_groups // num_chunks``).

The spy uses ``*args, **kwargs`` passthrough so future ``F.group_norm`` kwargs
do not break the test.

CPU-only by construction: the tests use a small float32 tensor and never
allocate a real model or GPU memory.
"""

from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops as comfy_ops  # noqa: E402
import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
from comfy.ldm.seedvr.vae import (  # noqa: E402
    causal_norm_wrapper,
    set_norm_limit,
)


_NUM_CHANNELS = 8
_NUM_GROUPS = 4
_TENSOR_SHAPE = (1, 8, 2, 4, 4)

# Both ``ops.GroupNorm`` subclasses appear in production paths depending on
# the active backend. The dispatch gate at ``vae.py:509`` reads
# ``isinstance(norm_layer, ops.GroupNorm)`` and matches both via MRO.
_GROUPNORM_SUBCLASSES = [
    pytest.param(
        comfy_ops.disable_weight_init.GroupNorm,
        id="disable_weight_init",
    ),
    pytest.param(
        comfy_ops.manual_cast.GroupNorm,
        id="manual_cast",
    ),
]


@pytest.mark.parametrize("groupnorm_cls", _GROUPNORM_SUBCLASSES)
def test_seedvr_groupnorm_default_limit_uses_full_groupnorm_path(groupnorm_cls):
    real_group_norm = vae_mod.F.group_norm
    set_norm_limit(None)
    try:
        gn = groupnorm_cls(
            num_channels=_NUM_CHANNELS, num_groups=_NUM_GROUPS
        )
        gn.eval()

        forward_hook_calls = []

        def _hook(module, inputs, output):
            forward_hook_calls.append(tuple(inputs[0].shape))

        spy_calls = []

        def _group_norm_spy(input_tensor, num_groups_arg, *args, **kwargs):
            spy_calls.append({
                "num_groups": int(num_groups_arg),
                "input_shape": tuple(int(s) for s in input_tensor.shape),
            })
            return real_group_norm(input_tensor, num_groups_arg, *args, **kwargs)

        handle = gn.register_forward_hook(_hook)
        try:
            with patch.object(vae_mod.F, "group_norm", side_effect=_group_norm_spy):
                out_tensor = causal_norm_wrapper(gn, torch.randn(*_TENSOR_SHAPE))
        finally:
            handle.remove()

        full_calls = len(forward_hook_calls)
        chunked_calls = sum(
            1 for entry in spy_calls if entry["num_groups"] < _NUM_GROUPS
        )

        assert tuple(int(s) for s in out_tensor.shape) == _TENSOR_SHAPE, (
            f"causal_norm_wrapper output shape {tuple(out_tensor.shape)} does not "
            f"match input shape {_TENSOR_SHAPE}"
        )
        assert full_calls == 1, (
            f"default-limit (inf) GroupNorm gate must take the full-forward path "
            f"(register_forward_hook fires exactly once); got full_calls={full_calls}"
        )
        assert chunked_calls == 0, (
            f"default-limit (inf) GroupNorm gate must NOT take the chunked path "
            f"(no F.group_norm call with num_groups<{_NUM_GROUPS}); got "
            f"chunked_calls={chunked_calls}"
        )
    finally:
        set_norm_limit(None)


@pytest.mark.parametrize("groupnorm_cls", _GROUPNORM_SUBCLASSES)
def test_seedvr_groupnorm_low_limit_uses_chunked_groupnorm_path(groupnorm_cls):
    real_group_norm = vae_mod.F.group_norm
    set_norm_limit(1e-9)
    try:
        gn = groupnorm_cls(
            num_channels=_NUM_CHANNELS, num_groups=_NUM_GROUPS
        )
        gn.eval()

        forward_hook_calls = []

        def _hook(module, inputs, output):
            forward_hook_calls.append(tuple(inputs[0].shape))

        spy_calls = []

        def _group_norm_spy(input_tensor, num_groups_arg, *args, **kwargs):
            spy_calls.append({
                "num_groups": int(num_groups_arg),
                "input_shape": tuple(int(s) for s in input_tensor.shape),
            })
            return real_group_norm(input_tensor, num_groups_arg, *args, **kwargs)

        handle = gn.register_forward_hook(_hook)
        try:
            with patch.object(vae_mod.F, "group_norm", side_effect=_group_norm_spy):
                out_tensor = causal_norm_wrapper(gn, torch.randn(*_TENSOR_SHAPE))
        finally:
            handle.remove()

        full_calls = len(forward_hook_calls)
        chunked_calls = sum(
            1 for entry in spy_calls if entry["num_groups"] < _NUM_GROUPS
        )

        assert tuple(int(s) for s in out_tensor.shape) == _TENSOR_SHAPE, (
            f"causal_norm_wrapper output shape {tuple(out_tensor.shape)} does not "
            f"match input shape {_TENSOR_SHAPE}"
        )
        assert full_calls == 0, (
            f"low-limit GroupNorm gate must NOT take the full-forward path "
            f"(register_forward_hook should not fire); got full_calls={full_calls}"
        )
        assert chunked_calls > 0, (
            f"low-limit GroupNorm gate must take the chunked path "
            f"(at least one F.group_norm call with num_groups<{_NUM_GROUPS}); got "
            f"chunked_calls={chunked_calls}"
        )
    finally:
        set_norm_limit(None)
