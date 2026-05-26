"""Regression test: SeedVR2 resize schema input ids must match
execute() positional parameter order. Drift between the two would silently
swap arguments at runtime; this test fails loudly on any future drift.

The schema input attribute is `.id` (verified live via Python introspection
on the upstream class -- there is no `.name`).

`comfy.model_management` is stubbed via `patch.dict(sys.modules, ...)` for
the import performed inside this test, so importing
`comfy_extras.nodes_seedvr` here does not call
`torch.cuda.is_available()` or trigger other GPU/server-side
initialization through that dependency. Live introspection indicated that
`comfy_extras.nodes_seedvr` pulls in `comfy.model_management`
transitively here (not `nodes`, not `server`).

The test snapshots three pieces of import state before patching and
restores all three in `finally` via a sentinel:

1. `sys.modules["comfy_extras.nodes_seedvr"]`
2. `comfy.model_management` package attribute on the `comfy` package
3. `comfy_extras.nodes_seedvr` attribute on the `comfy_extras` package

If any of the three was set before the test, it is restored verbatim;
if it was unset, it is deleted on exit. This prevents the test from
clobbering a real `comfy.model_management` (or
`comfy_extras.nodes_seedvr`) module that another test may have
legitimately imported earlier in the same pytest process, while still
preventing the test's mock from leaking into later tests that import
the real `comfy_extras.nodes_seedvr`."""

import importlib
import inspect
import sys
from unittest.mock import MagicMock, patch

from comfy.cli_args import args as cli_args


def test_seedvr_node_signature_matches_schema():
    mock_model_management = MagicMock()
    mock_model_management.xformers_enabled.return_value = False
    mock_model_management.xformers_enabled_vae.return_value = False
    mock_model_management.sage_attention_enabled.return_value = False
    mock_model_management.flash_attention_enabled.return_value = False
    sentinel = object()
    prior_cpu = cli_args.cpu
    cli_args.cpu = True

    comfy_module_pre = sys.modules.get("comfy")
    comfy_extras_module_pre = sys.modules.get("comfy_extras")
    prior_comfy_mm_attr = (
        getattr(comfy_module_pre, "model_management", sentinel)
        if comfy_module_pre is not None
        else sentinel
    )
    prior_comfy_extras_seedvr_attr = (
        getattr(comfy_extras_module_pre, "nodes_seedvr", sentinel)
        if comfy_extras_module_pre is not None
        else sentinel
    )
    prior_comfy_extras_seedvr_module = sys.modules.get("comfy_extras.nodes_seedvr", sentinel)

    with patch.dict(sys.modules, {"comfy.model_management": mock_model_management}):
        if comfy_module_pre is not None:
            setattr(comfy_module_pre, "model_management", mock_model_management)
        sys.modules.pop("comfy_extras.nodes_seedvr", None)
        try:
            nodes_seedvr = importlib.import_module("comfy_extras.nodes_seedvr")
            for node_cls in (
                nodes_seedvr.SeedVR2Resize,
                nodes_seedvr.SeedVR2ResizeAdvanced,
            ):
                schema_ids = [i.id for i in node_cls.define_schema().inputs]
                exec_params = [
                    p
                    for p in inspect.signature(node_cls.execute).parameters.keys()
                    if p != "cls"
                ]
                assert schema_ids == exec_params, (
                    f"{node_cls.__name__} schema input ids do not match "
                    f"execute() parameter order: schema_ids={schema_ids}, "
                    f"exec_params={exec_params}"
                )
        finally:
            if prior_comfy_extras_seedvr_module is sentinel:
                sys.modules.pop("comfy_extras.nodes_seedvr", None)
            else:
                sys.modules["comfy_extras.nodes_seedvr"] = prior_comfy_extras_seedvr_module
            cli_args.cpu = prior_cpu
            comfy_extras_module = sys.modules.get("comfy_extras")
            if comfy_extras_module is not None:
                if prior_comfy_extras_seedvr_attr is sentinel:
                    if hasattr(comfy_extras_module, "nodes_seedvr"):
                        delattr(comfy_extras_module, "nodes_seedvr")
                else:
                    setattr(comfy_extras_module, "nodes_seedvr", prior_comfy_extras_seedvr_attr)
            comfy_module = sys.modules.get("comfy")
            if comfy_module is not None:
                if prior_comfy_mm_attr is sentinel:
                    if hasattr(comfy_module, "model_management"):
                        delattr(comfy_module, "model_management")
                else:
                    setattr(comfy_module, "model_management", prior_comfy_mm_attr)
