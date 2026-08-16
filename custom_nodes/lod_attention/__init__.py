"""Loader shim so ComfyUI picks up the LoD attention node.

The implementation lives in ``lodx_dit/`` at the repo root, outside
``custom_nodes``, so that the prototype stays one self-contained package that
can also be imported by its own tests and benchmarks without a running server.

``--dit-gpus`` is applied here rather than from ``main.py`` so the only change
to core is the argparse line that defines it.
"""

import logging

from comfy.cli_args import args

from lodx_dit.comfy_node import comfy_entrypoint  # noqa: F401

if getattr(args, "dit_gpus", 1) > 1:
    import comfy.memory_management

    if comfy.memory_management.aimdo_enabled:
        logging.warning(
            "[LoD-PP] --dit-gpus is not compatible with DynamicVRAM: the "
            "patcher streams weights to a single load_device and would undo "
            "the split every op. Add --disable-dynamic-vram to use it.")
    else:
        from lodx_dit.pipeline import install
        install(args.dit_gpus)
