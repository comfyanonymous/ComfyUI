"""Loader shim so ComfyUI picks up the LoD attention node.

The implementation lives in ``lodx_dit/`` at the repo root, outside
``custom_nodes``, so that the prototype stays one self-contained package that
can also be imported by its own tests and benchmarks without a running server.
"""

from lodx_dit.comfy_node import comfy_entrypoint  # noqa: F401
