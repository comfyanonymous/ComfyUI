"""comfy_api_sealed_worker — torch-free type definitions for sealed worker children.

Drop-in replacement for comfy_api.latest._util type imports in sealed workers
that do not have torch installed. Contains only data type definitions (TrimeshData,
etc.) with numpy-only dependencies.

Usage in serializers:
    if _IMPORT_TORCH:
        from comfy_api.latest._util.trimesh_types import TrimeshData
    else:
        from comfy_api_sealed_worker.trimesh_types import TrimeshData
"""

from .trimesh_types import TrimeshData

__all__ = ["TrimeshData"]
