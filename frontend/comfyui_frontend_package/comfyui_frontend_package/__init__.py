from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("comfyui_frontend_package")
except PackageNotFoundError:
    __version__ = "unknown"
