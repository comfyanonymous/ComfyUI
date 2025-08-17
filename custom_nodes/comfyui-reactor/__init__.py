import sys
import os

repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)
original_modules = sys.modules.copy()

# Place aside existing modules if using a1111 web ui
modules_used = [
    "modules",
    "modules.images",
    "modules.processing",
    "modules.scripts_postprocessing",
    "modules.scripts",
    "modules.shared",
]
original_webui_modules = {}
for module in modules_used:
    if module in sys.modules:
        original_webui_modules[module] = sys.modules.pop(module)

# Proceed with node setup
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Clean up imports
# Remove repo directory from path
sys.path.remove(repo_dir)
# Remove any new modules
modules_to_remove = []
for module in sys.modules:
    if module not in original_modules and not module.startswith("google.protobuf") and not module.startswith("onnx") and not module.startswith("cv2"):
        modules_to_remove.append(module)
for module in modules_to_remove:
    del sys.modules[module]

# Restore original modules
sys.modules.update(original_webui_modules)
