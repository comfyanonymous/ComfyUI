# There are some circumstances where comfy/__init__.py may already exist, so let's patch it here too
from comfy_compatibility.workspace import auto_patch_workspace_and_restart

auto_patch_workspace_and_restart()
