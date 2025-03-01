"""
Red Ribbon - A collection of custom nodes for ComfyUI
"""

import easy_nodes
import os

# Version information
__version__ = "0.1.0"

# NOTE This only needs to be called once.
easy_nodes.initialize_easy_nodes(default_category="Red Ribbon")

# Import all modules - this must come after calling initialize_easy_nodes
from . import main

# Get the combined node mappings for ComfyUI
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()

# Export so that ComfyUI can pick them up.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Optional: export the node list to a file so that e.g. ComfyUI-Manager can pick it up.
easy_nodes.save_node_list(
    os.path.join(os.path.dirname(__file__), "red_ribbon_node_list.json")
)

print(f"Red Ribbon v{__version__}: Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes")