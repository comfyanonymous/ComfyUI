"""
Utility functions for Red Ribbon custom nodes
"""

def merge_node_mappings(mappings_list):
    """
    Merge multiple node class mappings into one
    
    Args:
        mappings_list: List of (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS) tuples
        
    Returns:
        tuple: Combined (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
    """
    combined_class_mappings = {}
    combined_display_mappings = {}
    
    for class_mapping, display_mapping in mappings_list:
        combined_class_mappings.update(class_mapping)
        if display_mapping:
            combined_display_mappings.update(display_mapping)
    
    return combined_class_mappings, combined_display_mappings

class UtilityNode:
    """Node with utility functions for Red Ribbon"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["debug", "info", "log"], {"default": "info"}),
                "message": ("STRING", {"multiline": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "log"
    CATEGORY = "Red Ribbon/Utils"
    
    def log(self, mode, message):
        # Log the message according to the specified mode
        formatted = f"[{mode.upper()}] {message}"
        print(formatted)
        return (formatted,)

# Dictionary of nodes to be imported by main.py
NODE_CLASS_MAPPINGS = {
    "UtilityNode": UtilityNode
}

# Add display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UtilityNode": "Red Ribbon Utility"
}

def utils():
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS