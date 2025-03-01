"""
Social Toolkit Module for Red Ribbon
"""

class SocialToolkitNode:
    """Node for social media integration tools"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "platform": (["twitter", "instagram", "facebook"], {"default": "twitter"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "Red Ribbon/Social"
    
    def process(self, text, platform):
        # Process the text for social media
        return (f"[{platform.upper()}]: {text}",)

# Dictionary of nodes to be imported by main.py
NODE_CLASS_MAPPINGS = {
    "SocialToolkitNode": SocialToolkitNode
}

# Add display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SocialToolkitNode": "Social Media Toolkit"
}

# Function to be called from main.py
def socialtoolkit():
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS