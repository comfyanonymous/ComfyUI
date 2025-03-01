"""
Plug-in-Play Transformer - Main entrance file for transformer functionality
"""

from . import PiPTransformerNode

class TransformerAPI:
    """API for accessing Transformer functionality from other modules"""
    
    def __init__(self, resources, configs):
        self.configs = configs
        self.resources = resources

# Main function that can be called when using this as a script
def main():
    print("Plug-in-Play Transformer module loaded successfully")
    print("Available tools:")
    print("- PiPTransformerNode: Node for ComfyUI integration")
    print("- TransformerAPI: API for programmatic access")

if __name__ == "__main__":
    main()