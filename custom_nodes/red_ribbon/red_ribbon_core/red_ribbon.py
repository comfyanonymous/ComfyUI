"""
Red Ribbon - Main entrance file for core Red Ribbon functionality
"""

from . import RedRibbonNode

class RedRibbonAPI:
    """API for accessing Red Ribbon functionality from other modules"""
    
    def __init__(self, resources, configs):
        self.configs = configs
        self.resources = resources

    def create_text_embedding(self, text):
        """Create an embedding for the given text
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embedding vector
        """
        # In a real implementation, this would use an actual embedding model
        embedding = [ord(char) for char in text]
        return embedding


# Main function that can be called when using this as a script
def main():
    print("Red Ribbon core module loaded successfully")
    print("Available tools:")
    print("- RedRibbonNode: Node for ComfyUI integration")
    print("- RedRibbonAPI: API for programmatic access")

if __name__ == "__main__":
    main()