"""
Utils - Main entrance file for Red Ribbon utility functions
"""

from . import UtilityNode, merge_node_mappings

class UtilsAPI:
    """API for accessing utility functionality from other modules"""
    
    @staticmethod
    def log_message(mode, message):
        """Log a message with the specified mode
        
        Args:
            mode (str): Log mode (debug, info, log)
            message (str): The message to log
            
        Returns:
            str: Formatted log message
        """
        formatted = f"[{mode.upper()}] {message}"
        print(formatted)
        return formatted
    
    @staticmethod
    def combine_mappings(mappings_list):
        """Wrapper for merge_node_mappings function"""
        return merge_node_mappings(mappings_list)

# Main function that can be called when using this as a script
def main():
    print("Red Ribbon Utils module loaded successfully")
    print("Available tools:")
    print("- UtilityNode: Node for ComfyUI integration")
    print("- UtilsAPI: API for programmatic access")
    print("- merge_node_mappings: Function for merging node mappings")

if __name__ == "__main__":
    main()