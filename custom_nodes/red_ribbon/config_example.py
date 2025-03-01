#!/usr/bin/env python
# Example of using the Configs class

from configs import Configs

def main():
    # Get the configuration instance
    config = Configs.get()
    
    # Access configuration values as read-only properties
    print("API URL:", config.API_URL)
    print("Debug Mode:", config.DEBUG_MODE)
    print("Max Batch Size:", config.MAX_BATCH_SIZE)
    
    # Access nested dictionary values
    print("\nModel Paths:")
    for model_type, path in config.MODEL_PATHS.items():
        print(f"  {model_type}: {path}")
    
    print("\nCustom Settings:")
    for key, value in config.CUSTOM_SETTINGS.items():
        print(f"  {key}: {value}")
    
    # Attempt to modify a value (will raise an error due to frozen=True)
    try:
        config.API_URL = "http://new-url.com"
    except Exception as e:
        print(f"\nAttempted to modify API_URL and got: {type(e).__name__}: {e}")
    
    # Configuration stays unchanged
    print("\nAPI URL is still:", config.API_URL)

if __name__ == "__main__":
    main()