"""
Red Ribbon - Main entrance file for the entire Red Ribbon package
"""

# Import components from subdirectories
from .socialtoolkit.socialtoolkit import SocialToolkitAPI
from .red_ribbon_core.red_ribbon import RedRibbonAPI
from .plug_in_play_transformer.plug_in_play_transformer import TransformerAPI
from .utils.utils import UtilsAPI


from .main import package



# Main function that can be called when using this as a script
def main():
    print("Red Ribbon package loaded successfully")
    print(f"Version: {package.version()}")
    print("Available components:")
    print("- SocialToolkit")
    print("- RedRibbon Core")
    print("- Plug-in-Play Transformer")
    print("- Utils")
    while True:
        choice_was = input("Enter your choice: ")
        match choice_was:
            case "SocialToolkit":
                print("SocialToolkitAPI")
            case "RedRibbon Core":
                print("RedRibbonAPI")
            case "Plug-in-Play Transformer":
                print("TransformerAPI")
            case "Utils":
                print("UtilsAPI")
            case _:
                print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()