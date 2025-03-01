import importlib
import inspect
from typing import Type

from easy_nodes import register_type
from easy_nodes.easy_nodes import AnythingVerifier
try:
    from pydantic import BaseModel
except ImportError:
    print("Pydantic not found. Please install it with 'pip install pydantic'")
    BaseModel = object  # Fallback if pydantic isn't installed


def registration_callback(register_these_classes: list[Type[BaseModel]]) -> None:
    for this_class in register_these_classes:
        with_its_class_name_in_all_caps: str = this_class.__qualname__.upper()
        register_type(this_class, with_its_class_name_in_all_caps, verifier=AnythingVerifier())


def register_pydantic_models(
    module_names: list[str],
) -> None:
    """
    Loads Pydantic classes from specified modules and registers them.
    
    Args:
        module_names: list of module names to search for Pydantic models
        registration_callback: Optional function to call for each model (for registration)
            If None, a dummy registration function will be used
            
    Returns:
        Side-effect: registers Pydantic models with EasyNodes.
    """
    models = []
    for module_name in module_names:
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find all Pydantic classes in the module
            for _, obj in inspect.getmembers(module):
                # Check if it's a class and a subclass of BaseModel but not BaseModel itself
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseModel) and 
                    obj is not BaseModel):
                    models.append(obj)
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
        except Exception as e:
            print(f"Error processing module {module_name}: {e}")

    # Register the model using the provided callback
    try:
        registration_callback(models)
    except Exception as e:
        print(f"{type(e)} registering models: {e}")

    return models

# Example usage:
# if __name__ == "__main__":
#     modules_to_scan = ["your_module.models", "another_module.types"]
#     models = register_pydantic_models(modules_to_scan)
#     print(f"Found {len(models)} Pydantic models")










