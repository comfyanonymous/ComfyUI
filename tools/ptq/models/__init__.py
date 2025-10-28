"""
Model recipe registry for PTQ toolkit.

Recipes define model-specific quantization logic and are registered
via the @register_recipe decorator.
"""
from typing import Dict, Type
from .base import ModelRecipe


_RECIPE_REGISTRY: Dict[str, Type[ModelRecipe]] = {}


def register_recipe(recipe_cls: Type[ModelRecipe]):
    """
    Decorator to register a model recipe.

    Example:
        @register_recipe
        class FluxDevRecipe(ModelRecipe):
            @classmethod
            def name(cls):
                return "flux_dev"
            ...

    Args:
        recipe_cls: Recipe class inheriting from ModelRecipe

    Returns:
        The recipe class (unchanged)
    """
    recipe_name = recipe_cls.name()
    if recipe_name in _RECIPE_REGISTRY:
        raise ValueError(f"Recipe '{recipe_name}' is already registered")

    _RECIPE_REGISTRY[recipe_name] = recipe_cls
    return recipe_cls


def get_recipe_class(name: str) -> Type[ModelRecipe]:
    """
    Get recipe class by name.

    Args:
        name: Recipe name (e.g., 'flux_dev')

    Returns:
        Recipe class

    Raises:
        ValueError: If recipe name is not found
    """
    if name not in _RECIPE_REGISTRY:
        available = ", ".join(sorted(_RECIPE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type '{name}'. "
            f"Available recipes: {available}"
        )
    return _RECIPE_REGISTRY[name]


def list_recipes():
    """
    List all available recipe names.

    Returns:
        List of recipe names (sorted)
    """
    return sorted(_RECIPE_REGISTRY.keys())


# Import recipe modules to trigger registration
from . import flux  # noqa: F401, E402



