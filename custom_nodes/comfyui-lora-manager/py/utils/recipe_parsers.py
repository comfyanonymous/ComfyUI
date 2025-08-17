"""
Legacy recipe_parsers module that redirects to the new recipes package.

This file is kept for backwards compatibility and now imports the refactored modules.
"""

import logging
import warnings

# Show deprecation warning
warnings.warn(
    "The module 'py.utils.recipe_parsers' is deprecated. Use 'py.recipes' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location
from ..recipes.constants import GEN_PARAM_KEYS, VALID_LORA_TYPES
from ..recipes.base import RecipeMetadataParser
from ..recipes.parsers import (
    RecipeFormatParser,
    ComfyMetadataParser,
    MetaFormatParser,
    AutomaticMetadataParser
)
from ..recipes.factory import RecipeParserFactory

# Redirect all imports
__all__ = [
    'GEN_PARAM_KEYS',
    'VALID_LORA_TYPES',
    'RecipeMetadataParser',
    'RecipeFormatParser',
    'ComfyMetadataParser',
    'MetaFormatParser',
    'AutomaticMetadataParser',
    'RecipeParserFactory'
]
