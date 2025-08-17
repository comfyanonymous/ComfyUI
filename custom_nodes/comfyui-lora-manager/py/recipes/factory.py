"""Factory for creating recipe metadata parsers."""

import logging
from .parsers import (
    RecipeFormatParser,
    ComfyMetadataParser,
    MetaFormatParser,
    AutomaticMetadataParser,
    CivitaiApiMetadataParser
)
from .base import RecipeMetadataParser

logger = logging.getLogger(__name__)

class RecipeParserFactory:
    """Factory for creating recipe metadata parsers"""
    
    @staticmethod
    def create_parser(metadata) -> RecipeMetadataParser:
        """
        Create appropriate parser based on the metadata content
        
        Args:
            metadata: The metadata from the image (dict or str)
            
        Returns:
            Appropriate RecipeMetadataParser implementation
        """
        # First, try CivitaiApiMetadataParser for dict input
        if isinstance(metadata, dict):
            try:
                if CivitaiApiMetadataParser().is_metadata_matching(metadata):
                    return CivitaiApiMetadataParser()
            except Exception as e:
                logger.debug(f"CivitaiApiMetadataParser check failed: {e}")
                pass
            
            # Convert dict to string for other parsers that expect string input
            try:
                import json
                metadata_str = json.dumps(metadata)
            except Exception as e:
                logger.debug(f"Failed to convert dict to JSON string: {e}")
                return None
        else:
            metadata_str = metadata
        
        # Try ComfyMetadataParser which requires valid JSON
        try:
            if ComfyMetadataParser().is_metadata_matching(metadata_str):
                return ComfyMetadataParser()
        except Exception:
            # If JSON parsing fails, move on to other parsers
            pass
        
        # Check other parsers that expect string input
        if RecipeFormatParser().is_metadata_matching(metadata_str):
            return RecipeFormatParser()
        elif AutomaticMetadataParser().is_metadata_matching(metadata_str):
            return AutomaticMetadataParser()
        elif MetaFormatParser().is_metadata_matching(metadata_str):
            return MetaFormatParser()
        else:
            return None
