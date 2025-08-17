"""Base classes for recipe parsers."""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from ..config import config
from ..utils.constants import VALID_LORA_TYPES

logger = logging.getLogger(__name__)

class RecipeMetadataParser(ABC):
    """Interface for parsing recipe metadata from image user comments"""

    METADATA_MARKER = None

    @abstractmethod
    def is_metadata_matching(self, user_comment: str) -> bool:
        """Check if the user comment matches the metadata format"""
        pass
    
    @abstractmethod
    async def parse_metadata(self, user_comment: str, recipe_scanner=None, civitai_client=None) -> Dict[str, Any]:
        """
        Parse metadata from user comment and return structured recipe data
        
        Args:
            user_comment: The EXIF UserComment string from the image
            recipe_scanner: Optional recipe scanner instance for local LoRA lookup
            civitai_client: Optional Civitai client for fetching model information
            
        Returns:
            Dict containing parsed recipe data with standardized format
        """
        pass
    
    async def populate_lora_from_civitai(self, lora_entry: Dict[str, Any], civitai_info_tuple: Tuple[Dict[str, Any], Optional[str]], 
                                         recipe_scanner=None, base_model_counts=None, hash_value=None) -> Optional[Dict[str, Any]]:
        """
        Populate a lora entry with information from Civitai API response
        
        Args:
            lora_entry: The lora entry to populate
            civitai_info_tuple: The response tuple from Civitai API (data, error_msg)
            recipe_scanner: Optional recipe scanner for local file lookup
            base_model_counts: Optional dict to track base model counts
            hash_value: Optional hash value to use if not available in civitai_info
            
        Returns:
            The populated lora_entry dict if type is valid, None otherwise
        """
        try:
            # Unpack the tuple to get the actual data
            civitai_info, error_msg = civitai_info_tuple if isinstance(civitai_info_tuple, tuple) else (civitai_info_tuple, None)
            
            if not civitai_info or civitai_info.get("error") == "Model not found":
                # Model not found or deleted
                lora_entry['isDeleted'] = True
                lora_entry['thumbnailUrl'] = '/loras_static/images/no-preview.png'
                return lora_entry
                
            # Get model type and validate
            model_type = civitai_info.get('model', {}).get('type', '').lower()
            lora_entry['type'] = model_type
            if model_type not in VALID_LORA_TYPES:
                logger.debug(f"Skipping non-LoRA model type: {model_type}")
                return None

            # Check if this is an early access lora
            if civitai_info.get('earlyAccessEndsAt'):
                # Convert earlyAccessEndsAt to a human-readable date
                early_access_date = civitai_info.get('earlyAccessEndsAt', '')
                lora_entry['isEarlyAccess'] = True
                lora_entry['earlyAccessEndsAt'] = early_access_date
                
            # Update model name if available
            if 'model' in civitai_info and 'name' in civitai_info['model']:
                lora_entry['name'] = civitai_info['model']['name']
            
            lora_entry['id'] = civitai_info.get('id')
            lora_entry['modelId'] = civitai_info.get('modelId')
            
            # Update version if available
            if 'name' in civitai_info:
                lora_entry['version'] = civitai_info.get('name', '')
            
            # Get thumbnail URL from first image
            if 'images' in civitai_info and civitai_info['images']:
                lora_entry['thumbnailUrl'] = civitai_info['images'][0].get('url', '')
            
            # Get base model
            current_base_model = civitai_info.get('baseModel', '')
            lora_entry['baseModel'] = current_base_model
            
            # Update base model counts if tracking them
            if base_model_counts is not None and current_base_model:
                base_model_counts[current_base_model] = base_model_counts.get(current_base_model, 0) + 1
            
            # Get download URL
            lora_entry['downloadUrl'] = civitai_info.get('downloadUrl', '')
            
            # Process file information if available
            if 'files' in civitai_info:
                # Find the primary model file (type="Model" and primary=true) in the files list
                model_file = next((file for file in civitai_info.get('files', []) 
                                    if file.get('type') == 'Model' and file.get('primary') == True), None)
                
                if model_file:
                    # Get size
                    lora_entry['size'] = model_file.get('sizeKB', 0) * 1024
                    
                    # Get SHA256 hash
                    sha256 = model_file.get('hashes', {}).get('SHA256', hash_value)
                    if sha256:
                        lora_entry['hash'] = sha256.lower()
                    
                    # Check if exists locally
                    if recipe_scanner and lora_entry['hash']:
                        lora_scanner = recipe_scanner._lora_scanner
                        exists_locally = lora_scanner.has_hash(lora_entry['hash'])
                        if exists_locally:
                            try:
                                local_path = lora_scanner.get_path_by_hash(lora_entry['hash'])
                                lora_entry['existsLocally'] = True
                                lora_entry['localPath'] = local_path
                                lora_entry['file_name'] = os.path.splitext(os.path.basename(local_path))[0]
                                
                                # Get thumbnail from local preview if available
                                lora_cache = await lora_scanner.get_cached_data()
                                lora_item = next((item for item in lora_cache.raw_data 
                                                    if item['sha256'].lower() == lora_entry['hash'].lower()), None)
                                if lora_item and 'preview_url' in lora_item:
                                    lora_entry['thumbnailUrl'] = config.get_preview_static_url(lora_item['preview_url'])
                            except Exception as e:
                                logger.error(f"Error getting local lora path: {e}")
                        else:
                            # For missing LoRAs, get file_name from model_file.name
                            file_name = model_file.get('name', '')
                            lora_entry['file_name'] = os.path.splitext(file_name)[0] if file_name else ''
                
        except Exception as e:
            logger.error(f"Error populating lora from Civitai info: {e}")
            
        return lora_entry
        
    async def populate_checkpoint_from_civitai(self, checkpoint: Dict[str, Any], civitai_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Populate checkpoint information from Civitai API response
        
        Args:
            checkpoint: The checkpoint entry to populate
            civitai_info: The response from Civitai API
            
        Returns:
            The populated checkpoint dict
        """
        try:
            if civitai_info and civitai_info.get("error") != "Model not found":
                # Update model name if available
                if 'model' in civitai_info and 'name' in civitai_info['model']:
                    checkpoint['name'] = civitai_info['model']['name']
                
                # Update version if available
                if 'name' in civitai_info:
                    checkpoint['version'] = civitai_info.get('name', '')
                
                # Get thumbnail URL from first image
                if 'images' in civitai_info and civitai_info['images']:
                    checkpoint['thumbnailUrl'] = civitai_info['images'][0].get('url', '')
                
                # Get base model
                checkpoint['baseModel'] = civitai_info.get('baseModel', '')
                
                # Get download URL
                checkpoint['downloadUrl'] = civitai_info.get('downloadUrl', '')
            else:
                # Model not found or deleted
                checkpoint['isDeleted'] = True
        except Exception as e:
            logger.error(f"Error populating checkpoint from Civitai info: {e}")
            
        return checkpoint
