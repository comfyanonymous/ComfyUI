"""Parser for meta format (Lora_N Model hash) metadata."""

import re
import logging
from typing import Dict, Any
from ..base import RecipeMetadataParser
from ..constants import GEN_PARAM_KEYS

logger = logging.getLogger(__name__)

class MetaFormatParser(RecipeMetadataParser):
    """Parser for images with meta format metadata (Lora_N Model hash format)"""
    
    METADATA_MARKER = r'Lora_\d+ Model hash:'
    
    def is_metadata_matching(self, user_comment: str) -> bool:
        """Check if the user comment matches the metadata format"""
        return re.search(self.METADATA_MARKER, user_comment, re.IGNORECASE | re.DOTALL) is not None
    
    async def parse_metadata(self, user_comment: str, recipe_scanner=None, civitai_client=None) -> Dict[str, Any]:
        """Parse metadata from images with meta format metadata"""
        try:
            # Extract prompt and negative prompt
            parts = user_comment.split('Negative prompt:', 1)
            prompt = parts[0].strip()
            
            # Initialize metadata
            metadata = {"prompt": prompt, "loras": []}
            
            # Extract negative prompt and parameters if available
            if len(parts) > 1:
                negative_and_params = parts[1]
                
                # Extract negative prompt - everything until the first parameter (usually "Steps:")
                param_start = re.search(r'([A-Za-z]+): ', negative_and_params)
                if param_start:
                    neg_prompt = negative_and_params[:param_start.start()].strip()
                    metadata["negative_prompt"] = neg_prompt
                    params_section = negative_and_params[param_start.start():]
                else:
                    params_section = negative_and_params
                
                # Extract key-value parameters (Steps, Sampler, Seed, etc.)
                param_pattern = r'([A-Za-z_0-9 ]+): ([^,]+)'
                params = re.findall(param_pattern, params_section)
                for key, value in params:
                    clean_key = key.strip().lower().replace(' ', '_')
                    metadata[clean_key] = value.strip()
            
            # Extract LoRA information
            # Pattern to match lora entries: Lora_0 Model name: ArtVador I.safetensors, Lora_0 Model hash: 08f7133a58, etc.
            lora_pattern = r'Lora_(\d+) Model name: ([^,]+), Lora_\1 Model hash: ([^,]+), Lora_\1 Strength model: ([^,]+), Lora_\1 Strength clip: ([^,]+)'
            lora_matches = re.findall(lora_pattern, user_comment)
            
            # If the regular pattern doesn't match, try a more flexible approach
            if not lora_matches:
                # First find all Lora indices
                lora_indices = set(re.findall(r'Lora_(\d+)', user_comment))
                
                # For each index, extract the information
                for idx in lora_indices:
                    lora_info = {}
                    
                    # Extract model name
                    name_match = re.search(f'Lora_{idx} Model name: ([^,]+)', user_comment)
                    if name_match:
                        lora_info['name'] = name_match.group(1).strip()
                    
                    # Extract model hash
                    hash_match = re.search(f'Lora_{idx} Model hash: ([^,]+)', user_comment)
                    if hash_match:
                        lora_info['hash'] = hash_match.group(1).strip()
                    
                    # Extract strength model
                    strength_model_match = re.search(f'Lora_{idx} Strength model: ([^,]+)', user_comment)
                    if strength_model_match:
                        lora_info['strength_model'] = float(strength_model_match.group(1).strip())
                    
                    # Extract strength clip
                    strength_clip_match = re.search(f'Lora_{idx} Strength clip: ([^,]+)', user_comment)
                    if strength_clip_match:
                        lora_info['strength_clip'] = float(strength_clip_match.group(1).strip())
                    
                    # Only add if we have at least name and hash
                    if 'name' in lora_info and 'hash' in lora_info:
                        lora_matches.append((idx, lora_info['name'], lora_info['hash'], 
                                            str(lora_info.get('strength_model', 1.0)), 
                                            str(lora_info.get('strength_clip', 1.0))))
            
            # Process LoRAs
            base_model_counts = {}
            loras = []
            
            for match in lora_matches:
                if len(match) == 5:  # Regular pattern match
                    idx, name, hash_value, strength_model, strength_clip = match
                else:  # Flexible approach match
                    continue  # Should not happen now
                
                # Clean up the values
                name = name.strip()
                if name.endswith('.safetensors'):
                    name = name[:-12]  # Remove .safetensors extension
                    
                hash_value = hash_value.strip()
                weight = float(strength_model)  # Use model strength as weight
                
                # Initialize lora entry with default values
                lora_entry = {
                    'name': name,
                    'type': 'lora',
                    'weight': weight,
                    'existsLocally': False,
                    'localPath': None,
                    'file_name': name,
                    'hash': hash_value,
                    'thumbnailUrl': '/loras_static/images/no-preview.png',
                    'baseModel': '',
                    'size': 0,
                    'downloadUrl': '',
                    'isDeleted': False
                }
                
                # Get info from Civitai by hash if available
                if civitai_client and hash_value:
                    try:
                        civitai_info = await civitai_client.get_model_by_hash(hash_value)
                        # Populate lora entry with Civitai info
                        populated_entry = await self.populate_lora_from_civitai(
                            lora_entry, 
                            civitai_info, 
                            recipe_scanner,
                            base_model_counts,
                            hash_value
                        )
                        if populated_entry is None:
                            continue  # Skip invalid LoRA types
                        lora_entry = populated_entry
                    except Exception as e:
                        logger.error(f"Error fetching Civitai info for LoRA hash {hash_value}: {e}")
                
                loras.append(lora_entry)
            
            # Extract model information
            model = None
            if 'model' in metadata:
                model = metadata['model']
            
            # Set base_model to the most common one from civitai_info
            base_model = None
            if base_model_counts:
                base_model = max(base_model_counts.items(), key=lambda x: x[1])[0]
            
            # Extract generation parameters for recipe metadata
            gen_params = {}
            for key in GEN_PARAM_KEYS:
                if key in metadata:
                    gen_params[key] = metadata.get(key, '')
            
            # Try to extract size information if available
            if 'width' in metadata and 'height' in metadata:
                gen_params['size'] = f"{metadata['width']}x{metadata['height']}"
            
            return {
                'base_model': base_model,
                'loras': loras,
                'gen_params': gen_params,
                'raw_metadata': metadata,
                'from_meta_format': True
            }
            
        except Exception as e:
            logger.error(f"Error parsing meta format metadata: {e}", exc_info=True)
            return {"error": str(e), "loras": []}
