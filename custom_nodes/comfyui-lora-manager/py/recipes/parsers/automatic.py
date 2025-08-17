"""Parser for Automatic1111 metadata format."""

import re
import json
import logging
from typing import Dict, Any
from ..base import RecipeMetadataParser
from ..constants import GEN_PARAM_KEYS

logger = logging.getLogger(__name__)

class AutomaticMetadataParser(RecipeMetadataParser):
    """Parser for Automatic1111 metadata format"""
    
    METADATA_MARKER = r"Steps: \d+"
    
    # Regular expressions for extracting specific metadata
    HASHES_REGEX = r', Hashes:\s*({[^}]+})'
    LORA_HASHES_REGEX = r', Lora hashes:\s*"([^"]+)"'
    CIVITAI_RESOURCES_REGEX = r', Civitai resources:\s*(\[\{.*?\}\])'
    CIVITAI_METADATA_REGEX = r', Civitai metadata:\s*(\{.*?\})'
    EXTRANETS_REGEX = r'<(lora|hypernet):([^:]+):(-?[0-9.]+)>'
    MODEL_HASH_PATTERN = r'Model hash: ([a-zA-Z0-9]+)'
    VAE_HASH_PATTERN = r'VAE hash: ([a-zA-Z0-9]+)'
    
    def is_metadata_matching(self, user_comment: str) -> bool:
        """Check if the user comment matches the Automatic1111 format"""
        return re.search(self.METADATA_MARKER, user_comment) is not None
    
    async def parse_metadata(self, user_comment: str, recipe_scanner=None, civitai_client=None) -> Dict[str, Any]:
        """Parse metadata from Automatic1111 format"""
        try:
            # Split on Negative prompt if it exists
            if "Negative prompt:" in user_comment:
                parts = user_comment.split('Negative prompt:', 1)
                prompt = parts[0].strip()
                negative_and_params = parts[1] if len(parts) > 1 else ""
            else:
                # No negative prompt section
                param_start = re.search(self.METADATA_MARKER, user_comment)
                if param_start:
                    prompt = user_comment[:param_start.start()].strip()
                    negative_and_params = user_comment[param_start.start():]
                else:
                    prompt = user_comment.strip()
                    negative_and_params = ""
            
            # Initialize metadata
            metadata = {
                "prompt": prompt,
                "loras": []
            }
            
            # Extract negative prompt and parameters
            if negative_and_params:
                # If we split on "Negative prompt:", check for params section
                if "Negative prompt:" in user_comment:
                    param_start = re.search(r'Steps: ', negative_and_params)
                    if param_start:
                        neg_prompt = negative_and_params[:param_start.start()].strip()
                        metadata["negative_prompt"] = neg_prompt
                        params_section = negative_and_params[param_start.start():]
                    else:
                        metadata["negative_prompt"] = negative_and_params.strip()
                        params_section = ""
                else:
                    # No negative prompt, entire section is params
                    params_section = negative_and_params
                
                # Extract generation parameters
                if params_section:
                    # Extract Civitai resources
                    civitai_resources_match = re.search(self.CIVITAI_RESOURCES_REGEX, params_section)
                    if civitai_resources_match:
                        try:
                            civitai_resources = json.loads(civitai_resources_match.group(1))
                            metadata["civitai_resources"] = civitai_resources
                            params_section = params_section.replace(civitai_resources_match.group(0), '')
                        except json.JSONDecodeError:
                            logger.error("Error parsing Civitai resources JSON")
                    
                    # Extract Hashes
                    hashes_match = re.search(self.HASHES_REGEX, params_section)
                    if hashes_match:
                        try:
                            hashes = json.loads(hashes_match.group(1))
                            # Process hash keys
                            processed_hashes = {}
                            for key, value in hashes.items():
                                # Convert Model: or LORA: prefix to lowercase if present
                                if ':' in key:
                                    prefix, name = key.split(':', 1)
                                    prefix = prefix.lower()
                                else:
                                    prefix = ''
                                    name = key

                                # Clean up the name part
                                if '/' in name:
                                    name = name.split('/')[-1]  # Get last part after /
                                if '.safetensors' in name:
                                    name = name.split('.safetensors')[0]  # Remove .safetensors
                                
                                # Reconstruct the key
                                new_key = f"{prefix}:{name}" if prefix else name
                                processed_hashes[new_key] = value

                            metadata["hashes"] = processed_hashes
                            # Remove hashes from params section to not interfere with other parsing
                            params_section = params_section.replace(hashes_match.group(0), '')
                        except json.JSONDecodeError:
                            logger.error("Error parsing hashes JSON")
                    
                    # Extract Lora hashes in alternative format
                    lora_hashes_match = re.search(self.LORA_HASHES_REGEX, params_section)
                    if not hashes_match and lora_hashes_match:
                        try:
                            lora_hashes_str = lora_hashes_match.group(1)
                            lora_hash_entries = lora_hashes_str.split(', ')
                            
                            # Initialize hashes dict if it doesn't exist
                            if "hashes" not in metadata:
                                metadata["hashes"] = {}
                                
                            # Parse each lora hash entry (format: "name: hash")
                            for entry in lora_hash_entries:
                                if ': ' in entry:
                                    lora_name, lora_hash = entry.split(': ', 1)
                                    # Add as lora type in the same format as regular hashes
                                    metadata["hashes"][f"lora:{lora_name}"] = lora_hash.strip()
                            
                            # Remove lora hashes from params section
                            params_section = params_section.replace(lora_hashes_match.group(0), '')
                        except Exception as e:
                            logger.error(f"Error parsing Lora hashes: {e}")
                    
                    # Extract basic parameters
                    param_pattern = r'([A-Za-z\s]+): ([^,]+)'
                    params = re.findall(param_pattern, params_section)
                    gen_params = {}
                    
                    for key, value in params:
                        clean_key = key.strip().lower().replace(' ', '_')
                        
                        # Skip if not in recognized gen param keys
                        if clean_key not in GEN_PARAM_KEYS:
                            continue
                            
                        # Convert numeric values
                        if clean_key in ['steps', 'seed']:
                            try:
                                gen_params[clean_key] = int(value.strip())
                            except ValueError:
                                gen_params[clean_key] = value.strip()
                        elif clean_key in ['cfg_scale']:
                            try:
                                gen_params[clean_key] = float(value.strip())
                            except ValueError:
                                gen_params[clean_key] = value.strip()
                        else:
                            gen_params[clean_key] = value.strip()
                    
                    # Extract size if available and add to gen_params if a recognized key
                    size_match = re.search(r'Size: (\d+)x(\d+)', params_section)
                    if size_match and 'size' in GEN_PARAM_KEYS:
                        width, height = size_match.groups()
                        gen_params['size'] = f"{width}x{height}"
                    
                    # Add prompt and negative_prompt to gen_params if they're in GEN_PARAM_KEYS
                    if 'prompt' in GEN_PARAM_KEYS and 'prompt' in metadata:
                        gen_params['prompt'] = metadata['prompt']
                    if 'negative_prompt' in GEN_PARAM_KEYS and 'negative_prompt' in metadata:
                        gen_params['negative_prompt'] = metadata['negative_prompt']
                    
                    metadata["gen_params"] = gen_params
            
            # Extract LoRA information 
            loras = []
            base_model_counts = {}
            
            # First use Civitai resources if available (more reliable source)
            if metadata.get("civitai_resources"):
                for resource in metadata.get("civitai_resources", []):
                    # --- Added: Parse 'air' field if present ---
                    air = resource.get("air")
                    if air:
                        # Format: urn:air:sdxl:lora:civitai:1221007@1375651
                        # Or: urn:air:sdxl:checkpoint:civitai:623891@2019115
                        air_pattern = r"urn:air:[^:]+:(?P<type>[^:]+):civitai:(?P<modelId>\d+)@(?P<modelVersionId>\d+)"
                        air_match = re.match(air_pattern, air)
                        if air_match:
                            air_type = air_match.group("type")
                            air_modelId = int(air_match.group("modelId"))
                            air_modelVersionId = int(air_match.group("modelVersionId"))
                            # checkpoint/lycoris/lora/hypernet
                            resource["type"] = air_type
                            resource["modelId"] = air_modelId
                            resource["modelVersionId"] = air_modelVersionId
                    # --- End added ---

                    if resource.get("type") in ["lora", "lycoris", "hypernet"] and resource.get("modelVersionId"):
                        # Initialize lora entry
                        lora_entry = {
                            'id': resource.get("modelVersionId", 0),
                            'modelId': resource.get("modelId", 0),
                            'name': resource.get("modelName", "Unknown LoRA"),
                            'version': resource.get("modelVersionName", resource.get("versionName", "")),
                            'type': resource.get("type", "lora"),
                            'weight': round(float(resource.get("weight", 1.0)), 2),
                            'existsLocally': False,
                            'thumbnailUrl': '/loras_static/images/no-preview.png',
                            'baseModel': '',
                            'size': 0,
                            'downloadUrl': '',
                            'isDeleted': False
                        }
                        
                        # Get additional info from Civitai
                        if civitai_client:
                            try:
                                civitai_info = await civitai_client.get_model_version_info(resource.get("modelVersionId"))
                                populated_entry = await self.populate_lora_from_civitai(
                                    lora_entry,
                                    civitai_info,
                                    recipe_scanner,
                                    base_model_counts
                                )
                                if populated_entry is None:
                                    continue  # Skip invalid LoRA types
                                lora_entry = populated_entry
                            except Exception as e:
                                logger.error(f"Error fetching Civitai info for LoRA {lora_entry['name']}: {e}")
                        
                        loras.append(lora_entry)
            
            # If no LoRAs from Civitai resources or to supplement, extract from metadata["hashes"]
            if not loras or len(loras) == 0:
                # Extract lora weights from extranet tags in prompt (for later use)
                lora_weights = {}
                lora_matches = re.findall(self.EXTRANETS_REGEX, prompt)
                for lora_type, lora_name, lora_weight in lora_matches:
                    key = f"{lora_type}:{lora_name}"
                    lora_weights[key] = round(float(lora_weight), 2)
                
                # Use hashes from metadata as the primary source
                if metadata.get("hashes"):
                    for hash_key, lora_hash in metadata.get("hashes", {}).items():
                        # Only process lora or hypernet types
                        if not hash_key.startswith(("lora:", "hypernet:")):
                            continue
                            
                        lora_type, lora_name = hash_key.split(':', 1)
                        
                        # Get weight from extranet tags if available, else default to 1.0
                        weight = lora_weights.get(hash_key, 1.0)
                        
                        # Initialize lora entry
                        lora_entry = {
                            'name': lora_name,
                            'type': lora_type,  # 'lora' or 'hypernet'
                            'weight': weight,
                            'hash': lora_hash,
                            'existsLocally': False,
                            'localPath': None,
                            'file_name': lora_name,
                            'thumbnailUrl': '/loras_static/images/no-preview.png',
                            'baseModel': '',
                            'size': 0,
                            'downloadUrl': '',
                            'isDeleted': False
                        }
                        
                        # Try to get info from Civitai
                        if civitai_client:
                            try:
                                if lora_hash:
                                    # If we have hash, use it for lookup
                                    civitai_info = await civitai_client.get_model_by_hash(lora_hash)
                                else:
                                    civitai_info = None
                                
                                populated_entry = await self.populate_lora_from_civitai(
                                    lora_entry, 
                                    civitai_info, 
                                    recipe_scanner,
                                    base_model_counts,
                                    lora_hash
                                )
                                if populated_entry is None:
                                    continue  # Skip invalid LoRA types
                                lora_entry = populated_entry
                            except Exception as e:
                                logger.error(f"Error fetching Civitai info for LoRA {lora_name}: {e}")
                        
                        loras.append(lora_entry)
                
            # Try to get base model from resources or make educated guess
            base_model = None
            if base_model_counts:
                # Use the most common base model from the loras
                base_model = max(base_model_counts.items(), key=lambda x: x[1])[0]
            
            # Prepare final result structure
            # Make sure gen_params only contains recognized keys
            filtered_gen_params = {}
            for key in GEN_PARAM_KEYS:
                if key in metadata.get("gen_params", {}):
                    filtered_gen_params[key] = metadata["gen_params"][key]
            
            result = {
                'base_model': base_model,
                'loras': loras,
                'gen_params': filtered_gen_params,
                'from_automatic_metadata': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing Automatic1111 metadata: {e}", exc_info=True)
            return {"error": str(e), "loras": []}
