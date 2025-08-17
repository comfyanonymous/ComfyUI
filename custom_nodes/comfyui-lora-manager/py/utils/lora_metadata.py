from safetensors import safe_open
from typing import Dict, List, Tuple
from .model_utils import determine_base_model
import os
import logging
import json

logger = logging.getLogger(__name__)

async def extract_lora_metadata(file_path: str) -> Dict:
    """Extract essential metadata from safetensors file"""
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata:
                # Only extract base_model from ss_base_model_version
                base_model = determine_base_model(metadata.get("ss_base_model_version"))
                return {"base_model": base_model}
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {str(e)}")
    return {"base_model": "Unknown"}

async def extract_checkpoint_metadata(file_path: str) -> dict:
    """Extract metadata from a checkpoint file to determine model type and base model"""
    try:
        # Analyze filename for clues about the model
        filename = os.path.basename(file_path).lower()
        
        model_info = {
            'base_model': 'Unknown',
            'model_type': 'checkpoint'
        }
        
        # Detect base model from filename
        if 'xl' in filename or 'sdxl' in filename:
            model_info['base_model'] = 'SDXL'
        elif 'sd3' in filename:
            model_info['base_model'] = 'SD3'  
        elif 'sd2' in filename or 'v2' in filename:
            model_info['base_model'] = 'SD2.x'
        elif 'sd1' in filename or 'v1' in filename:
            model_info['base_model'] = 'SD1.5'
        
        # Detect model type from filename
        if 'inpaint' in filename:
            model_info['model_type'] = 'inpainting'
        elif 'anime' in filename:
            model_info['model_type'] = 'anime'
        elif 'realistic' in filename:
            model_info['model_type'] = 'realistic'
        
        # Try to peek at the safetensors file structure if available
        if file_path.endswith('.safetensors'):
            import json
            import struct
            
            with open(file_path, 'rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                header_json = f.read(header_size)
                header = json.loads(header_json)
                
                # Look for specific keys to identify model type
                metadata = header.get('__metadata__', {})
                if metadata:
                    # Try to determine if it's SDXL
                    if any(key.startswith('conditioner.embedders.1') for key in header):
                        model_info['base_model'] = 'SDXL'
                    
                    # Look for model type info
                    if metadata.get('modelspec.architecture') == 'SD-XL':
                        model_info['base_model'] = 'SDXL'
                    elif metadata.get('modelspec.architecture') == 'SD-3':
                        model_info['base_model'] = 'SD3'
                    
                    # Check for specific use case
                    if metadata.get('modelspec.purpose') == 'inpainting':
                        model_info['model_type'] = 'inpainting'
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error extracting checkpoint metadata for {file_path}: {e}")
        # Return default values
        return {'base_model': 'Unknown', 'model_type': 'checkpoint'}

async def extract_trained_words(file_path: str) -> Tuple[List[Tuple[str, int]], str]:
    """Extract trained words from a safetensors file and sort by frequency
    
    Args:
        file_path: Path to the safetensors file
        
    Returns:
        Tuple of:
        - List of (word, frequency) tuples sorted by frequency (highest first)
        - class_tokens value (or None if not found)
    """
    class_tokens = None
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            
            # Extract class_tokens from ss_datasets if present
            if metadata and "ss_datasets" in metadata:
                try:
                    datasets_data = json.loads(metadata["ss_datasets"])
                    # Look for class_tokens in the first subset
                    if datasets_data and isinstance(datasets_data, list) and datasets_data[0].get("subsets"):
                        subsets = datasets_data[0].get("subsets", [])
                        if subsets and isinstance(subsets, list) and len(subsets) > 0:
                            class_tokens = subsets[0].get("class_tokens")
                except Exception as e:
                    logger.error(f"Error parsing ss_datasets for class_tokens: {str(e)}")
            
            # Extract tag frequency as before
            if metadata and "ss_tag_frequency" in metadata:
                # Parse the JSON string into a dictionary
                tag_data = json.loads(metadata["ss_tag_frequency"])
                
                # The structure may have an outer key (like "image_dir" or "img")
                # We need to get the inner dictionary with the actual word frequencies
                if tag_data:
                    # Get the first key (usually "image_dir" or "img")
                    first_key = list(tag_data.keys())[0]
                    words_dict = tag_data[first_key]
                    
                    # Sort words by frequency (highest first)
                    sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)
                    return sorted_words, class_tokens
    except Exception as e:
        logger.error(f"Error extracting trained words from {file_path}: {str(e)}")
    
    return [], class_tokens