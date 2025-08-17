from difflib import SequenceMatcher
import os
from typing import Dict
from ..services.service_registry import ServiceRegistry
from ..config import config
from ..services.settings_manager import settings
from .constants import CIVITAI_MODEL_TAGS
import asyncio

def get_lora_info(lora_name):
    """Get the lora path and trigger words from cache"""
    async def _get_lora_info_async():
        scanner = await ServiceRegistry.get_lora_scanner()
        cache = await scanner.get_cached_data()
        
        for item in cache.raw_data:
            if item.get('file_name') == lora_name:
                file_path = item.get('file_path')
                if file_path:
                    for root in config.loras_roots:
                        root = root.replace(os.sep, '/')
                        if file_path.startswith(root):
                            relative_path = os.path.relpath(file_path, root).replace(os.sep, '/')
                            # Get trigger words from civitai metadata
                            civitai = item.get('civitai', {})
                            trigger_words = civitai.get('trainedWords', []) if civitai else []
                            return relative_path, trigger_words
        return lora_name, []
    
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we're in a running loop, we need to use a different approach
        # Create a new thread to run the async code
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_get_lora_info_async())
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
            
    except RuntimeError:
        # No event loop is running, we can use asyncio.run()
        return asyncio.run(_get_lora_info_async())

def fuzzy_match(text: str, pattern: str, threshold: float = 0.85) -> bool:
        """
        Check if text matches pattern using fuzzy matching.
        Returns True if similarity ratio is above threshold.
        """
        if not pattern or not text:
            return False
        
        # Convert both to lowercase for case-insensitive matching
        text = text.lower()
        pattern = pattern.lower()
        
        # Split pattern into words
        search_words = pattern.split()
        
        # Check each word
        for word in search_words:
            # First check if word is a substring (faster)
            if word in text:
                continue
            
            # If not found as substring, try fuzzy matching
            # Check if any part of the text matches this word
            found_match = False
            for text_part in text.split():
                ratio = SequenceMatcher(None, text_part, word).ratio()
                if ratio >= threshold:
                    found_match = True
                    break
                
            if not found_match:
                return False
        
        # All words found either as substrings or fuzzy matches
        return True

def calculate_recipe_fingerprint(loras):
    """
    Calculate a unique fingerprint for a recipe based on its LoRAs.
    
    The fingerprint is created by sorting LoRA hashes, filtering invalid entries,
    normalizing strength values to 2 decimal places, and joining in format:
    hash1:strength1|hash2:strength2|...
    
    Args:
        loras (list): List of LoRA dictionaries with hash and strength values
        
    Returns:
        str: The calculated fingerprint
    """
    if not loras:
        return ""
    
    # Filter valid entries and extract hash and strength
    valid_loras = []
    for lora in loras:
        # Skip excluded loras
        if lora.get("exclude", False):
            continue
            
        # Get the hash - use modelVersionId as fallback if hash is empty
        hash_value = lora.get("hash", "").lower()
        if not hash_value and lora.get("isDeleted", False) and lora.get("modelVersionId"):
            hash_value = str(lora.get("modelVersionId"))
            
        # Skip entries without a valid hash
        if not hash_value:
            continue
            
        # Normalize strength to 2 decimal places (check both strength and weight fields)
        strength = round(float(lora.get("strength", lora.get("weight", 1.0))), 2)
        
        valid_loras.append((hash_value, strength))
    
    # Sort by hash
    valid_loras.sort()
    
    # Join in format hash1:strength1|hash2:strength2|...
    fingerprint = "|".join([f"{hash_value}:{strength}" for hash_value, strength in valid_loras])
    
    return fingerprint

def calculate_relative_path_for_model(model_data: Dict, model_type: str = 'lora') -> str:
    """Calculate relative path for existing model using template from settings

    Args:
        model_data: Model data from scanner cache
        model_type: Type of model ('lora', 'checkpoint', 'embedding')

    Returns:
        Relative path string (empty string for flat structure)
    """
    # Get path template from settings for specific model type
    path_template = settings.get_download_path_template(model_type)

    # If template is empty, return empty path (flat structure)
    if not path_template:
        return ''

    # Get base model name from model metadata
    civitai_data = model_data.get('civitai', {})

    # For CivitAI models, prefer civitai data only if 'id' exists; for non-CivitAI models, use model_data directly
    if civitai_data and civitai_data.get('id') is not None:
        base_model = civitai_data.get('baseModel', '')
        # Get author from civitai creator data
        author = civitai_data.get('creator', {}).get('username', 'Anonymous')
    else:
        # Fallback to model_data fields for non-CivitAI models
        base_model = model_data.get('base_model', '')
        author = 'Anonymous'  # Default for non-CivitAI models

    model_tags = model_data.get('tags', [])

    # Apply mapping if available
    base_model_mappings = settings.get('base_model_path_mappings', {})
    mapped_base_model = base_model_mappings.get(base_model, base_model)

    # Find the first Civitai model tag that exists in model_tags
    first_tag = ''
    for civitai_tag in CIVITAI_MODEL_TAGS:
        if civitai_tag in model_tags:
            first_tag = civitai_tag
            break

    # If no Civitai model tag found, fallback to first tag
    if not first_tag and model_tags:
        first_tag = model_tags[0]

    if not first_tag:
        first_tag = 'no tags'  # Default if no tags available

    # Format the template with available data
    formatted_path = path_template
    formatted_path = formatted_path.replace('{base_model}', mapped_base_model)
    formatted_path = formatted_path.replace('{first_tag}', first_tag)
    formatted_path = formatted_path.replace('{author}', author)

    return formatted_path

def remove_empty_dirs(path):
    """Recursively remove empty directories starting from the given path.
    
    Args:
        path (str): Root directory to start cleaning from
        
    Returns:
        int: Number of empty directories removed
    """
    removed_count = 0
    
    if not os.path.isdir(path):
        return removed_count
        
    # List all files in directory
    files = os.listdir(path)
    
    # Process all subdirectories first
    for file in files:
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            removed_count += remove_empty_dirs(full_path)
    
    # Check if directory is now empty (after processing subdirectories)
    if not os.listdir(path):
        try:
            os.rmdir(path)
            removed_count += 1
        except OSError:
            pass
    
    return removed_count
