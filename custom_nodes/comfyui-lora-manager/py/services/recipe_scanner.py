import os
import logging
import asyncio
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from ..config import config
from .recipe_cache import RecipeCache
from .service_registry import ServiceRegistry
from .lora_scanner import LoraScanner
from ..utils.utils import fuzzy_match
from natsort import natsorted
import sys

logger = logging.getLogger(__name__)

class RecipeScanner:
    """Service for scanning and managing recipe images"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_instance(cls, lora_scanner: Optional[LoraScanner] = None):
        """Get singleton instance of RecipeScanner"""
        async with cls._lock:
            if cls._instance is None:
                if not lora_scanner:
                    # Get lora scanner from service registry if not provided
                    lora_scanner = await ServiceRegistry.get_lora_scanner()
                cls._instance = cls(lora_scanner)
            return cls._instance
    
    def __new__(cls, lora_scanner: Optional[LoraScanner] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lora_scanner = lora_scanner
            cls._instance._civitai_client = None  # Will be lazily initialized
        return cls._instance
    
    def __init__(self, lora_scanner: Optional[LoraScanner] = None):
        # Ensure initialization only happens once
        if not hasattr(self, '_initialized'):
            self._cache: Optional[RecipeCache] = None
            self._initialization_lock = asyncio.Lock()
            self._initialization_task: Optional[asyncio.Task] = None
            self._is_initializing = False
            if lora_scanner:
                self._lora_scanner = lora_scanner
            self._initialized = True
    
    async def _get_civitai_client(self):
        """Lazily initialize CivitaiClient from registry"""
        if self._civitai_client is None:
            self._civitai_client = await ServiceRegistry.get_civitai_client()
        return self._civitai_client
    
    async def initialize_in_background(self) -> None:
        """Initialize cache in background using thread pool"""
        try:
            # Set initial empty cache to avoid None reference errors
            if self._cache is None:
                self._cache = RecipeCache(
                    raw_data=[],
                    sorted_by_name=[],
                    sorted_by_date=[]
                )
            
            # Mark as initializing to prevent concurrent initializations
            self._is_initializing = True
            
            try:
                # Start timer
                start_time = time.time()
                
                # Use thread pool to execute CPU-intensive operations
                loop = asyncio.get_event_loop()
                cache = await loop.run_in_executor(
                    None,  # Use default thread pool
                    self._initialize_recipe_cache_sync  # Run synchronous version in thread
                )
                
                # Calculate elapsed time and log it
                elapsed_time = time.time() - start_time
                recipe_count = len(cache.raw_data) if cache and hasattr(cache, 'raw_data') else 0
                logger.info(f"Recipe cache initialized in {elapsed_time:.2f} seconds. Found {recipe_count} recipes")
            finally:
                # Mark initialization as complete regardless of outcome
                self._is_initializing = False
        except Exception as e:
            logger.error(f"Recipe Scanner: Error initializing cache in background: {e}")
    
    def _initialize_recipe_cache_sync(self):
        """Synchronous version of recipe cache initialization for thread pool execution"""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create a synchronous method to bypass the async lock
            def sync_initialize_cache():
                # We need to implement scan_all_recipes logic synchronously here
                # instead of calling the async method to avoid event loop issues
                recipes = []
                recipes_dir = self.recipes_dir
                
                if not recipes_dir or not os.path.exists(recipes_dir):
                    logger.warning(f"Recipes directory not found: {recipes_dir}")
                    return recipes
                
                # Get all recipe JSON files in the recipes directory
                recipe_files = []
                for root, _, files in os.walk(recipes_dir):
                    recipe_count = sum(1 for f in files if f.lower().endswith('.recipe.json'))
                    if recipe_count > 0:
                        for file in files:
                            if file.lower().endswith('.recipe.json'):
                                recipe_files.append(os.path.join(root, file))
                
                # Process each recipe file
                for recipe_path in recipe_files:
                    try:
                        with open(recipe_path, 'r', encoding='utf-8') as f:
                            recipe_data = json.load(f)
                        
                        # Validate recipe data
                        if not recipe_data or not isinstance(recipe_data, dict):
                            logger.warning(f"Invalid recipe data in {recipe_path}")
                            continue
                        
                        # Ensure required fields exist
                        required_fields = ['id', 'file_path', 'title']
                        if not all(field in recipe_data for field in required_fields):
                            logger.warning(f"Missing required fields in {recipe_path}")
                            continue
                        
                        # Ensure the image file exists
                        image_path = recipe_data.get('file_path')
                        if not os.path.exists(image_path):
                            recipe_dir = os.path.dirname(recipe_path)
                            image_filename = os.path.basename(image_path)
                            alternative_path = os.path.join(recipe_dir, image_filename)
                            if os.path.exists(alternative_path):
                                recipe_data['file_path'] = alternative_path
                        
                        # Ensure loras array exists
                        if 'loras' not in recipe_data:
                            recipe_data['loras'] = []
                        
                        # Ensure gen_params exists
                        if 'gen_params' not in recipe_data:
                            recipe_data['gen_params'] = {}
                        
                        # Add to list without async operations
                        recipes.append(recipe_data)
                    except Exception as e:
                        logger.error(f"Error loading recipe file {recipe_path}: {e}")
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                
                # Update cache with the collected data
                self._cache.raw_data = recipes
                
                # Create a simplified resort function that doesn't use await
                if hasattr(self._cache, "resort"):
                    try:
                        # Sort by name
                        self._cache.sorted_by_name = natsorted(
                            self._cache.raw_data,
                            key=lambda x: x.get('title', '').lower()
                        )
                        
                        # Sort by date (modified or created)
                        self._cache.sorted_by_date = sorted(
                            self._cache.raw_data,
                            key=lambda x: x.get('modified', x.get('created_date', 0)),
                            reverse=True
                        )
                    except Exception as e:
                        logger.error(f"Error sorting recipe cache: {e}")
                
                return self._cache
            
            # Run our sync initialization that avoids lock conflicts
            return sync_initialize_cache()
        except Exception as e:
            logger.error(f"Error in thread-based recipe cache initialization: {e}")
            return self._cache if hasattr(self, '_cache') else None
        finally:
            # Clean up the event loop
            loop.close()

    @property
    def recipes_dir(self) -> str:
        """Get path to recipes directory"""
        if not config.loras_roots:
            return ""
        
        # config.loras_roots already sorted case-insensitively, use the first one
        recipes_dir = os.path.join(config.loras_roots[0], "recipes")
        os.makedirs(recipes_dir, exist_ok=True)
        
        return recipes_dir
    
    async def get_cached_data(self, force_refresh: bool = False) -> RecipeCache:
        """Get cached recipe data, refresh if needed"""
        # If cache is already initialized and no refresh is needed, return it immediately
        if self._cache is not None and not force_refresh:
            return self._cache

        # If another initialization is already in progress, wait for it to complete
        if self._is_initializing and not force_refresh:
            return self._cache or RecipeCache(raw_data=[], sorted_by_name=[], sorted_by_date=[])

        # If force refresh is requested, initialize the cache directly
        if force_refresh:
            # Try to acquire the lock with a timeout to prevent deadlocks
            try:
                async with self._initialization_lock:
                    # Mark as initializing to prevent concurrent initializations
                    self._is_initializing = True
                    
                    try:
                        # Scan for recipe data directly
                        raw_data = await self.scan_all_recipes()
                        
                        # Update cache
                        self._cache = RecipeCache(
                            raw_data=raw_data,
                            sorted_by_name=[],
                            sorted_by_date=[]
                        )
                        
                        # Resort cache
                        await self._cache.resort()
                        
                        return self._cache
                    
                    except Exception as e:
                        logger.error(f"Recipe Manager: Error initializing cache: {e}", exc_info=True)
                        # Create empty cache on error
                        self._cache = RecipeCache(
                            raw_data=[],
                            sorted_by_name=[],
                            sorted_by_date=[]
                        )
                        return self._cache
                    finally:
                        # Mark initialization as complete
                        self._is_initializing = False
            
            except Exception as e:
                logger.error(f"Unexpected error in get_cached_data: {e}")
        
        # Return the cache (may be empty or partially initialized)
        return self._cache or RecipeCache(raw_data=[], sorted_by_name=[], sorted_by_date=[])
    
    async def scan_all_recipes(self) -> List[Dict]:
        """Scan all recipe JSON files and return metadata"""
        recipes = []
        recipes_dir = self.recipes_dir
        
        if not recipes_dir or not os.path.exists(recipes_dir):
            logger.warning(f"Recipes directory not found: {recipes_dir}")
            return recipes
        
        # Get all recipe JSON files in the recipes directory
        recipe_files = []
        for root, _, files in os.walk(recipes_dir):
            recipe_count = sum(1 for f in files if f.lower().endswith('.recipe.json'))
            if recipe_count > 0:
                for file in files:
                    if file.lower().endswith('.recipe.json'):
                        recipe_files.append(os.path.join(root, file))
        
        # Process each recipe file
        for recipe_path in recipe_files:
            recipe_data = await self._load_recipe_file(recipe_path)
            if recipe_data:
                recipes.append(recipe_data)
        
        return recipes
    
    async def _load_recipe_file(self, recipe_path: str) -> Optional[Dict]:
        """Load recipe data from a JSON file"""
        try:
            with open(recipe_path, 'r', encoding='utf-8') as f:
                recipe_data = json.load(f)
            
            # Validate recipe data
            if not recipe_data or not isinstance(recipe_data, dict):
                logger.warning(f"Invalid recipe data in {recipe_path}")
                return None
            
            # Ensure required fields exist
            required_fields = ['id', 'file_path', 'title']
            for field in required_fields:
                if field not in recipe_data:
                    logger.warning(f"Missing required field '{field}' in {recipe_path}")
                    return None
            
            # Ensure the image file exists
            image_path = recipe_data.get('file_path')
            if not os.path.exists(image_path):
                logger.warning(f"Recipe image not found: {image_path}")
                # Try to find the image in the same directory as the recipe
                recipe_dir = os.path.dirname(recipe_path)
                image_filename = os.path.basename(image_path)
                alternative_path = os.path.join(recipe_dir, image_filename)
                if os.path.exists(alternative_path):
                    recipe_data['file_path'] = alternative_path
                else:
                    logger.warning(f"Could not find alternative image path for {image_path}")
            
            # Ensure loras array exists
            if 'loras' not in recipe_data:
                recipe_data['loras'] = []
            
            # Ensure gen_params exists
            if 'gen_params' not in recipe_data:
                recipe_data['gen_params'] = {}
            
            # Update lora information with local paths and availability
            await self._update_lora_information(recipe_data)

            # Calculate and update fingerprint if missing
            if 'loras' in recipe_data and 'fingerprint' not in recipe_data:
                from ..utils.utils import calculate_recipe_fingerprint
                fingerprint = calculate_recipe_fingerprint(recipe_data['loras'])
                recipe_data['fingerprint'] = fingerprint
                
                # Write updated recipe data back to file
                try:
                    with open(recipe_path, 'w', encoding='utf-8') as f:
                        json.dump(recipe_data, f, indent=4, ensure_ascii=False)
                    logger.info(f"Added fingerprint to recipe: {recipe_path}")
                except Exception as e:
                    logger.error(f"Error writing updated recipe with fingerprint: {e}")
            
            return recipe_data
        except Exception as e:
            logger.error(f"Error loading recipe file {recipe_path}: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None
    
    async def _update_lora_information(self, recipe_data: Dict) -> bool:
        """Update LoRA information with hash and file_name
        
        Returns:
            bool: True if metadata was updated
        """
        if not recipe_data.get('loras'):
            return False
        
        metadata_updated = False
        
        for lora in recipe_data['loras']:
            # Skip deleted loras that were already marked
            if lora.get('isDeleted', False):
                continue
                
            # Skip if already has complete information
            if 'hash' in lora and 'file_name' in lora and lora['file_name']:
                continue
                
            # If has modelVersionId but no hash, look in lora cache first, then fetch from Civitai
            if 'modelVersionId' in lora and not lora.get('hash'):
                model_version_id = lora['modelVersionId']

                # Try to find in lora cache first
                hash_from_cache = await self._find_hash_in_lora_cache(model_version_id)
                if hash_from_cache:
                    lora['hash'] = hash_from_cache
                    metadata_updated = True
                else:
                    # If not in cache, fetch from Civitai
                    result = await self._get_hash_from_civitai(model_version_id)
                    if isinstance(result, tuple):
                        hash_from_civitai, is_deleted = result
                        if hash_from_civitai:
                            lora['hash'] = hash_from_civitai
                            metadata_updated = True
                        elif is_deleted:
                            # Mark the lora as deleted if it was not found on Civitai
                            lora['isDeleted'] = True
                            logger.warning(f"Marked lora with modelVersionId {model_version_id} as deleted")
                            metadata_updated = True
                    else:
                        logger.debug(f"Could not get hash for modelVersionId {model_version_id}")
            
            # If has hash but no file_name, look up in lora library
            if 'hash' in lora and (not lora.get('file_name') or not lora['file_name']):
                hash_value = lora['hash']
                
                if self._lora_scanner.has_hash(hash_value):
                    lora_path = self._lora_scanner.get_path_by_hash(hash_value)
                    if lora_path:
                        file_name = os.path.splitext(os.path.basename(lora_path))[0]
                        lora['file_name'] = file_name
                        metadata_updated = True
                else:
                    # Lora not in library
                    lora['file_name'] = ''
                    metadata_updated = True
        
        return metadata_updated
    
    async def _find_hash_in_lora_cache(self, model_version_id: str) -> Optional[str]:
        """Find hash in lora cache based on modelVersionId"""
        try:
            # Get all loras from cache
            if not self._lora_scanner:
                return None
                
            cache = await self._lora_scanner.get_cached_data()
            if not cache or not cache.raw_data:
                return None
                
            # Find lora with matching civitai.id
            for lora in cache.raw_data:
                civitai_data = lora.get('civitai', {})
                if civitai_data and str(civitai_data.get('id', '')) == str(model_version_id):
                    return lora.get('sha256')
                    
            return None
        except Exception as e:
            logger.error(f"Error finding hash in lora cache: {e}")
            return None
    
    async def _get_hash_from_civitai(self, model_version_id: str) -> Optional[str]:
        """Get hash from Civitai API"""
        try:
            # Get CivitaiClient from ServiceRegistry
            civitai_client = await self._get_civitai_client()
            if not civitai_client:
                logger.error("Failed to get CivitaiClient from ServiceRegistry")
                return None
                
            version_info, error_msg = await civitai_client.get_model_version_info(model_version_id)
            
            if not version_info:
                if error_msg and "model not found" in error_msg.lower():
                    logger.warning(f"Model with version ID {model_version_id} was not found on Civitai - marking as deleted")
                    return None, True  # Return None hash and True for isDeleted flag
                else:
                    logger.debug(f"Could not get hash for modelVersionId {model_version_id}: {error_msg}")
                    return None, False  # Return None hash but not marked as deleted
                    
            # Get hash from the first file
            for file_info in version_info.get('files', []):
                if file_info.get('hashes', {}).get('SHA256'):
                    return file_info['hashes']['SHA256'], False  # Return hash with False for isDeleted flag
                    
            logger.debug(f"No SHA256 hash found in version info for ID: {model_version_id}")
            return None, False
        except Exception as e:
            logger.error(f"Error getting hash from Civitai: {e}")
            return None, False

    async def _determine_base_model(self, loras: List[Dict]) -> Optional[str]:
        """Determine the most common base model among LoRAs"""
        base_models = {}
        
        # Count occurrences of each base model
        for lora in loras:
            if 'hash' in lora:
                lora_path = self._lora_scanner.get_path_by_hash(lora['hash'])
                if lora_path:
                    base_model = await self._get_base_model_for_lora(lora_path)
                    if base_model:
                        base_models[base_model] = base_models.get(base_model, 0) + 1
        
        # Return the most common base model
        if base_models:
            return max(base_models.items(), key=lambda x: x[1])[0]
        return None

    async def _get_base_model_for_lora(self, lora_path: str) -> Optional[str]:
        """Get base model for a LoRA from cache"""
        try:
            if not self._lora_scanner:
                return None
            
            cache = await self._lora_scanner.get_cached_data()
            if not cache or not cache.raw_data:
                return None
            
            # Find matching lora in cache
            for lora in cache.raw_data:
                if lora.get('file_path') == lora_path:
                    return lora.get('base_model')
                
            return None
        except Exception as e:
            logger.error(f"Error getting base model for lora: {e}")
            return None

    async def get_paginated_data(self, page: int, page_size: int, sort_by: str = 'date', search: str = None, filters: dict = None, search_options: dict = None, lora_hash: str = None, bypass_filters: bool = True):
        """Get paginated and filtered recipe data
        
        Args:
            page: Current page number (1-based)
            page_size: Number of items per page
            sort_by: Sort method ('name' or 'date')
            search: Search term
            filters: Dictionary of filters to apply
            search_options: Dictionary of search options to apply
            lora_hash: Optional SHA256 hash of a LoRA to filter recipes by
            bypass_filters: If True, ignore other filters when a lora_hash is provided
        """
        cache = await self.get_cached_data()

        # Get base dataset
        filtered_data = cache.sorted_by_date if sort_by == 'date' else cache.sorted_by_name
        
        # Special case: Filter by LoRA hash (takes precedence if bypass_filters is True)
        if lora_hash:
            # Filter recipes that contain this LoRA hash
            filtered_data = [
                item for item in filtered_data
                if 'loras' in item and any(
                    lora.get('hash', '').lower() == lora_hash.lower() 
                    for lora in item['loras']
                )
            ]
            
            if bypass_filters:
                # Skip other filters if bypass_filters is True
                pass
            # Otherwise continue with normal filtering after applying LoRA hash filter
        
        # Skip further filtering if we're only filtering by LoRA hash with bypass enabled
        if not (lora_hash and bypass_filters):
            # Apply search filter
            if search:
                # Default search options if none provided
                if not search_options:
                    search_options = {
                        'title': True,
                        'tags': True,
                        'lora_name': True,
                        'lora_model': True
                    }
                
                # Build the search predicate based on search options
                def matches_search(item):
                    # Search in title if enabled
                    if search_options.get('title', True):
                        if fuzzy_match(str(item.get('title', '')), search):
                            return True
                    
                    # Search in tags if enabled
                    if search_options.get('tags', True) and 'tags' in item:
                        for tag in item['tags']:
                            if fuzzy_match(tag, search):
                                return True
                    
                    # Search in lora file names if enabled
                    if search_options.get('lora_name', True) and 'loras' in item:
                        for lora in item['loras']:
                            if fuzzy_match(str(lora.get('file_name', '')), search):
                                return True
                    
                    # Search in lora model names if enabled
                    if search_options.get('lora_model', True) and 'loras' in item:
                        for lora in item['loras']:
                            if fuzzy_match(str(lora.get('modelName', '')), search):
                                return True
                    
                    # No match found
                    return False
                
                # Filter the data using the search predicate
                filtered_data = [item for item in filtered_data if matches_search(item)]
            
            # Apply additional filters
            if filters:
                # Filter by base model
                if 'base_model' in filters and filters['base_model']:
                    filtered_data = [
                        item for item in filtered_data
                        if item.get('base_model', '') in filters['base_model']
                    ]
                
                # Filter by tags
                if 'tags' in filters and filters['tags']:
                    filtered_data = [
                        item for item in filtered_data
                        if any(tag in item.get('tags', []) for tag in filters['tags'])
                    ]

        # Calculate pagination
        total_items = len(filtered_data)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        # Get paginated items
        paginated_items = filtered_data[start_idx:end_idx]
        
        # Add inLibrary information for each lora
        for item in paginated_items:
            if 'loras' in item:
                for lora in item['loras']:
                    if 'hash' in lora and lora['hash']:
                        lora['inLibrary'] = self._lora_scanner.has_hash(lora['hash'].lower())
                        lora['preview_url'] = self._lora_scanner.get_preview_url_by_hash(lora['hash'].lower())
                        lora['localPath'] = self._lora_scanner.get_path_by_hash(lora['hash'].lower())
        
        result = {
            'items': paginated_items,
            'total': total_items,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_items + page_size - 1) // page_size
        }
        
        return result
    
    async def get_recipe_by_id(self, recipe_id: str) -> dict:
        """Get a single recipe by ID with all metadata and formatted URLs
        
        Args:
            recipe_id: The ID of the recipe to retrieve
            
        Returns:
            Dict containing the recipe data or None if not found
        """
        if not recipe_id:
            return None
            
        # Get all recipes from cache
        cache = await self.get_cached_data()
        
        # Find the recipe with the specified ID
        recipe = next((r for r in cache.raw_data if str(r.get('id', '')) == recipe_id), None)
        
        if not recipe:
            return None
            
        # Format the recipe with all needed information
        formatted_recipe = {**recipe}  # Copy all fields
        
        # Format file path to URL
        if 'file_path' in formatted_recipe:
            formatted_recipe['file_url'] = self._format_file_url(formatted_recipe['file_path'])
            
        # Format dates for display
        for date_field in ['created_date', 'modified']:
            if date_field in formatted_recipe:
                formatted_recipe[f"{date_field}_formatted"] = self._format_timestamp(formatted_recipe[date_field])
                
        # Add lora metadata
        if 'loras' in formatted_recipe:
            for lora in formatted_recipe['loras']:
                if 'hash' in lora and lora['hash']:
                    lora_hash = lora['hash'].lower()
                    lora['inLibrary'] = self._lora_scanner.has_hash(lora_hash)
                    lora['preview_url'] = self._lora_scanner.get_preview_url_by_hash(lora_hash)
                    lora['localPath'] = self._lora_scanner.get_path_by_hash(lora_hash)
                    
        return formatted_recipe
        
    def _format_file_url(self, file_path: str) -> str:
        """Format file path as URL for serving in web UI"""
        if not file_path:
            return '/loras_static/images/no-preview.png'
            
        try:
            # Format file path as a URL that will work with static file serving
            recipes_dir = os.path.join(config.loras_roots[0], "recipes").replace(os.sep, '/')
            if file_path.replace(os.sep, '/').startswith(recipes_dir):
                relative_path = os.path.relpath(file_path, config.loras_roots[0]).replace(os.sep, '/')
                return f"/loras_static/root1/preview/{relative_path}"
                
            # If not in recipes dir, try to create a valid URL from the file name
            file_name = os.path.basename(file_path)
            return f"/loras_static/root1/preview/recipes/{file_name}"
        except Exception as e:
            logger.error(f"Error formatting file URL: {e}")
            return '/loras_static/images/no-preview.png'
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    async def update_recipe_metadata(self, recipe_id: str, metadata: dict) -> bool:
        """Update recipe metadata (like title and tags) in both file system and cache
        
        Args:
            recipe_id: The ID of the recipe to update
            metadata: Dictionary containing metadata fields to update (title, tags, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        import os
        import json
        
        # First, find the recipe JSON file path
        recipe_json_path = os.path.join(self.recipes_dir, f"{recipe_id}.recipe.json")
        
        if not os.path.exists(recipe_json_path):
            return False
            
        try:
            # Load existing recipe data
            with open(recipe_json_path, 'r', encoding='utf-8') as f:
                recipe_data = json.load(f)
                
            # Update fields
            for key, value in metadata.items():
                recipe_data[key] = value
                
            # Save updated recipe
            with open(recipe_json_path, 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, indent=4, ensure_ascii=False)
                
            # Update the cache if it exists
            if self._cache is not None:
                await self._cache.update_recipe_metadata(recipe_id, metadata)
                
            # If the recipe has an image, update its EXIF metadata
            from ..utils.exif_utils import ExifUtils
            image_path = recipe_data.get('file_path')
            if image_path and os.path.exists(image_path):
                ExifUtils.append_recipe_metadata(image_path, recipe_data)
                
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error updating recipe metadata: {e}", exc_info=True)
            return False

    async def update_lora_filename_by_hash(self, hash_value: str, new_file_name: str) -> Tuple[int, int]:
        """Update file_name in all recipes that contain a LoRA with the specified hash.
        
        Args:
            hash_value: The SHA256 hash value of the LoRA
            new_file_name: The new file_name to set
            
        Returns:
            Tuple[int, int]: (number of recipes updated in files, number of recipes updated in cache)
        """
        if not hash_value or not new_file_name:
            return 0, 0
            
        # Always use lowercase hash for consistency
        hash_value = hash_value.lower()
        
        # Get recipes directory
        recipes_dir = self.recipes_dir
        if not recipes_dir or not os.path.exists(recipes_dir):
            logger.warning(f"Recipes directory not found: {recipes_dir}")
            return 0, 0
            
        # Check if cache is initialized
        cache_initialized = self._cache is not None
        cache_updated_count = 0
        file_updated_count = 0
        
        # Get all recipe JSON files in the recipes directory
        recipe_files = []
        for root, _, files in os.walk(recipes_dir):
            for file in files:
                if file.lower().endswith('.recipe.json'):
                    recipe_files.append(os.path.join(root, file))
        
        # Process each recipe file
        for recipe_path in recipe_files:
            try:
                # Load the recipe data
                with open(recipe_path, 'r', encoding='utf-8') as f:
                    recipe_data = json.load(f)
                
                # Skip if no loras or invalid structure
                if not recipe_data or not isinstance(recipe_data, dict) or 'loras' not in recipe_data:
                    continue
                
                # Check if any lora has matching hash
                file_updated = False
                for lora in recipe_data.get('loras', []):
                    if 'hash' in lora and lora['hash'].lower() == hash_value:
                        # Update file_name
                        old_file_name = lora.get('file_name', '')
                        lora['file_name'] = new_file_name
                        file_updated = True
                        logger.info(f"Updated file_name in recipe {recipe_path}: {old_file_name} -> {new_file_name}")
                
                # If updated, save the file
                if file_updated:
                    with open(recipe_path, 'w', encoding='utf-8') as f:
                        json.dump(recipe_data, f, indent=4, ensure_ascii=False)
                    file_updated_count += 1
                    
                    # Also update in cache if it exists
                    if cache_initialized:
                        recipe_id = recipe_data.get('id')
                        if recipe_id:
                            for cache_item in self._cache.raw_data:
                                if cache_item.get('id') == recipe_id:
                                    # Replace loras array with updated version
                                    cache_item['loras'] = recipe_data['loras']
                                    cache_updated_count += 1
                                    break
            
            except Exception as e:
                logger.error(f"Error updating recipe file {recipe_path}: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
        
        # Resort cache if updates were made
        if cache_initialized and cache_updated_count > 0:
            await self._cache.resort()
            logger.info(f"Resorted recipe cache after updating {cache_updated_count} items")
            
        return file_updated_count, cache_updated_count

    async def find_recipes_by_fingerprint(self, fingerprint: str) -> list:
        """Find recipes with a matching fingerprint
        
        Args:
            fingerprint: The recipe fingerprint to search for
            
        Returns:
            List of recipe details that match the fingerprint
        """
        if not fingerprint:
            return []
            
        # Get all recipes from cache
        cache = await self.get_cached_data()
        
        # Find recipes with matching fingerprint
        matching_recipes = []
        for recipe in cache.raw_data:
            if recipe.get('fingerprint') == fingerprint:
                recipe_details = {
                    'id': recipe.get('id'),
                    'title': recipe.get('title'),
                    'file_url': self._format_file_url(recipe.get('file_path')),
                    'modified': recipe.get('modified'),
                    'created_date': recipe.get('created_date'),
                    'lora_count': len(recipe.get('loras', []))
                }
                matching_recipes.append(recipe_details)
        
        return matching_recipes
        
    async def find_all_duplicate_recipes(self) -> dict:
        """Find all recipe duplicates based on fingerprints
        
        Returns:
            Dictionary where keys are fingerprints and values are lists of recipe IDs
        """
        # Get all recipes from cache
        cache = await self.get_cached_data()
        
        # Group recipes by fingerprint
        fingerprint_groups = {}
        for recipe in cache.raw_data:
            fingerprint = recipe.get('fingerprint')
            if not fingerprint:
                continue
                
            if fingerprint not in fingerprint_groups:
                fingerprint_groups[fingerprint] = []
                
            fingerprint_groups[fingerprint].append(recipe.get('id'))
        
        # Filter to only include groups with more than one recipe
        duplicate_groups = {k: v for k, v in fingerprint_groups.items() if len(v) > 1}
        
        return duplicate_groups
