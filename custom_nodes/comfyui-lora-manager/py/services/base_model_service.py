from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import logging

from ..utils.models import BaseModelMetadata
from ..utils.constants import NSFW_LEVELS
from .settings_manager import settings
from ..utils.utils import fuzzy_match

logger = logging.getLogger(__name__)

class BaseModelService(ABC):
    """Base service class for all model types"""
    
    def __init__(self, model_type: str, scanner, metadata_class: Type[BaseModelMetadata]):
        """Initialize the service
        
        Args:
            model_type: Type of model (lora, checkpoint, etc.)
            scanner: Model scanner instance
            metadata_class: Metadata class for this model type
        """
        self.model_type = model_type
        self.scanner = scanner
        self.metadata_class = metadata_class
    
    async def get_paginated_data(self, page: int, page_size: int, sort_by: str = 'name', 
                               folder: str = None, search: str = None, fuzzy_search: bool = False,
                               base_models: list = None, tags: list = None,
                               search_options: dict = None, hash_filters: dict = None,
                               favorites_only: bool = False, **kwargs) -> Dict:
        """Get paginated and filtered model data
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            sort_by: Sort criteria, e.g. 'name', 'name:asc', 'name:desc', 'date', 'date:asc', 'date:desc'
            folder: Folder filter
            search: Search term
            fuzzy_search: Whether to use fuzzy search
            base_models: List of base models to filter by
            tags: List of tags to filter by
            search_options: Search options dict
            hash_filters: Hash filtering options
            favorites_only: Filter for favorites only
            **kwargs: Additional model-specific filters
            
        Returns:
            Dict containing paginated results
        """
        cache = await self.scanner.get_cached_data()

        # Parse sort_by into sort_key and order
        if ':' in sort_by:
            sort_key, order = sort_by.split(':', 1)
            sort_key = sort_key.strip()
            order = order.strip().lower()
            if order not in ('asc', 'desc'):
                order = 'asc'
        else:
            sort_key = sort_by.strip()
            order = 'asc'

        # Get default search options if not provided
        if search_options is None:
            search_options = {
                'filename': True,
                'modelname': True,
                'tags': False,
                'recursive': False,
            }

        # Get the base data set using new sort logic
        filtered_data = await cache.get_sorted_data(sort_key, order)
        
        # Apply hash filtering if provided (highest priority)
        if hash_filters:
            filtered_data = await self._apply_hash_filters(filtered_data, hash_filters)
            
            # Jump to pagination for hash filters
            return self._paginate(filtered_data, page, page_size)
        
        # Apply common filters
        filtered_data = await self._apply_common_filters(
            filtered_data, folder, base_models, tags, favorites_only, search_options
        )
        
        # Apply search filtering
        if search:
            filtered_data = await self._apply_search_filters(
                filtered_data, search, fuzzy_search, search_options
            )
        
        # Apply model-specific filters
        filtered_data = await self._apply_specific_filters(filtered_data, **kwargs)
        
        return self._paginate(filtered_data, page, page_size)
    
    async def _apply_hash_filters(self, data: List[Dict], hash_filters: Dict) -> List[Dict]:
        """Apply hash-based filtering"""
        single_hash = hash_filters.get('single_hash')
        multiple_hashes = hash_filters.get('multiple_hashes')
        
        if single_hash:
            # Filter by single hash
            single_hash = single_hash.lower()
            return [
                item for item in data
                if item.get('sha256', '').lower() == single_hash
            ]
        elif multiple_hashes:
            # Filter by multiple hashes
            hash_set = set(hash.lower() for hash in multiple_hashes)
            return [
                item for item in data
                if item.get('sha256', '').lower() in hash_set
            ]
        
        return data
    
    async def _apply_common_filters(self, data: List[Dict], folder: str = None, 
                                  base_models: list = None, tags: list = None,
                                  favorites_only: bool = False, search_options: dict = None) -> List[Dict]:
        """Apply common filters that work across all model types"""
        # Apply SFW filtering if enabled in settings
        if settings.get('show_only_sfw', False):
            data = [
                item for item in data
                if not item.get('preview_nsfw_level') or item.get('preview_nsfw_level') < NSFW_LEVELS['R']
            ]
        
        # Apply favorites filtering if enabled
        if favorites_only:
            data = [
                item for item in data
                if item.get('favorite', False) is True
            ]
        
        # Apply folder filtering
        if folder is not None:
            if search_options and search_options.get('recursive', False):
                # Recursive folder filtering - include all subfolders
                data = [
                    item for item in data
                    if item['folder'].startswith(folder)
                ]
            else:
                # Exact folder filtering
                data = [
                    item for item in data
                    if item['folder'] == folder
                ]
        
        # Apply base model filtering
        if base_models and len(base_models) > 0:
            data = [
                item for item in data
                if item.get('base_model') in base_models
            ]
        
        # Apply tag filtering
        if tags and len(tags) > 0:
            data = [
                item for item in data
                if any(tag in item.get('tags', []) for tag in tags)
            ]
        
        return data
    
    async def _apply_search_filters(self, data: List[Dict], search: str, 
                                  fuzzy_search: bool, search_options: dict) -> List[Dict]:
        """Apply search filtering"""
        search_results = []
        
        for item in data:
            # Search by file name
            if search_options.get('filename', True):
                if fuzzy_search:
                    if fuzzy_match(item.get('file_name', ''), search):
                        search_results.append(item)
                        continue
                elif search.lower() in item.get('file_name', '').lower():
                    search_results.append(item)
                    continue
            
            # Search by model name
            if search_options.get('modelname', True):
                if fuzzy_search:
                    if fuzzy_match(item.get('model_name', ''), search):
                        search_results.append(item)
                        continue
                elif search.lower() in item.get('model_name', '').lower():
                    search_results.append(item)
                    continue
            
            # Search by tags
            if search_options.get('tags', False) and 'tags' in item:
                if any((fuzzy_match(tag, search) if fuzzy_search else search.lower() in tag.lower()) 
                       for tag in item['tags']):
                    search_results.append(item)
                    continue
            
            # Search by creator
            civitai = item.get('civitai')
            creator_username = ''
            if civitai and isinstance(civitai, dict):
                creator = civitai.get('creator')
                if creator and isinstance(creator, dict):
                    creator_username = creator.get('username', '')
            if search_options.get('creator', False) and creator_username:
                if fuzzy_search:
                    if fuzzy_match(creator_username, search):
                        search_results.append(item)
                        continue
                elif search.lower() in creator_username.lower():
                    search_results.append(item)
                    continue
        
        return search_results
    
    async def _apply_specific_filters(self, data: List[Dict], **kwargs) -> List[Dict]:
        """Apply model-specific filters - to be overridden by subclasses if needed"""
        return data
    
    def _paginate(self, data: List[Dict], page: int, page_size: int) -> Dict:
        """Apply pagination to filtered data"""
        total_items = len(data)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        return {
            'items': data[start_idx:end_idx],
            'total': total_items,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_items + page_size - 1) // page_size
        }
    
    @abstractmethod
    async def format_response(self, model_data: Dict) -> Dict:
        """Format model data for API response - must be implemented by subclasses"""
        pass
    
    # Common service methods that delegate to scanner
    async def get_top_tags(self, limit: int = 20) -> List[Dict]:
        """Get top tags sorted by frequency"""
        return await self.scanner.get_top_tags(limit)
    
    async def get_base_models(self, limit: int = 20) -> List[Dict]:
        """Get base models sorted by frequency"""
        return await self.scanner.get_base_models(limit)
    
    def has_hash(self, sha256: str) -> bool:
        """Check if a model with given hash exists"""
        return self.scanner.has_hash(sha256)
    
    def get_path_by_hash(self, sha256: str) -> Optional[str]:
        """Get file path for a model by its hash"""
        return self.scanner.get_path_by_hash(sha256)
    
    def get_hash_by_path(self, file_path: str) -> Optional[str]:
        """Get hash for a model by its file path"""
        return self.scanner.get_hash_by_path(file_path)
    
    async def scan_models(self, force_refresh: bool = False, rebuild_cache: bool = False):
        """Trigger model scanning"""
        return await self.scanner.get_cached_data(force_refresh=force_refresh, rebuild_cache=rebuild_cache)
    
    async def get_model_info_by_name(self, name: str):
        """Get model information by name"""
        return await self.scanner.get_model_info_by_name(name)
    
    def get_model_roots(self) -> List[str]:
        """Get model root directories"""
        return self.scanner.get_model_roots()
    
    async def get_folder_tree(self, model_root: str) -> Dict:
        """Get hierarchical folder tree for a specific model root"""
        cache = await self.scanner.get_cached_data()
        
        # Build tree structure from folders
        tree = {}
        
        for folder in cache.folders:
            # Check if this folder belongs to the specified model root
            folder_belongs_to_root = False
            for root in self.scanner.get_model_roots():
                if root == model_root:
                    folder_belongs_to_root = True
                    break
            
            if not folder_belongs_to_root:
                continue
            
            # Split folder path into components
            parts = folder.split('/') if folder else []
            current_level = tree
            
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
        
        return tree
    
    async def get_unified_folder_tree(self) -> Dict:
        """Get unified folder tree across all model roots"""
        cache = await self.scanner.get_cached_data()
        
        # Build unified tree structure by analyzing all relative paths
        unified_tree = {}
        
        # Get all model roots for path normalization
        model_roots = self.scanner.get_model_roots()
        
        for folder in cache.folders:
            if not folder:  # Skip empty folders
                continue
            
            # Find which root this folder belongs to by checking the actual file paths
            # This is a simplified approach - we'll use the folder as-is since it should already be relative
            relative_path = folder
            
            # Split folder path into components
            parts = relative_path.split('/')
            current_level = unified_tree
            
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
        
        return unified_tree