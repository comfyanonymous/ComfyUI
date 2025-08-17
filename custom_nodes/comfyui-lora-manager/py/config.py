import os
import platform
import folder_paths # type: ignore
from typing import List
import logging
import sys
import json
import urllib.parse

# Check if running in standalone mode
standalone_mode = 'nodes' not in sys.modules

logger = logging.getLogger(__name__)

class Config:
    """Global configuration for LoRA Manager"""
    
    def __init__(self):
        self.templates_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        self.static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
        # Path mapping dictionary, target to link mapping
        self._path_mappings = {}
        # Static route mapping dictionary, target to route mapping
        self._route_mappings = {}
        self.loras_roots = self._init_lora_paths()
        self.checkpoints_roots = None
        self.unet_roots = None
        self.embeddings_roots = None
        self.base_models_roots = self._init_checkpoint_paths()
        self.embeddings_roots = self._init_embedding_paths()
        # Scan symbolic links during initialization
        self._scan_symbolic_links()
        
        if not standalone_mode:
            # Save the paths to settings.json when running in ComfyUI mode
            self.save_folder_paths_to_settings()

    def save_folder_paths_to_settings(self):
        """Save folder paths to settings.json for standalone mode to use later"""
        try:
            # Check if we're running in ComfyUI mode (not standalone)           
            # Load existing settings
            settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')
            settings = {}
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            
            # Update settings with paths
            settings['folder_paths'] = {
                'loras': self.loras_roots,
                'checkpoints': self.checkpoints_roots,
                'unet': self.unet_roots,
                'embeddings': self.embeddings_roots,
            }
            
            # Add default roots if there's only one item and key doesn't exist
            if len(self.loras_roots) == 1 and "default_lora_root" not in settings:
                settings["default_lora_root"] = self.loras_roots[0]
            
            if self.checkpoints_roots and len(self.checkpoints_roots) == 1 and "default_checkpoint_root" not in settings:
                settings["default_checkpoint_root"] = self.checkpoints_roots[0]

            if self.embeddings_roots and len(self.embeddings_roots) == 1 and "default_embedding_root" not in settings:
                settings["default_embedding_root"] = self.embeddings_roots[0]
            
            # Save settings
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            
            logger.info("Saved folder paths to settings.json")
        except Exception as e:
            logger.warning(f"Failed to save folder paths: {e}")

    def _is_link(self, path: str) -> bool:
        try:
            if os.path.islink(path):
                return True
            if platform.system() == 'Windows':
                try:
                    import ctypes
                    FILE_ATTRIBUTE_REPARSE_POINT = 0x400
                    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
                    return attrs != -1 and (attrs & FILE_ATTRIBUTE_REPARSE_POINT)
                except Exception as e:
                    logger.error(f"Error checking Windows reparse point: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking link status for {path}: {e}")
            return False

    def _scan_symbolic_links(self):
        """Scan all symbolic links in LoRA, Checkpoint, and Embedding root directories"""
        for root in self.loras_roots:
            self._scan_directory_links(root)
        
        for root in self.base_models_roots:
            self._scan_directory_links(root)
            
        for root in self.embeddings_roots:
            self._scan_directory_links(root)

    def _scan_directory_links(self, root: str):
        """Recursively scan symbolic links in a directory"""
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if self._is_link(entry.path):
                        target_path = os.path.realpath(entry.path)
                        if os.path.isdir(target_path):
                            self.add_path_mapping(entry.path, target_path)
                            self._scan_directory_links(target_path)
                    elif entry.is_dir(follow_symlinks=False):
                        self._scan_directory_links(entry.path)
        except Exception as e:
            logger.error(f"Error scanning links in {root}: {e}")

    def add_path_mapping(self, link_path: str, target_path: str):
        """Add a symbolic link path mapping
        target_path: actual target path
        link_path: symbolic link path
        """
        normalized_link = os.path.normpath(link_path).replace(os.sep, '/')
        normalized_target = os.path.normpath(target_path).replace(os.sep, '/')
        # Keep the original mapping: target path -> link path
        self._path_mappings[normalized_target] = normalized_link
        logger.info(f"Added path mapping: {normalized_target} -> {normalized_link}")

    def add_route_mapping(self, path: str, route: str):
        """Add a static route mapping"""
        normalized_path = os.path.normpath(path).replace(os.sep, '/')
        self._route_mappings[normalized_path] = route
        # logger.info(f"Added route mapping: {normalized_path} -> {route}")

    def map_path_to_link(self, path: str) -> str:
        """Map a target path back to its symbolic link path"""
        normalized_path = os.path.normpath(path).replace(os.sep, '/')
        # Check if the path is contained in any mapped target path
        for target_path, link_path in self._path_mappings.items():
            if normalized_path.startswith(target_path):
                # If the path starts with the target path, replace with link path
                mapped_path = normalized_path.replace(target_path, link_path, 1)
                return mapped_path
        return path
    
    def map_link_to_path(self, link_path: str) -> str:
        """Map a symbolic link path back to the actual path"""
        normalized_link = os.path.normpath(link_path).replace(os.sep, '/')
        # Check if the path is contained in any mapped target path
        for target_path, link_path in self._path_mappings.items():
            if normalized_link.startswith(target_path):
                # If the path starts with the target path, replace with actual path
                mapped_path = normalized_link.replace(target_path, link_path, 1)
                return mapped_path
        return link_path

    def _init_lora_paths(self) -> List[str]:
        """Initialize and validate LoRA paths from ComfyUI settings"""
        try:
            raw_paths = folder_paths.get_folder_paths("loras")
            
            # Normalize and resolve symlinks, store mapping from resolved -> original
            path_map = {}
            for path in raw_paths:
                if os.path.exists(path):
                    real_path = os.path.normpath(os.path.realpath(path)).replace(os.sep, '/')
                    path_map[real_path] = path_map.get(real_path, path.replace(os.sep, "/"))  # preserve first seen
            
            # Now sort and use only the deduplicated real paths
            unique_paths = sorted(path_map.values(), key=lambda p: p.lower())
            logger.info("Found LoRA roots:" + ("\n - " + "\n - ".join(unique_paths) if unique_paths else "[]"))
            
            if not unique_paths:
                logger.warning("No valid loras folders found in ComfyUI configuration")
                return []
            
            for original_path in unique_paths:
                real_path = os.path.normpath(os.path.realpath(original_path)).replace(os.sep, '/')
                if real_path != original_path:
                    self.add_path_mapping(original_path, real_path)
            
            return unique_paths
        except Exception as e:
            logger.warning(f"Error initializing LoRA paths: {e}")
            return []

    def _init_checkpoint_paths(self) -> List[str]:
        """Initialize and validate checkpoint paths from ComfyUI settings"""
        try:
            # Get checkpoint paths from folder_paths
            raw_checkpoint_paths = folder_paths.get_folder_paths("checkpoints")
            raw_unet_paths = folder_paths.get_folder_paths("unet")
            
            # Normalize and resolve symlinks for checkpoints, store mapping from resolved -> original
            checkpoint_map = {}
            for path in raw_checkpoint_paths:
                if os.path.exists(path):
                    real_path = os.path.normpath(os.path.realpath(path)).replace(os.sep, '/')
                    checkpoint_map[real_path] = checkpoint_map.get(real_path, path.replace(os.sep, "/"))  # preserve first seen
            
            # Normalize and resolve symlinks for unet, store mapping from resolved -> original
            unet_map = {}
            for path in raw_unet_paths:
                if os.path.exists(path):
                    real_path = os.path.normpath(os.path.realpath(path)).replace(os.sep, '/')
                    unet_map[real_path] = unet_map.get(real_path, path.replace(os.sep, "/"))  # preserve first seen
            
            # Merge both maps and deduplicate by real path
            merged_map = {}
            for real_path, orig_path in {**checkpoint_map, **unet_map}.items():
                if real_path not in merged_map:
                    merged_map[real_path] = orig_path

            # Now sort and use only the deduplicated real paths
            unique_paths = sorted(merged_map.values(), key=lambda p: p.lower())
            
            # Split back into checkpoints and unet roots for class properties
            self.checkpoints_roots = [p for p in unique_paths if p in checkpoint_map.values()]
            self.unet_roots = [p for p in unique_paths if p in unet_map.values()]
            
            all_paths = unique_paths
            
            logger.info("Found checkpoint roots:" + ("\n - " + "\n - ".join(all_paths) if all_paths else "[]"))
            
            if not all_paths:
                logger.warning("No valid checkpoint folders found in ComfyUI configuration")
                return []
            
            # Initialize path mappings
            for original_path in all_paths:
                real_path = os.path.normpath(os.path.realpath(original_path)).replace(os.sep, '/')
                if real_path != original_path:
                    self.add_path_mapping(original_path, real_path)
            
            return all_paths
        except Exception as e:
            logger.warning(f"Error initializing checkpoint paths: {e}")
            return []

    def _init_embedding_paths(self) -> List[str]:
        """Initialize and validate embedding paths from ComfyUI settings"""
        try:
            raw_paths = folder_paths.get_folder_paths("embeddings")
            
            # Normalize and resolve symlinks, store mapping from resolved -> original
            path_map = {}
            for path in raw_paths:
                if os.path.exists(path):
                    real_path = os.path.normpath(os.path.realpath(path)).replace(os.sep, '/')
                    path_map[real_path] = path_map.get(real_path, path.replace(os.sep, "/"))  # preserve first seen
            
            # Now sort and use only the deduplicated real paths
            unique_paths = sorted(path_map.values(), key=lambda p: p.lower())
            logger.info("Found embedding roots:" + ("\n - " + "\n - ".join(unique_paths) if unique_paths else "[]"))
            
            if not unique_paths:
                logger.warning("No valid embeddings folders found in ComfyUI configuration")
                return []
            
            for original_path in unique_paths:
                real_path = os.path.normpath(os.path.realpath(original_path)).replace(os.sep, '/')
                if real_path != original_path:
                    self.add_path_mapping(original_path, real_path)
            
            return unique_paths
        except Exception as e:
            logger.warning(f"Error initializing embedding paths: {e}")
            return []

    def get_preview_static_url(self, preview_path: str) -> str:
        """Convert local preview path to static URL"""
        if not preview_path:
            return ""
        
        real_path = os.path.realpath(preview_path).replace(os.sep, '/')

        for path, route in self._route_mappings.items():
            if real_path.startswith(path):
                relative_path = os.path.relpath(real_path, path).replace(os.sep, '/')
                safe_parts = [urllib.parse.quote(part) for part in relative_path.split('/')]
                safe_path = '/'.join(safe_parts)
                return f'{route}/{safe_path}'

        return ""

# Global config instance
config = Config()
