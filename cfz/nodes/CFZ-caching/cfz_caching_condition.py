import os
import torch
import hashlib
import folder_paths
from pathlib import Path

CACHE_DIR = os.path.join(folder_paths.output_directory, "cfz_conditioning_cache")

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

def compare_revision(target_revision):
    """Simple version check - defaults to supporting lazy loading"""
    try:
        import comfy
        if hasattr(comfy, 'model_management') and hasattr(comfy.model_management, 'get_torch_device'):
            return True
    except:
        pass
    return False

lazy_options = {"lazy": True} if compare_revision(2543) else {}

class save_conditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),  # Removed lazy_options
                "cache_name": ("STRING", {"default": "my_conditioning"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "save_conditioning"
    CATEGORY     = "CFZ Save-Load Conditioning"

    def save_conditioning(self, conditioning, cache_name):
        """Save conditioning to cache with custom name"""
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        if not cache_name.strip():
            raise ValueError("Cache name cannot be empty")
        
        sanitized_name = self.sanitize_filename(cache_name)
        file_path = os.path.join(CACHE_DIR, f"{sanitized_name}.pt")
        file_path = self._resolve_path(file_path)

        print(f"[CFZ Save] Processed info:")
        print(f"  - sanitized name: '{sanitized_name}'")
        print(f"  - file path: {file_path}")

        # Check if conditioning is provided
        if conditioning is None:
            print("[CFZ Save] ✗ ERROR: Conditioning is None!")
            print("[CFZ Save] This suggests the CLIP Text Encode node is not connected properly")
            print("[CFZ Save] Or there's an issue with the node execution order")
            raise ValueError(f"No conditioning input provided for cache '{sanitized_name}'. Please connect conditioning input.")

        try:
            print(f"[CFZ Save] Attempting to save conditioning...")
            torch.save(conditioning, file_path)
            print(f"[CFZ Save] ✓ Successfully saved: {sanitized_name}.pt")
        except Exception as e:
            print(f"[CFZ Save] ✗ Error saving: {e}")
            raise ValueError(f"Failed to save conditioning '{sanitized_name}': {str(e)}")
        
        return (conditioning,)

    def sanitize_filename(self, filename):
        """Remove invalid characters from filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        filename = filename.strip(' .')
        
        if not filename:
            filename = "unnamed_conditioning"
            
        return filename

    def _resolve_path(self, path_str):
        """Resolve file path using ComfyUI's path system"""
        try:
            return Path(folder_paths.get_annotated_filepath(str(path_str)))
        except:
            return Path(path_str)

    @classmethod
    def IS_CHANGED(cls, conditioning, cache_name):
        return hashlib.sha256(f"{cache_name}".encode('utf-8')).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, conditioning, cache_name):
        """Validate inputs before execution"""
        if not cache_name.strip():
            return "Cache name cannot be empty"
        return True


class load_conditioning:
    @classmethod
    def INPUT_TYPES(cls):
        cached_files = cls.get_cached_files()
        
        return {
            "required": {
                "cache_name": (cached_files, {"default": cached_files[0] if cached_files else ""}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "load_conditioning"
    CATEGORY     = "CFZ Save-Load Conditioning"

    @classmethod
    def get_cached_files(cls):
        """Get list of available cached conditioning files"""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_files = []
            
            if not os.path.exists(CACHE_DIR):
                print(f"[CFZ Load] Cache directory doesn't exist: {CACHE_DIR}")
                return ["no_cache_directory"]
            
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.pt'):
                    cache_name = filename[:-3]
                    cache_files.append(cache_name)
            
            cache_files.sort()
            
            if cache_files:
                # print(f"[CFZ Load] Found {len(cache_files)} cached files")
                return cache_files
            else:
                print("[CFZ Load] No cache files found")
                return ["no_cache_files_found"]
            
        except Exception as e:
            print(f"[CFZ Load] Error reading cache directory: {e}")
            return ["error_reading_cache"]

    def load_conditioning(self, cache_name):
        """Load conditioning from selected cached file"""
        # print(f"[CFZ Load] Loading conditioning:")
        # print(f"  - cache_name: '{cache_name}'")
        
        if cache_name in ["no_cache_files_found", "error_reading_cache", "no_cache_directory", ""]:
            raise ValueError("No valid cached conditioning file selected")
        
        file_path = os.path.join(CACHE_DIR, f"{cache_name}.pt")
        file_path = self._resolve_path(file_path)
        
        # print(f"  - file path: {file_path}")
        # print(f"  - file exists: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            raise ValueError(f"Cached conditioning not found: {cache_name}.pt")
        
        try:
            cached_tensor = torch.load(file_path, map_location='cpu')
            print(f"[CFZ Load Cached Conditioning] ✓ Successfully loaded: {cache_name}.pt")
            return (cached_tensor,)
        except Exception as e:
            print(f"[CFZ Load] ✗ Error loading: {e}")
            raise ValueError(f"Error loading cached conditioning '{cache_name}': {str(e)}")

    def _resolve_path(self, path_str):
        """Resolve file path using ComfyUI's path system"""
        try:
            return Path(folder_paths.get_annotated_filepath(str(path_str)))
        except:
            return Path(path_str)

    @classmethod
    def IS_CHANGED(cls, cache_name):
        return cache_name

    @classmethod
    def VALIDATE_INPUTS(cls, cache_name):
        """Validate inputs before execution"""
        if cache_name in ["no_cache_files_found", "error_reading_cache", "no_cache_directory", ""]:
            return "No cached conditioning files available"
        
        cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pt")
        if not os.path.exists(cache_path):
            return f"Selected cache file does not exist: {cache_name}.pt"
        
        return True

class CFZ_PrintMarker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {"default": "Reached this step!", "multiline": True}),
            },
            "optional": {
                "trigger": (any_type, {}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = (any_type,)  # Pass through whatever was received
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "CFZ Utils/Debug"

    def run(self, message, trigger=None, unique_id=None, extra_pnginfo=None):
        print(f"\n[🔔 CFZ Marker] {message}\n")
        return (trigger,)
