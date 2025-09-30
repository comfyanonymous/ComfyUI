import os
import torch
import hashlib
import folder_paths
from pathlib import Path
import time
from datetime import datetime

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

# Global timer storage - shared across all marker instances
TIMER_STORAGE = {}

class save_conditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "cache_name": ("STRING", {"default": "my_conditioning"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "save_conditioning"
    CATEGORY     = "CFZ/conditioning"

    def save_conditioning(self, conditioning, cache_name):
        """Save conditioning to cache with custom name, supporting subdirectories"""
        
        if not cache_name.strip():
            raise ValueError("Cache name cannot be empty")
        
        # Normalize path separators (handle both \ and /)
        cache_name = cache_name.replace('\\', os.sep).replace('/', os.sep)
        
        # Split into directory and filename
        path_parts = cache_name.split(os.sep)
        sanitized_parts = [self.sanitize_filename(part) for part in path_parts]
        
        # Reconstruct the relative path
        relative_path = os.path.join(*sanitized_parts)
        
        # Create full file path
        full_path = os.path.join(CACHE_DIR, relative_path)
        
        # Extract directory and ensure it exists
        directory = os.path.dirname(full_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            print(f"[CFZ Save] Created/verified directory: {directory}")
        else:
            os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Add .pt extension if not present
        if not full_path.endswith('.pt'):
            full_path += '.pt'
        
        file_path = self._resolve_path(full_path)

        print(f"[CFZ Save] Processed info:")
        print(f"  - original name: '{cache_name}'")
        print(f"  - sanitized path: '{relative_path}'")
        print(f"  - full file path: {file_path}")

        # Check if conditioning is provided
        if conditioning is None:
            print("[CFZ Save] ❌ ERROR: Conditioning is None!")
            print("[CFZ Save] This suggests the CLIP Text Encode node is not connected properly")
            print("[CFZ Save] Or there's an issue with the node execution order")
            raise ValueError(f"No conditioning input provided for cache '{cache_name}'. Please connect conditioning input.")

        try:
            print(f"[CFZ Save] Attempting to save conditioning...")
            torch.save(conditioning, file_path)
            print(f"[CFZ Save] ✅ Successfully saved: {relative_path}.pt")
        except Exception as e:
            print(f"[CFZ Save] ❌ Error saving: {e}")
            raise ValueError(f"Failed to save conditioning '{cache_name}': {str(e)}")
        
        return (conditioning,)

    def sanitize_filename(self, filename):
        """Remove invalid characters from filename/directory name"""
        invalid_chars = '<>:"|?*'  # Removed / and \ to allow path separators
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        filename = filename.strip(' .')
        
        if not filename:
            filename = "unnamed"
            
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
    CATEGORY     = "CFZ/conditioning"

    @classmethod
    def get_cached_files(cls):
        """Get list of available cached conditioning files, including subdirectories"""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_files = []
            
            if not os.path.exists(CACHE_DIR):
                print(f"[CFZ Load] Cache directory doesn't exist: {CACHE_DIR}")
                return ["no_cache_directory"]
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(CACHE_DIR):
                for filename in files:
                    if filename.endswith('.pt'):
                        # Get relative path from CACHE_DIR
                        full_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(full_path, CACHE_DIR)
                        
                        # Remove .pt extension
                        cache_name = relative_path[:-3]
                        
                        # Normalize path separators for display
                        cache_name = cache_name.replace(os.sep, '/')
                        
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
        """Load conditioning from selected cached file, supporting subdirectories"""
        
        if cache_name in ["no_cache_files_found", "error_reading_cache", "no_cache_directory", ""]:
            raise ValueError("No valid cached conditioning file selected")
        
        # Normalize path separators
        cache_name = cache_name.replace('/', os.sep).replace('\\', os.sep)
        
        # Build full path
        file_path = os.path.join(CACHE_DIR, cache_name)
        
        # Add .pt extension if not present
        if not file_path.endswith('.pt'):
            file_path += '.pt'
        
        file_path = self._resolve_path(file_path)
        
        print(f"[CFZ Load] Loading conditioning from: {cache_name}")
        
        if not os.path.exists(file_path):
            raise ValueError(f"Cached conditioning not found: {cache_name}.pt")
        
        try:
            cached_tensor = torch.load(file_path, map_location='cpu')
            print(f"[CFZ Load Cached Conditioning] ✅ Successfully loaded: {cache_name}.pt")
            return (cached_tensor,)
        except Exception as e:
            print(f"[CFZ Load] ❌ Error loading: {e}")
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
        
        # Normalize path
        cache_name_normalized = cache_name.replace('/', os.sep).replace('\\', os.sep)
        cache_path = os.path.join(CACHE_DIR, cache_name_normalized)
        
        if not cache_path.endswith('.pt'):
            cache_path += '.pt'
        
        if not os.path.exists(cache_path):
            return f"Selected cache file does not exist: {cache_name}.pt"
        
        return True

class CFZ_PrintMarker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {"default": "Reached this step!", "multiline": True}),
                "timer_name": ("STRING", {"default": "workflow_timer"}),
                "is_start_point": ("BOOLEAN", {"default": False}),
                "is_end_point": ("BOOLEAN", {"default": False}),
                "show_current_time": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "trigger": (any_type, {}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "CFZ/utils/debug"

    def run(self, message, timer_name="workflow_timer", is_start_point=False, is_end_point=False, 
            show_current_time=True, trigger=None, unique_id=None, extra_pnginfo=None):
        
        current_time = time.time()
        current_timestamp = datetime.fromtimestamp(current_time).strftime("%H:%M:%S.%f")[:-3]
        
        # Handle timer logic
        timer_info = ""
        
        if is_start_point:
            TIMER_STORAGE[timer_name] = current_time
            timer_info = f" | TIMER START: '{timer_name}'"
        
        if is_end_point:
            if timer_name in TIMER_STORAGE:
                start_time = TIMER_STORAGE[timer_name]
                elapsed = current_time - start_time
                
                # Format elapsed time nicely
                if elapsed < 1:
                    elapsed_str = f"{elapsed*1000:.1f}ms"
                elif elapsed < 60:
                    elapsed_str = f"{elapsed:.2f}s"
                else:
                    minutes = int(elapsed // 60)
                    seconds = elapsed % 60
                    elapsed_str = f"{minutes}m {seconds:.2f}s"
                
                timer_info = f" | TIMER END: '{timer_name}' - Elapsed: {elapsed_str}"
                
                # Clean up the timer
                del TIMER_STORAGE[timer_name]
            else:
                timer_info = f" | TIMER ERROR: No start point found for '{timer_name}'"
        
        # Build the output message
        output_parts = []
        
        if show_current_time:
            output_parts.append(f"[{current_timestamp}]")
        
        output_parts.append(f"CFZ Marker")
        output_parts.append(message)
        
        if timer_info:
            output_parts.append(timer_info)
        
        final_message = " ".join(output_parts)
        print(f"\n{final_message}\n")
        
        return (trigger,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute to ensure timing is accurate
        return float("NaN")
