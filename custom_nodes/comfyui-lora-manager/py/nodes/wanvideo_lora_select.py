from comfy.comfy_types import IO # type: ignore
import folder_paths # type: ignore
from ..utils.utils import get_lora_info
from .utils import FlexibleOptionalInputType, any_type, get_loras_list
import logging

logger = logging.getLogger(__name__)

class WanVideoLoraSelect:
    NAME = "WanVideo Lora Select (LoraManager)"
    CATEGORY = "Lora Manager/stackers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load the LORA model with less VRAM usage, slower loading"}),
                "text": (IO.STRING, {
                    "multiline": True, 
                    "dynamicPrompts": True, 
                    "tooltip": "Format: <lora:lora_name:strength> separated by spaces or punctuation",
                    "placeholder": "LoRA syntax input: <lora:name:strength>"
                }),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("WANVIDLORA", IO.STRING, IO.STRING)
    RETURN_NAMES = ("lora", "trigger_words", "active_loras")
    FUNCTION = "process_loras"
    
    def process_loras(self, text, low_mem_load=False, **kwargs):
        loras_list = []
        all_trigger_words = []
        active_loras = []
        
        # Process existing prev_lora if available
        prev_lora = kwargs.get('prev_lora', None)
        if prev_lora is not None:
            loras_list.extend(prev_lora)
        
        # Get blocks if available
        blocks = kwargs.get('blocks', {})
        selected_blocks = blocks.get("selected_blocks", {})
        layer_filter = blocks.get("layer_filter", "")
        
        # Process loras from kwargs with support for both old and new formats
        loras_from_widget = get_loras_list(kwargs)
        for lora in loras_from_widget:
            if not lora.get('active', False):
                continue
                
            lora_name = lora['name']
            model_strength = float(lora['strength'])
            clip_strength = float(lora.get('clipStrength', model_strength))
            
            # Get lora path and trigger words
            lora_path, trigger_words = get_lora_info(lora_name)
            
            # Create lora item for WanVideo format
            lora_item = {
                "path": folder_paths.get_full_path("loras", lora_path),
                "strength": model_strength,
                "name": lora_path.split(".")[0],
                "blocks": selected_blocks,
                "layer_filter": layer_filter,
                "low_mem_load": low_mem_load,
            }
            
            # Add to list and collect active loras
            loras_list.append(lora_item)
            active_loras.append((lora_name, model_strength, clip_strength))
            
            # Add trigger words to collection
            all_trigger_words.extend(trigger_words)
        
        # Format trigger_words for output
        trigger_words_text = ",, ".join(all_trigger_words) if all_trigger_words else ""
        
        # Format active_loras for output
        formatted_loras = []
        for name, model_strength, clip_strength in active_loras:
            if abs(model_strength - clip_strength) > 0.001:
                # Different model and clip strengths
                formatted_loras.append(f"<lora:{name}:{str(model_strength).strip()}:{str(clip_strength).strip()}>")
            else:
                # Same strength for both
                formatted_loras.append(f"<lora:{name}:{str(model_strength).strip()}>")
                
        active_loras_text = " ".join(formatted_loras)

        return (loras_list, trigger_words_text, active_loras_text)
