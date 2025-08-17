from comfy.comfy_types import IO # type: ignore
import os
from ..utils.utils import get_lora_info
from .utils import FlexibleOptionalInputType, any_type, extract_lora_name, get_loras_list

import logging

logger = logging.getLogger(__name__)

class LoraStacker:
    NAME = "Lora Stacker (LoraManager)"
    CATEGORY = "Lora Manager/stackers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (IO.STRING, {
                    "multiline": True, 
                    "dynamicPrompts": True, 
                    "tooltip": "Format: <lora:lora_name:strength> separated by spaces or punctuation",
                    "placeholder": "LoRA syntax input: <lora:name:strength>"
                }),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("LORA_STACK", IO.STRING, IO.STRING)
    RETURN_NAMES = ("LORA_STACK", "trigger_words", "active_loras")
    FUNCTION = "stack_loras"
    
    def stack_loras(self, text, **kwargs):
        """Stacks multiple LoRAs based on the kwargs input without loading them."""
        stack = []
        active_loras = []
        all_trigger_words = []
        
        # Process existing lora_stack if available
        lora_stack = kwargs.get('lora_stack', None)
        if (lora_stack):
            stack.extend(lora_stack)
            # Get trigger words from existing stack entries
            for lora_path, _, _ in lora_stack:
                lora_name = extract_lora_name(lora_path)
                _, trigger_words = get_lora_info(lora_name)
                all_trigger_words.extend(trigger_words)
        
        # Process loras from kwargs with support for both old and new formats
        loras_list = get_loras_list(kwargs)
        for lora in loras_list:
            if not lora.get('active', False):
                continue
                
            lora_name = lora['name']
            model_strength = float(lora['strength'])
            # Get clip strength - use model strength as default if not specified
            clip_strength = float(lora.get('clipStrength', model_strength))
            
            # Get lora path and trigger words
            lora_path, trigger_words = get_lora_info(lora_name)
            
            # Add to stack without loading
            # replace '/' with os.sep to avoid different OS path format
            stack.append((lora_path.replace('/', os.sep), model_strength, clip_strength))
            active_loras.append((lora_name, model_strength, clip_strength))
            
            # Add trigger words to collection
            all_trigger_words.extend(trigger_words)
        
        # use ',, ' to separate trigger words for group mode
        trigger_words_text = ",, ".join(all_trigger_words) if all_trigger_words else ""
        
        # Format active_loras with support for both formats
        formatted_loras = []
        for name, model_strength, clip_strength in active_loras:
            if abs(model_strength - clip_strength) > 0.001:
                # Different model and clip strengths
                formatted_loras.append(f"<lora:{name}:{str(model_strength).strip()}:{str(clip_strength).strip()}>")
            else:
                # Same strength for both
                formatted_loras.append(f"<lora:{name}:{str(model_strength).strip()}>")
                
        active_loras_text = " ".join(formatted_loras)

        return (stack, trigger_words_text, active_loras_text)
