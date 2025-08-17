import json
import re
from server import PromptServer # type: ignore
from .utils import FlexibleOptionalInputType, any_type
import logging

logger = logging.getLogger(__name__)


class TriggerWordToggle:
    NAME = "TriggerWord Toggle (LoraManager)"
    CATEGORY = "Lora Manager/utils"
    DESCRIPTION = "Toggle trigger words on/off"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, treats each group of trigger words as a single toggleable unit."
                }),
                "default_active": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Sets the default initial state (active or inactive) when trigger words are added."
                }),
            },
            "optional": FlexibleOptionalInputType(any_type),
            "hidden": {
                "id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_trigger_words",)
    FUNCTION = "process_trigger_words"

    def _get_toggle_data(self, kwargs, key='toggle_trigger_words'):
        """Helper to extract data from either old or new kwargs format"""
        if key not in kwargs:
            return None
            
        data = kwargs[key]
        # Handle new format: {'key': {'__value__': ...}}
        if isinstance(data, dict) and '__value__' in data:
            return data['__value__']
        # Handle old format: {'key': ...}
        else:
            return data

    def process_trigger_words(self, id, group_mode, default_active, **kwargs):
        # Handle both old and new formats for trigger_words
        trigger_words_data = self._get_toggle_data(kwargs, 'orinalMessage')
        trigger_words = trigger_words_data if isinstance(trigger_words_data, str) else ""
        
        filtered_triggers = trigger_words
        
        # Get toggle data with support for both formats
        trigger_data = self._get_toggle_data(kwargs, 'toggle_trigger_words')
        if trigger_data:
            try:
                # Convert to list if it's a JSON string
                if isinstance(trigger_data, str):
                    trigger_data = json.loads(trigger_data)
                
                # Create dictionaries to track active state of words or groups
                active_state = {item['text']: item.get('active', False) for item in trigger_data}
                
                if group_mode:
                    # Split by two or more consecutive commas to get groups
                    groups = re.split(r',{2,}', trigger_words)
                    # Remove leading/trailing whitespace from each group
                    groups = [group.strip() for group in groups]
                    
                    # Filter groups: keep those not in toggle_trigger_words or those that are active
                    filtered_groups = [group for group in groups if group not in active_state or active_state[group]]
                    
                    if filtered_groups:
                        filtered_triggers = ', '.join(filtered_groups)
                    else:
                        filtered_triggers = ""
                else:
                    # Original behavior for individual words mode
                    original_words = [word.strip() for word in trigger_words.split(',')]
                    # Filter out empty strings
                    original_words = [word for word in original_words if word]
                    filtered_words = [word for word in original_words if word not in active_state or active_state[word]]
                    
                    if filtered_words:
                        filtered_triggers = ', '.join(filtered_words)
                    else:
                        filtered_triggers = ""
                    
            except Exception as e:
                logger.error(f"Error processing trigger words: {e}")
            
        return (filtered_triggers,)