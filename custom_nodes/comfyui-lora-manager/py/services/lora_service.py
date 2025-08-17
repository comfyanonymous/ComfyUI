import os
import logging
from typing import Dict, List, Optional

from .base_model_service import BaseModelService
from ..utils.models import LoraMetadata
from ..config import config
from ..utils.routes_common import ModelRouteUtils

logger = logging.getLogger(__name__)

class LoraService(BaseModelService):
    """LoRA-specific service implementation"""
    
    def __init__(self, scanner):
        """Initialize LoRA service
        
        Args:
            scanner: LoRA scanner instance
        """
        super().__init__("lora", scanner, LoraMetadata)
    
    async def format_response(self, lora_data: Dict) -> Dict:
        """Format LoRA data for API response"""
        return {
            "model_name": lora_data["model_name"],
            "file_name": lora_data["file_name"],
            "preview_url": config.get_preview_static_url(lora_data.get("preview_url", "")),
            "preview_nsfw_level": lora_data.get("preview_nsfw_level", 0),
            "base_model": lora_data.get("base_model", ""),
            "folder": lora_data["folder"],
            "sha256": lora_data.get("sha256", ""),
            "file_path": lora_data["file_path"].replace(os.sep, "/"),
            "file_size": lora_data.get("size", 0),
            "modified": lora_data.get("modified", ""),
            "tags": lora_data.get("tags", []),
            "modelDescription": lora_data.get("modelDescription", ""),
            "from_civitai": lora_data.get("from_civitai", True),
            "usage_tips": lora_data.get("usage_tips", ""),
            "notes": lora_data.get("notes", ""),
            "favorite": lora_data.get("favorite", False),
            "civitai": ModelRouteUtils.filter_civitai_data(lora_data.get("civitai", {}))
        }
    
    async def _apply_specific_filters(self, data: List[Dict], **kwargs) -> List[Dict]:
        """Apply LoRA-specific filters"""
        # Handle first_letter filter for LoRAs
        first_letter = kwargs.get('first_letter')
        if first_letter:
            data = self._filter_by_first_letter(data, first_letter)
        
        return data
    
    def _filter_by_first_letter(self, data: List[Dict], letter: str) -> List[Dict]:
        """Filter data by first letter of model name
        
        Special handling:
        - '#': Numbers (0-9)
        - '@': Special characters (not alphanumeric)
        - '漢': CJK characters
        """
        filtered_data = []
        
        for lora in data:
            model_name = lora.get('model_name', '')
            if not model_name:
                continue
                
            first_char = model_name[0].upper()
            
            if letter == '#' and first_char.isdigit():
                filtered_data.append(lora)
            elif letter == '@' and not first_char.isalnum():
                # Special characters (not alphanumeric)
                filtered_data.append(lora)
            elif letter == '漢' and self._is_cjk_character(first_char):
                # CJK characters
                filtered_data.append(lora)
            elif letter.upper() == first_char:
                # Regular alphabet matching
                filtered_data.append(lora)
                
        return filtered_data
    
    def _is_cjk_character(self, char: str) -> bool:
        """Check if character is a CJK character"""
        # Define Unicode ranges for CJK characters
        cjk_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
            (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
            (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
            (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
            (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
            (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
            (0x30000, 0x3134F), # CJK Unified Ideographs Extension G
            (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
            (0x3300, 0x33FF),   # CJK Compatibility
            (0x3200, 0x32FF),   # Enclosed CJK Letters and Months
            (0x3100, 0x312F),   # Bopomofo
            (0x31A0, 0x31BF),   # Bopomofo Extended
            (0x3040, 0x309F),   # Hiragana
            (0x30A0, 0x30FF),   # Katakana
            (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
            (0xAC00, 0xD7AF),   # Hangul Syllables
            (0x1100, 0x11FF),   # Hangul Jamo
            (0xA960, 0xA97F),   # Hangul Jamo Extended-A
            (0xD7B0, 0xD7FF),   # Hangul Jamo Extended-B
        ]
        
        code_point = ord(char)
        return any(start <= code_point <= end for start, end in cjk_ranges)
    
    # LoRA-specific methods
    async def get_letter_counts(self) -> Dict[str, int]:
        """Get count of LoRAs for each letter of the alphabet"""
        cache = await self.scanner.get_cached_data()
        data = cache.raw_data
        
        # Define letter categories
        letters = {
            '#': 0,  # Numbers
            'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0,
            'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0,
            'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0,
            'Y': 0, 'Z': 0,
            '@': 0,  # Special characters
            '漢': 0   # CJK characters
        }
        
        # Count models for each letter
        for lora in data:
            model_name = lora.get('model_name', '')
            if not model_name:
                continue
                
            first_char = model_name[0].upper()
            
            if first_char.isdigit():
                letters['#'] += 1
            elif first_char in letters:
                letters[first_char] += 1
            elif self._is_cjk_character(first_char):
                letters['漢'] += 1
            elif not first_char.isalnum():
                letters['@'] += 1
                
        return letters
    
    async def get_lora_notes(self, lora_name: str) -> Optional[str]:
        """Get notes for a specific LoRA file"""
        cache = await self.scanner.get_cached_data()
        
        for lora in cache.raw_data:
            if lora['file_name'] == lora_name:
                return lora.get('notes', '')
        
        return None
    
    async def get_lora_trigger_words(self, lora_name: str) -> List[str]:
        """Get trigger words for a specific LoRA file"""
        cache = await self.scanner.get_cached_data()
        
        for lora in cache.raw_data:
            if lora['file_name'] == lora_name:
                civitai_data = lora.get('civitai', {})
                return civitai_data.get('trainedWords', [])
        
        return []
    
    async def get_lora_preview_url(self, lora_name: str) -> Optional[str]:
        """Get the static preview URL for a LoRA file"""
        cache = await self.scanner.get_cached_data()
        
        for lora in cache.raw_data:
            if lora['file_name'] == lora_name:
                preview_url = lora.get('preview_url')
                if preview_url:
                    return config.get_preview_static_url(preview_url)
        
        return None
    
    async def get_lora_civitai_url(self, lora_name: str) -> Dict[str, Optional[str]]:
        """Get the Civitai URL for a LoRA file"""
        cache = await self.scanner.get_cached_data()
        
        for lora in cache.raw_data:
            if lora['file_name'] == lora_name:
                civitai_data = lora.get('civitai', {})
                model_id = civitai_data.get('modelId')
                version_id = civitai_data.get('id')
                
                if model_id:
                    civitai_url = f"https://civitai.com/models/{model_id}"
                    if version_id:
                        civitai_url += f"?modelVersionId={version_id}"
                    
                    return {
                        'civitai_url': civitai_url,
                        'model_id': str(model_id),
                        'version_id': str(version_id) if version_id else None
                    }
        
        return {'civitai_url': None, 'model_id': None, 'version_id': None}
    
    def find_duplicate_hashes(self) -> Dict:
        """Find LoRAs with duplicate SHA256 hashes"""
        return self.scanner._hash_index.get_duplicate_hashes()
    
    def find_duplicate_filenames(self) -> Dict:
        """Find LoRAs with conflicting filenames"""
        return self.scanner._hash_index.get_duplicate_filenames()