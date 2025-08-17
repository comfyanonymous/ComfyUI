import logging
from typing import List

from ..utils.models import LoraMetadata
from ..config import config
from .model_scanner import ModelScanner
from .model_hash_index import ModelHashIndex  # Changed from LoraHashIndex to ModelHashIndex
import sys

logger = logging.getLogger(__name__)

class LoraScanner(ModelScanner):
    """Service for scanning and managing LoRA files"""
    
    def __init__(self):
        # Define supported file extensions
        file_extensions = {'.safetensors'}
        
        # Initialize parent class with ModelHashIndex
        super().__init__(
            model_type="lora",
            model_class=LoraMetadata, 
            file_extensions=file_extensions,
            hash_index=ModelHashIndex()  # Changed from LoraHashIndex to ModelHashIndex
        )
    
    def get_model_roots(self) -> List[str]:
        """Get lora root directories"""
        return config.loras_roots

    async def diagnose_hash_index(self):
        """Diagnostic method to verify hash index functionality"""
        print("\n\n*** DIAGNOSING LORA HASH INDEX ***\n\n", file=sys.stderr)
        
        # First check if the hash index has any entries
        if hasattr(self, '_hash_index'):
            index_entries = len(self._hash_index._hash_to_path)
            print(f"Hash index has {index_entries} entries", file=sys.stderr)
            
            # Print a few example entries if available
            if index_entries > 0:
                print("\nSample hash index entries:", file=sys.stderr)
                count = 0
                for hash_val, path in self._hash_index._hash_to_path.items():
                    if count < 5:  # Just show the first 5
                        print(f"Hash: {hash_val[:8]}... -> Path: {path}", file=sys.stderr)
                        count += 1
                    else:
                        break
        else:
            print("Hash index not initialized", file=sys.stderr)
        
        # Try looking up by a known hash for testing
        if not hasattr(self, '_hash_index') or not self._hash_index._hash_to_path:
            print("No hash entries to test lookup with", file=sys.stderr)
            return
        
        test_hash = next(iter(self._hash_index._hash_to_path.keys()))
        test_path = self._hash_index.get_path(test_hash)
        print(f"\nTest lookup by hash: {test_hash[:8]}... -> {test_path}", file=sys.stderr)
        
        # Also test reverse lookup
        test_hash_result = self._hash_index.get_hash(test_path)
        print(f"Test reverse lookup: {test_path} -> {test_hash_result[:8]}...\n\n", file=sys.stderr)

