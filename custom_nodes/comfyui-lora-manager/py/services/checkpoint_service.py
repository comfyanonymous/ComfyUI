import os
import logging
from typing import Dict, List, Optional

from .base_model_service import BaseModelService
from ..utils.models import CheckpointMetadata
from ..config import config
from ..utils.routes_common import ModelRouteUtils

logger = logging.getLogger(__name__)

class CheckpointService(BaseModelService):
    """Checkpoint-specific service implementation"""
    
    def __init__(self, scanner):
        """Initialize Checkpoint service
        
        Args:
            scanner: Checkpoint scanner instance
        """
        super().__init__("checkpoint", scanner, CheckpointMetadata)
    
    async def format_response(self, checkpoint_data: Dict) -> Dict:
        """Format Checkpoint data for API response"""
        return {
            "model_name": checkpoint_data["model_name"],
            "file_name": checkpoint_data["file_name"],
            "preview_url": config.get_preview_static_url(checkpoint_data.get("preview_url", "")),
            "preview_nsfw_level": checkpoint_data.get("preview_nsfw_level", 0),
            "base_model": checkpoint_data.get("base_model", ""),
            "folder": checkpoint_data["folder"],
            "sha256": checkpoint_data.get("sha256", ""),
            "file_path": checkpoint_data["file_path"].replace(os.sep, "/"),
            "file_size": checkpoint_data.get("size", 0),
            "modified": checkpoint_data.get("modified", ""),
            "tags": checkpoint_data.get("tags", []),
            "modelDescription": checkpoint_data.get("modelDescription", ""),
            "from_civitai": checkpoint_data.get("from_civitai", True),
            "notes": checkpoint_data.get("notes", ""),
            "model_type": checkpoint_data.get("model_type", "checkpoint"),
            "favorite": checkpoint_data.get("favorite", False),
            "civitai": ModelRouteUtils.filter_civitai_data(checkpoint_data.get("civitai", {}))
        }
    
    def find_duplicate_hashes(self) -> Dict:
        """Find Checkpoints with duplicate SHA256 hashes"""
        return self.scanner._hash_index.get_duplicate_hashes()
    
    def find_duplicate_filenames(self) -> Dict:
        """Find Checkpoints with conflicting filenames"""
        return self.scanner._hash_index.get_duplicate_filenames()