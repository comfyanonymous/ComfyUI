import logging
from typing import List

from ..utils.models import CheckpointMetadata
from ..config import config
from .model_scanner import ModelScanner
from .model_hash_index import ModelHashIndex

logger = logging.getLogger(__name__)

class CheckpointScanner(ModelScanner):
    """Service for scanning and managing checkpoint files"""
    
    def __init__(self):
        # Define supported file extensions
        file_extensions = {'.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft', '.gguf'}
        super().__init__(
            model_type="checkpoint",
            model_class=CheckpointMetadata,
            file_extensions=file_extensions,
            hash_index=ModelHashIndex()
        )

    def adjust_metadata(self, metadata, file_path, root_path):
        if hasattr(metadata, "model_type"):
            if root_path in config.checkpoints_roots:
                metadata.model_type = "checkpoint"
            elif root_path in config.unet_roots:
                metadata.model_type = "diffusion_model"
        return metadata

    def get_model_roots(self) -> List[str]:
        """Get checkpoint root directories"""
        return config.base_models_roots