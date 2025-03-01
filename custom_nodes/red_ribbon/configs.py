from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from pydantic import BaseModel, Field
from functools import lru_cache



class Paths(BaseModel):
    THIS_FILE = Path(__file__).resolve()
    RED_RIBBON_DIR = THIS_FILE.parent
    CUSTOM_NODES_DIR = RED_RIBBON_DIR.parent
    COMFYUI_DIR = CUSTOM_NODES_DIR.parent
    LLM_OUTPUTS_DIR = COMFYUI_DIR / "output" / "red_ribbon_outputs"
    LLM_MODELS_DIR = COMFYUI_DIR / "models" / "llm_models"

    class Config:
        frozen = True  # Make the model immutable (read-only)


class SocialToolkitConfigs(BaseModel):
    """Configuration for High Level Architecture workflow"""
    approved_document_sources: list[str]
    llm_api_config: dict[str, Any]
    document_retrieval_threshold: int = 10
    relevance_threshold: float = 0.7
    output_format: str = "json"

    codebook: Optional[dict[str, Any]] = None
    document_retrieval: Optional[dict[str, Any]] = None
    llm_service: Optional[dict[str, Any]] = None
    top10_retrieval: Optional[dict[str, Any]] = None
    relevance_assessment: Optional[dict[str, Any]] = None
    prompt_decision_tree: Optional[dict[str, Any]] = None


class ConfigsBase(BaseModel):
    """Base model for configuration with read-only fields."""
    
    class Config:
        frozen = True  # Make the model immutable (read-only)


@lru_cache()
def get_config() -> 'Configs':
    """
    Load configuration from YAML files and cache the result.
    Returns a read-only Configs object.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load main configs
    config_path = os.path.join(base_dir, "configs.yaml")
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Load private configs (overrides main configs)
    private_config_path = os.path.join(base_dir, "private_configs.yaml")
    private_config_data = {}
    if os.path.exists(private_config_path):
        with open(private_config_path, 'r') as f:
            private_config_data = yaml.safe_load(f) or {}
    
    # Merge configs, with private taking precedence
    merged_config = {**config_data, **private_config_data}
    
    return Configs(**merged_config)


class Configs(ConfigsBase):
    """
    Configuration constants loaded from YAML files.
    All fields are read-only. 
    
    Loads from:
    - configs.yaml (base configuration)
    - private_configs.yaml (overrides base configuration)
    """
    # Add your configuration fields here with defaults
    # Example:
    API_URL: str = Field("http://localhost:8000", description="API URL")
    DEBUG_MODE: bool = Field(default=False, description="Enable debug mode")
    MAX_BATCH_SIZE: int = Field(default=4, description="Maximum batch size")
    MODEL_PATHS: Dict[str, str] = Field(default_factory=dict, description="Paths to models")
    CUSTOM_SETTINGS: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration settings")

    _paths: Paths = Field(default_factory=Paths)
    _socialtoolkit: SocialToolkitConfigs = Field(default_factory=SocialToolkitConfigs)  
    
    # Access the singleton instance through this class method
    @classmethod
    def get(cls) -> 'Configs':
        """Get the singleton instance of Configs."""
        return get_config()
    
    @property
    def paths(self) -> Paths:
        return self._paths
    
    @property
    def socialtoolkit(self) -> SocialToolkitConfigs:
        return self._socialtoolkit