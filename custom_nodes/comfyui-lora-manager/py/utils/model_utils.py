from typing import Optional

# Base model mapping based on version string
BASE_MODEL_MAPPING = {
    "sd_1.5": "SD 1.5",
    "sd-v1-5": "SD 1.5",
    "sd-v2-1": "SD 2.1",
    "sdxl": "SDXL 1.0",
    "sd-v2": "SD 2.0",
    "flux1": "Flux.1 D",
    "flux.1 d": "Flux.1 D",
    "illustrious": "Illustrious",
    "il": "Illustrious",
    "pony": "Pony",
    "Hunyuan Video": "Hunyuan Video"
}

def determine_base_model(version_string: Optional[str]) -> str:
    """Determine base model from version string in safetensors metadata"""
    if not version_string:
        return "Unknown"
    
    version_lower = version_string.lower()
    for key, value in BASE_MODEL_MAPPING.items():
        if key in version_lower:
            return value
    
    # TODO: Add more base model mappings
    return version_string 