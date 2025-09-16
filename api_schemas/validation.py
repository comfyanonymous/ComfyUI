"""
JSON Schema validation utilities for ComfyUI API
"""
import json
import os
import logging
from typing import Dict, Any, Tuple, Optional

try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception  # Fallback for type hints


def load_prompt_schema() -> Optional[Dict[str, Any]]:
    """
    Load the prompt format JSON schema from file.
    
    Returns:
        Dict containing the schema, or None if not found/invalid
    """
    schema_path = os.path.join(os.path.dirname(__file__), "prompt_format.json")
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load prompt schema: {e}")
        return None


def validate_prompt_format(data: Dict[str, Any], warn_only: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate prompt data against the JSON schema.
    
    Args:
        data: The prompt data to validate
        warn_only: If True, log warnings instead of raising errors
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not JSONSCHEMA_AVAILABLE:
        if warn_only:
            logging.debug("jsonschema not available, skipping schema validation")
        return True, None
    
    schema = load_prompt_schema()
    if schema is None:
        if warn_only:
            logging.debug("Could not load schema, skipping validation")
        return True, None
    
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        error_msg = f"Prompt format validation failed: {e.message}"
        if e.path:
            error_msg += f" at path: {'.'.join(str(p) for p in e.path)}"
        
        if warn_only:
            logging.warning(f"Schema validation warning: {error_msg}")
            return True, error_msg  # Still return True for warnings
        else:
            return False, error_msg


def get_schema_info() -> Dict[str, Any]:
    """
    Get information about the schema validation capability.
    
    Returns:
        Dict containing schema validation status and info
    """
    info = {
        "jsonschema_available": JSONSCHEMA_AVAILABLE,
        "schema_loaded": load_prompt_schema() is not None,
    }
    
    if JSONSCHEMA_AVAILABLE:
        info["jsonschema_version"] = getattr(jsonschema, "__version__", "unknown")
    
    return info
