"""
Feature flags module for ComfyUI WebSocket protocol negotiation.

This module handles capability negotiation between frontend and backend,
allowing graceful protocol evolution while maintaining backward compatibility.
"""

from typing import Any, Dict

from comfy.cli_args import args

# Default server capabilities
SERVER_FEATURE_FLAGS: Dict[str, Any] = {
    "supports_preview_metadata": True,
    "max_upload_size": args.max_upload_size * 1024 * 1024, # Convert MB to bytes
}


def get_connection_feature(
    sockets_metadata: Dict[str, Dict[str, Any]],
    sid: str,
    feature_name: str,
    default: Any = False
) -> Any:
    """
    Get a feature flag value for a specific connection.

    Args:
        sockets_metadata: Dictionary of socket metadata
        sid: Session ID of the connection
        feature_name: Name of the feature to check
        default: Default value if feature not found

    Returns:
        Feature value or default if not found
    """
    if sid not in sockets_metadata:
        return default

    return sockets_metadata[sid].get("feature_flags", {}).get(feature_name, default)


def supports_feature(
    sockets_metadata: Dict[str, Dict[str, Any]],
    sid: str,
    feature_name: str
) -> bool:
    """
    Check if a connection supports a specific feature.

    Args:
        sockets_metadata: Dictionary of socket metadata
        sid: Session ID of the connection
        feature_name: Name of the feature to check

    Returns:
        Boolean indicating if feature is supported
    """
    return get_connection_feature(sockets_metadata, sid, feature_name, False) is True


def get_server_features() -> Dict[str, Any]:
    """
    Get the server's feature flags.

    Returns:
        Dictionary of server feature flags
    """
    return SERVER_FEATURE_FLAGS.copy()
