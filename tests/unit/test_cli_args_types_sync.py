
import pytest
import sys
from unittest.mock import patch
from comfy import cli_args
from comfy import cli_args_types

def test_cli_args_types_completeness():
    """
    Verify that cli_args_types.Configuration matches the actual arguments defined in cli_args.
    """
    # Get actual configuration defaults
    # Parse with empty args to get defaults
    parser = cli_args._create_parser()
    with patch.object(parser, 'parse_known_args_with_config_files', return_value=(parser.parse_known_args([])[0], [], [])):
        actual_config = cli_args._parse_args(parser, args_parsing=True)

    # Get type definition
    type_config = cli_args_types.Configuration()

    actual_keys = set(actual_config.keys())
    type_keys = set(type_config.keys())

    # Check for missing keys in type definition
    missing_in_types = actual_keys - type_keys
    assert not missing_in_types, f"Keys present in actual config but missing in types: {missing_in_types}"

    # Check for extra keys in type definition (warning level usually, but here strict)
    # We allow exact match or superset if types has deprecated stuff? 
    # But for now let's assume close parity.
    # extra_in_types = type_keys - actual_keys
    # if extra_in_types:
    #     print(f"WARNING: Keys in types but not in actual: {extra_in_types}")
    #     # Not asserting here as sometimes types carry legacy or helper fields
    
    # Check specific types if needed.
    # Verify new fields exist
    assert hasattr(type_config, "disable_auto_launch")
    assert hasattr(type_config, "cache_ram")
    assert hasattr(type_config, "enable_manager")
    assert hasattr(type_config, "disable_manager_ui")
    assert hasattr(type_config, "enable_manager_legacy_ui")
    assert hasattr(type_config, "disable_async_offload")
    assert hasattr(type_config, "disable_pinned_memory")

    # Verify type mismatches we fixed
    # async_offload should be Optional[int] in annotation, defaulting to None in value
    # But runtime value from default_configuration might be None (or 2 if set)
    # Configuration init defaults it to None.
    assert type_config.async_offload is None 
