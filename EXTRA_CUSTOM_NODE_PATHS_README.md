# Extra Custom Node Paths

This feature allows you to load custom nodes from multiple directories, similar to how extra model paths work in ComfyUI.

## Overview

ComfyUI now supports loading custom nodes from multiple directories through a YAML configuration file called `extra_custom_node_paths.yaml`. This is useful when you want to:

- Organize custom nodes in different locations
- Share custom nodes between multiple ComfyUI installations
- Keep custom nodes separate from the main ComfyUI directory

## Configuration File

Create a file called `extra_custom_node_paths.yaml` in your ComfyUI root directory:

```yaml
# Configuration for custom nodes
# The custom_nodes directory is always called "custom_nodes" and is located under base_path
# Each config section can only have one base_path

# Example configuration using base_path
# This will look for custom_nodes under the specified base_path
custom_nodes_example:
    base_path: /path/to/base1

# Another example with a different path
other_custom_nodes:
    base_path: /path/to/base

# You can add more custom node path configurations as needed
# additional_custom_nodes:
#     base_path: another/base/path
```

### Configuration Options

- **base_path**: The base directory path where the `custom_nodes` subdirectory is located
  - Can be relative or absolute paths
  - Relative paths are resolved relative to the YAML file location
  - The system automatically looks for a `custom_nodes` subdirectory under each base_path
```

## How It Works

1. **Automatic Loading**: ComfyUI automatically looks for `extra_custom_node_paths.yaml` in the root directory
2. **Path Resolution**: All base paths are resolved to absolute paths and validated
3. **Subdirectory Detection**: The system automatically appends `custom_nodes` to each base path
4. **Integration**: Custom node paths are added to the existing custom nodes system
5. **Compatibility**: Works seamlessly with existing custom node functionality

## Example Use Cases

### Shared Custom Nodes

```yaml
# Share custom nodes between multiple ComfyUI installations
shared_nodes:
    base_path: /shared/custom_nodes
```

### Relative Paths

```yaml
# Use relative paths for portable configurations
portable_config:
    base_path: ../shared_custom_nodes
```

## File Structure

The custom node directories should follow the standard ComfyUI custom node structure:

```
base_path/
└── custom_nodes/
    ├── node_name_1/
    │   ├── __init__.py
    │   └── nodes.py
    ├── node_name_2/
    │   ├── __init__.py
    │   └── nodes.py
    └── ...
```

## Troubleshooting

### Path Not Found
- Ensure the directory exists and is accessible
- Check file permissions
- Verify the path is correctly formatted in the YAML file
- Make sure there's a `custom_nodes` subdirectory under the specified base_path

### Custom Nodes Not Loading
- Check the ComfyUI console for error messages
- Verify the YAML syntax is correct
- Ensure the custom node directories contain valid Python modules

## Technical Details

- Custom node paths are loaded during ComfyUI startup
- Paths are validated for existence before being added
- The system maintains the order of paths (default paths first)
- Duplicate paths are automatically handled
- All paths are normalized and resolved to absolute paths
- Each config section can only specify one base_path

## Compatibility

This feature is compatible with:
- All existing custom node functionality
- The existing custom node loading system
- Command line arguments and configuration files
- All supported operating systems

## See Also

- [Extra Model Paths](../extra_model_paths.yaml) - Similar functionality for model paths
- [Custom Nodes Documentation](../custom_nodes/) - General custom node information
- [ComfyUI CLI Arguments](../comfy/cli_args.py) - Command line options
