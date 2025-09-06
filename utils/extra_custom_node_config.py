import os
import yaml
import folder_paths
import logging

def load_extra_custom_node_path_config(yaml_path):
    """
    Load extra custom node paths from a YAML configuration file.
    Similar to load_extra_path_config but specifically for custom nodes.
    Each config section can have one base_path, and custom_nodes will be looked for under that path.
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
    except Exception as e:
        logging.error(f"Failed to load extra custom node paths config from {yaml_path}: {e}")
        return
    
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    
    # Collect all custom node paths from the YAML configuration
    all_custom_node_paths = []
    
    if not config:
        return

    for c in config:
        conf = config[c]
        if conf is None:
            continue
        
        # Handle base_path (creates custom_nodes subdirectory)
        if "base_path" in conf:
            base_path = conf["base_path"]
            # Process the base path
            expanded_path = os.path.expandvars(os.path.expanduser(base_path))
            if not os.path.isabs(expanded_path):
                expanded_path = os.path.abspath(os.path.join(yaml_dir, expanded_path))
            
            # Create the custom_nodes subdirectory path
            custom_nodes_path = os.path.join(expanded_path, "custom_nodes")
            all_custom_node_paths.append(custom_nodes_path)
    
    # Add all custom node paths to the custom_nodes folder list
    if all_custom_node_paths:
        for custom_node_path in all_custom_node_paths:
            if os.path.exists(custom_node_path):
                logging.info(f"Adding extra custom node search path: {custom_node_path}")
                folder_paths.add_custom_node_directory(custom_node_path)
            else:
                logging.warning(f"Custom node path does not exist, skipping: {custom_node_path}")
        
        logging.info(f"Added {len(all_custom_node_paths)} custom node directories")
    else:
        logging.info("No custom node paths found in configuration")

def get_current_custom_node_paths() -> list[str]:
    """Get the current list of all custom node paths for debugging purposes"""
    return folder_paths.get_custom_nodes_directories()
