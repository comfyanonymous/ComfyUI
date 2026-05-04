#!/usr/bin/env python3
"""
Shader Blueprint Updater

Syncs GLSL shader files between this folder and blueprint JSON files.

File naming convention:
    {Blueprint Name}_{node_id}.frag

Usage:
    python update_blueprints.py extract   # Extract shaders from JSONs to here
    python update_blueprints.py patch     # Patch shaders back into JSONs
    python update_blueprints.py           # Same as patch (default)
"""

import json
import logging
import sys
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

GLSL_DIR = Path(__file__).parent
BLUEPRINTS_DIR = GLSL_DIR.parent


def get_blueprint_files():
    """Get all blueprint JSON files."""
    return sorted(BLUEPRINTS_DIR.glob("*.json"))


def sanitize_filename(name):
    """Convert blueprint name to safe filename."""
    return re.sub(r'[^\w\-]', '_', name)


def extract_shaders():
    """Extract all shaders from blueprint JSONs to this folder."""
    extracted = 0
    for json_path in get_blueprint_files():
        blueprint_name = json_path.stem

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Skipping %s: %s", json_path.name, e)
            continue

        # Find GLSLShader nodes in subgraphs
        for subgraph in data.get('definitions', {}).get('subgraphs', []):
            for node in subgraph.get('nodes', []):
                if node.get('type') == 'GLSLShader':
                    node_id = node.get('id')
                    widgets = node.get('widgets_values', [])

                    # Find shader code (first string that looks like GLSL)
                    for widget in widgets:
                        if isinstance(widget, str) and widget.startswith('#version'):
                            safe_name = sanitize_filename(blueprint_name)
                            frag_name = f"{safe_name}_{node_id}.frag"
                            frag_path = GLSL_DIR / frag_name

                            with open(frag_path, 'w') as f:
                                f.write(widget)

                            logger.info("  Extracted: %s", frag_name)
                            extracted += 1
                            break

    logger.info("\nExtracted %d shader(s)", extracted)


def patch_shaders():
    """Patch shaders from this folder back into blueprint JSONs."""
    # Build lookup: blueprint_name -> [(node_id, shader_code), ...]
    shader_updates = {}

    for frag_path in sorted(GLSL_DIR.glob("*.frag")):
        # Parse filename: {blueprint_name}_{node_id}.frag
        parts = frag_path.stem.rsplit('_', 1)
        if len(parts) != 2:
            logger.warning("Skipping %s: invalid filename format", frag_path.name)
            continue

        blueprint_name, node_id_str = parts

        try:
            node_id = int(node_id_str)
        except ValueError:
            logger.warning("Skipping %s: invalid node_id", frag_path.name)
            continue

        with open(frag_path, 'r') as f:
            shader_code = f.read()

        if blueprint_name not in shader_updates:
            shader_updates[blueprint_name] = []
        shader_updates[blueprint_name].append((node_id, shader_code))

    # Apply updates to JSON files
    patched = 0
    for json_path in get_blueprint_files():
        blueprint_name = sanitize_filename(json_path.stem)

        if blueprint_name not in shader_updates:
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Error reading %s: %s", json_path.name, e)
            continue

        modified = False
        for node_id, shader_code in shader_updates[blueprint_name]:
            # Find the node and update
            for subgraph in data.get('definitions', {}).get('subgraphs', []):
                for node in subgraph.get('nodes', []):
                    if node.get('id') == node_id and node.get('type') == 'GLSLShader':
                        widgets = node.get('widgets_values', [])
                        if len(widgets) > 0 and widgets[0] != shader_code:
                            widgets[0] = shader_code
                            modified = True
                            logger.info("  Patched: %s (node %d)", json_path.name, node_id)
                            patched += 1

        if modified:
            with open(json_path, 'w') as f:
                json.dump(data, f)

    if patched == 0:
        logger.info("No changes to apply.")
    else:
        logger.info("\nPatched %d shader(s)", patched)


def main():
    if len(sys.argv) < 2:
        command = "patch"
    else:
        command = sys.argv[1].lower()

    if command == "extract":
        logger.info("Extracting shaders from blueprints...")
        extract_shaders()
    elif command in ("patch", "update", "apply"):
        logger.info("Patching shaders into blueprints...")
        patch_shaders()
    else:
        logger.info(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
