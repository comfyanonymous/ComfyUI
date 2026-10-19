# GLSL Shader Sources

This folder contains the GLSL fragment shaders extracted from blueprint JSON files for easier editing and version control.

## File Naming Convention

`{Blueprint_Name}_{node_id}.frag`

- **Blueprint_Name**: The JSON filename with spaces/special chars replaced by underscores
- **node_id**: The GLSLShader node ID within the subgraph

## Usage

```bash
# Extract shaders from blueprint JSONs to this folder
python update_blueprints.py extract

# Patch edited shaders back into blueprint JSONs
python update_blueprints.py patch
```

## Workflow

1. Run `extract` to pull current shaders from JSONs
2. Edit `.frag` files
3. Run `patch` to update the blueprint JSONs
4. Test
5. Commit both `.frag` files and updated JSONs
