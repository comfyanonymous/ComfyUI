# ComfyUI Prompt API Schema Documentation

## Overview

This document describes the JSON Schema for ComfyUI's `/prompt` API endpoint. The schema provides formal validation and documentation for the prompt format used when submitting workflows via the API.

## Schema Location

- **Schema File**: `schemas/prompt.json`
- **Schema ID**: `https://github.com/comfyanonymous/ComfyUI/schemas/prompt.json`

## Quick Start

### Validating a Prompt

```python
import json
import jsonschema

with open('schemas/prompt.json') as f:
    schema = json.load(f)

with open('your_prompt.json') as f:
    prompt = json.load(f)

# Validate
jsonschema.validate(prompt, schema)
print("✅ Prompt is valid!")
```

## Format Overview

The prompt format is a JSON object containing:

```json
{
  "prompt": {
    "node_id_1": { /* node definition */ },
    "node_id_2": { /* node definition */ },
    ...
  },
  "extra_data": { /* optional */ },
  "client_id": "optional-client-id"
}
```

### Node Structure

Each node has:
- `class_type`: The node class (e.g., "KSampler", "CLIPTextEncode")
- `inputs`: Input parameters
- `_meta`: Optional metadata

### Input Values

Inputs can be:
1. **Direct values**: Numbers, strings, booleans, arrays, objects
2. **Node links**: `["node_id", output_index]` to reference another node

## Examples

### Basic Text-to-Image

```json
{
  "prompt": {
    "3": {
      "class_type": "KSampler",
      "inputs": {
        "seed": 123456789,
        "steps": 20,
        "cfg": 8.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "model": ["4", 0],
        "positive": ["6", 0],
        "negative": ["7", 0],
        "latent_image": ["5", 0]
      }
    },
    "4": {
      "class_type": "CheckpointLoaderSimple",
      "inputs": {
        "ckpt_name": "model.safetensors"
      }
    },
    "5": {
      "class_type": "EmptyLatentImage",
      "inputs": {
        "width": 512,
        "height": 512,
        "batch_size": 1
      }
    },
    "6": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "beautiful scenery",
        "clip": ["4", 1]
      }
    },
    "7": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "text, watermark",
        "clip": ["4", 1]
      }
    },
    "8": {
      "class_type": "VAEDecode",
      "inputs": {
        "samples": ["3", 0],
        "vae": ["4", 2]
      }
    },
    "9": {
      "class_type": "SaveImage",
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": ["8", 0]
      }
    }
  }
}
```

## Workflow vs Prompt Format

### Workflow Format (UI Internal)

Used by the web interface for saving/loading workflows.

```json
{
  "last_node_id": 9,
  "last_link_id": 5,
  "nodes": [...],
  "links": [...],
  "groups": [...]
}
```

### Prompt Format (API)

Used by the `/prompt` API endpoint for execution.

```json
{
  "prompt": {
    "3": { "class_type": "KSampler", "inputs": {...} },
    ...
  }
}
```

**Key Differences**:
- Workflow contains UI metadata (positions, colors)
- Prompt is execution-focused (just nodes and connections)
- Workflow is converted to prompt format before execution

## Common Node Types

### Loaders
- `CheckpointLoaderSimple`: Load model checkpoints
- `VAELoader`: Load VAE models
- `LoraLoader`: Load LoRA models

### Encoders
- `CLIPTextEncode`: Encode text prompts
- `CLIPTextEncodeSDXL`: SDXL text encoding

### Samplers
- `KSampler`: Standard sampling
- `KSamplerAdvanced`: Advanced sampling options

### Image Operations
- `EmptyLatentImage`: Create empty latent
- `VAEDecode`: Decode latent to image
- `VAEDecodeTiled`: Tiled decoding for large images
- `SaveImage`: Save output image

## Future Enhancements

### Dynamic Schemas

Future versions may support:
- Node-specific validation based on installed custom nodes
- Input type validation beyond basic JSON types
- Required vs optional input validation
- Dependency checking between nodes

### IDE Integration

The schema enables:
- Auto-completion in IDEs (VS Code, PyCharm, etc.)
- Real-time validation during development
- Documentation tooltips

## Contributing

To extend the schema:

1. Add new node types to the definitions
2. Include examples for complex use cases
3. Update documentation with validation rules
4. Test with existing prompts

## References

- [ComfyUI API Documentation](https://docs.comfy.org/)
- [JSON Schema Specification](https://json-schema.org/)
- [Original Feature Request #8899](https://github.com/Comfy-Org/ComfyUI/issues/8899)
