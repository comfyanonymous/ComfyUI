# ComfyUI Prompt API JSON Schema

This directory contains JSON Schema definitions for the ComfyUI API, providing formal specification, validation, and IDE support for API integrations.

## 📁 Files

- **`prompt_format.json`** - JSON Schema for the `/prompt` endpoint request format
- **`validation.py`** - Python utilities for schema validation 
- **`README.md`** - This documentation file

## 🚀 Quick Start

### 1. Get the Schema

The JSON Schema is available at: `GET /schema/prompt` or `GET /api/schema/prompt`

```bash
curl http://localhost:8188/schema/prompt
```

### 2. Enable Validation (Optional)

Add `?validate_schema=true` to your POST requests for server-side validation:

```bash
curl -X POST http://localhost:8188/prompt?validate_schema=true \
  -H "Content-Type: application/json" \
  -d @your_prompt.json
```

### 3. IDE Setup

Most modern IDEs support JSON Schema for autocomplete and validation:

**VS Code:**
```json
{
  "json.schemas": [
    {
      "fileMatch": ["**/comfyui_prompt*.json"],
      "url": "http://localhost:8188/schema/prompt"
    }
  ]
}
```

**IntelliJ/PyCharm:**
Settings → Languages & Frameworks → Schemas and DTDs → JSON Schema Mappings

## 📋 Schema Overview

The prompt format schema defines the structure for ComfyUI workflow execution requests:

```json
{
  "prompt": {
    "1": {
      "class_type": "CheckpointLoaderSimple",
      "inputs": {
        "ckpt_name": "model.safetensors"
      }
    },
    "2": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "a beautiful landscape",
        "clip": ["1", 1]
      }
    }
  },
  "prompt_id": "optional-uuid",
  "client_id": "optional-client-id"
}
```

### Key Properties

- **`prompt`** (required) - The node graph defining the workflow
- **`prompt_id`** (optional) - Unique identifier for tracking execution
- **`number`** (optional) - Execution priority (lower = higher priority)
- **`front`** (optional) - If true, prioritize this execution
- **`extra_data`** (optional) - Additional metadata
- **`client_id`** (optional) - WebSocket client identifier
- **`partial_execution_targets`** (optional) - Array of specific nodes to execute

### Node Structure

Each node in the prompt is keyed by a numeric string ID and contains:

- **`class_type`** (required) - The ComfyUI node class name
- **`inputs`** (required) - Input values or connections to other nodes
- **`_meta`** (optional) - Metadata not used in execution

### Input Types

Node inputs can be:

1. **Direct values:** `"text": "hello world"`
2. **Node connections:** `"clip": ["1", 0]` (node_id, output_slot)

## 🛠️ Validation

### Python Integration

```python
from api_schemas.validation import validate_prompt_format

data = {"prompt": {...}}
is_valid, error_msg = validate_prompt_format(data)

if not is_valid:
    print(f"Validation failed: {error_msg}")
```

### Server-Side Validation

Enable validation with query parameter:

```
POST /prompt?validate_schema=true
```

Returns `400 Bad Request` with detailed error information if validation fails.

## 🔧 Development

### Requirements

For validation features:
```bash
pip install jsonschema
```

### Schema Updates

When updating the schema:

1. Modify `prompt_format.json`
2. Test with real ComfyUI workflows
3. Update examples and documentation
4. Verify backward compatibility

### Testing

```python
# Test schema loading
from api_schemas.validation import load_prompt_schema
schema = load_prompt_schema()
assert schema is not None

# Test validation
from api_schemas.validation import validate_prompt_format
valid_prompt = {"prompt": {"1": {"class_type": "TestNode", "inputs": {}}}}
is_valid, error = validate_prompt_format(valid_prompt)
assert is_valid
```

## 📚 Examples

### Basic Text-to-Image Workflow

```json
{
  "prompt": {
    "1": {
      "class_type": "CheckpointLoaderSimple", 
      "inputs": {
        "ckpt_name": "model.safetensors"
      }
    },
    "2": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "a beautiful landscape",
        "clip": ["1", 1]
      }
    },
    "3": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "blurry, low quality", 
        "clip": ["1", 1]
      }
    },
    "4": {
      "class_type": "KSampler",
      "inputs": {
        "seed": 12345,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "model": ["1", 0],
        "positive": ["2", 0],
        "negative": ["3", 0],
        "latent_image": ["5", 0]
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
      "class_type": "VAEDecode",
      "inputs": {
        "samples": ["4", 0],
        "vae": ["1", 2]
      }
    },
    "7": {
      "class_type": "SaveImage",
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": ["6", 0]
      }
    }
  }
}
```

### Partial Execution

```json
{
  "prompt": {
    "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}
  },
  "partial_execution_targets": ["2"],
  "prompt_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## 🤝 Contributing

This schema implementation addresses GitHub issue [#8899](https://github.com/comfyanonymous/ComfyUI/issues/8899).

To contribute:

1. Test with real ComfyUI workflows
2. Report issues or inaccuracies
3. Suggest improvements for better IDE support
4. Help with documentation and examples

## 📄 License

This follows the same license as ComfyUI main repository.
