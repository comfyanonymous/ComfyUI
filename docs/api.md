# API Documentation

This document provides detailed information about the ComfyUI API for developers wanting to integrate with or automate ComfyUI.

## Table of Contents
- [API Overview](#api-overview)
- [REST API Endpoints](#rest-api-endpoints)
- [WebSocket API](#websocket-api)
- [Python SDK](#python-sdk)
- [Examples](#examples)
- [Best Practices](#best-practices)

## API Overview

ComfyUI provides a comprehensive API that allows for:
- Executing workflows
- Retrieving generated images
- Managing models
- Monitoring task status
- Controlling the server

The API is accessible via HTTP REST endpoints and WebSocket for real-time updates.

## REST API Endpoints

### Server Information

#### Get System Stats
```
GET /system_stats
```
Returns system information including RAM, VRAM, Python version, and PyTorch version.

#### Get Object Info
```
GET /object_info
```
Returns information about all available nodes.

#### Get Object Info for Specific Node
```
GET /object_info/{node_class}
```
Returns information about a specific node class.

### Workflow Execution

#### Execute Workflow
```
POST /prompt
```
Execute a workflow with the given prompt.

**Request Body:**
```json
{
  "prompt": {
    // Workflow JSON
  },
  "client_id": "optional_client_id"
}
```

**Response:**
```json
{
  "prompt_id": "uuid",
  "number": 1,
  "node_errors": {}
}
```

#### Get Execution History
```
GET /history
```
Returns execution history.

#### Get Specific Execution
```
GET /history/{prompt_id}
```
Returns details of a specific execution.

### Queue Management

#### Get Queue
```
GET /queue
```
Returns the current execution queue.

#### Modify Queue
```
POST /queue
```
Modify the execution queue (clear or delete items).

**Request Body:**
```json
{
  "clear": true,
  "delete": ["prompt_id1", "prompt_id2"]
}
```

#### Interrupt Processing
```
POST /interrupt
```
Interrupts the current processing task.

### Model Management

#### Get Available Models
```
GET /models
```
Returns a list of available model types.

#### Get Models of a Specific Type
```
GET /models/{folder}
```
Returns a list of available models in the specified folder.

#### View Model Metadata
```
GET /view_metadata/{folder_name}?filename=model.safetensors
```
Returns metadata for a specific model file.

### File Operations

#### Upload Image
```
POST /upload/image
```
Upload an image to use in workflows.

#### View Image
```
GET /view?filename=image.png&type=output
```
Returns an image file from the specified location.

## WebSocket API

ComfyUI provides real-time updates via WebSocket connection at `/ws`.

### Connection

Connect to the WebSocket endpoint:
```javascript
const socket = new WebSocket('ws://localhost:8188/ws');
```

### Message Types

The WebSocket API sends messages with the following types:

#### Status Update
```json
{
  "type": "status",
  "data": {
    "status": {
      "exec_info": {
        "queue_remaining": 0
      }
    },
    "sid": "session_id"
  }
}
```

#### Execution Started
```json
{
  "type": "execution_start",
  "data": {
    "prompt_id": "uuid"
  }
}
```

#### Executing Node
```json
{
  "type": "executing",
  "data": {
    "node": "node_id",
    "prompt_id": "uuid"
  }
}
```

#### Progress Update
```json
{
  "type": "progress",
  "data": {
    "value": 1,
    "max": 100,
    "prompt_id": "uuid",
    "node": "node_id"
  }
}
```

#### Execution Complete
```json
{
  "type": "executed",
  "data": {
    "node": "node_id",
    "output": {
      "images": [
        {
          "filename": "image.png",
          "subfolder": "outputs",
          "type": "output"
        }
      ]
    },
    "prompt_id": "uuid"
  }
}
```

#### Execution Error
```json
{
  "type": "execution_error",
  "data": {
    "prompt_id": "uuid",
    "node_id": "node_id",
    "exception_message": "Error message",
    "exception_type": "Exception type",
    "traceback": ["Traceback lines"]
  }
}
```

## Python SDK

ComfyUI provides a Python SDK for easier integration with Python applications.

### Installation

```bash
uv pip install comfyui-client
```

### Basic Usage

```python
from comfyui_client import ComfyUIClient

# Initialize client
client = ComfyUIClient(host="localhost", port=8188)

# Load a workflow from file
workflow = client.load_workflow("my_workflow.json")

# Execute workflow
result = client.execute_workflow(workflow)

# Get generated images
images = client.get_images(result)

# Save images
for i, img in enumerate(images):
    img.save(f"result_{i}.png")
```

## Examples

### Execute a Simple Workflow with cURL

```bash
curl -X POST http://localhost:8188/prompt -H "Content-Type: application/json" -d @- << 'EOF'
{
  "prompt": {
    "3": {
      "inputs": {
        "seed": 1234,
        "steps": 20,
        "cfg": 7,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "denoise": 1,
        "model": ["4", 0],
        "positive": ["6", 0],
        "negative": ["7", 0],
        "latent_image": ["5", 0]
      },
      "class_type": "KSampler"
    },
    "4": {
      "inputs": {
        "ckpt_name": "dreamshaper_8.safetensors"
      },
      "class_type": "CheckpointLoaderSimple"
    },
    "5": {
      "inputs": {
        "width": 512,
        "height": 512,
        "batch_size": 1
      },
      "class_type": "EmptyLatentImage"
    },
    "6": {
      "inputs": {
        "text": "beautiful landscape, mountains, lake, sunset, detailed, realistic",
        "clip": ["4", 1]
      },
      "class_type": "CLIPTextEncode"
    },
    "7": {
      "inputs": {
        "text": "blurry, bad quality, low resolution, ugly",
        "clip": ["4", 1]
      },
      "class_type": "CLIPTextEncode"
    },
    "8": {
      "inputs": {
        "samples": ["3", 0],
        "vae": ["4", 2]
      },
      "class_type": "VAEDecode"
    },
    "9": {
      "inputs": {
        "filename_prefix": "output",
        "images": ["8", 0]
      },
      "class_type": "SaveImage"
    }
  }
}
EOF
```

### JavaScript WebSocket Example

```javascript
const socket = new WebSocket('ws://localhost:8188/ws');

socket.onopen = () => {
  console.log('Connected to ComfyUI');
};

socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case 'status':
      console.log('Status update:', message.data);
      break;
    case 'progress':
      console.log(`Progress: ${message.data.value}/${message.data.max}`);
      break;
    case 'executed':
      console.log('Node executed:', message.data);
      if (message.data.output && message.data.output.images) {
        const imagePath = message.data.output.images[0].filename;
        console.log('Image generated:', imagePath);
      }
      break;
    case 'execution_error':
      console.error('Error:', message.data.exception_message);
      break;
  }
};

// Execute a workflow
fetch('/prompt', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: workflow })
});
```

## Best Practices

### Performance

- Use batching for multiple related generations
- Avoid polling, use WebSocket for real-time updates
- Reuse model loading nodes across executions

### Error Handling

- Always check for and handle error responses
- Implement retries with backoff for transient errors
- Monitor WebSocket for execution_error messages

### Security

- Validate all inputs before sending to the API
- Use TLS when exposing ComfyUI outside your network
- Consider implementing authentication for public instances

### Resource Management

- Monitor system resources via /system_stats endpoint
- Implement server-side queue limits for multi-user setups
- Use the /free endpoint to release memory when needed