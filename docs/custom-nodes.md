# Custom Nodes Guide

This document provides detailed information for developers who want to create custom nodes for ComfyUI.

## Table of Contents
- [Introduction](#introduction)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Custom Node Structure](#custom-node-structure)
- [Creating Your First Node](#creating-your-first-node)
- [Node Inputs and Outputs](#node-inputs-and-outputs)
- [Advanced Node Development](#advanced-node-development)
- [Distributing Custom Nodes](#distributing-custom-nodes)
- [Best Practices](#best-practices)

## Introduction

ComfyUI's power comes from its extensibility through custom nodes. Custom nodes allow you to:

- Add new functionality not available in the core application
- Optimize existing workflows
- Create specialized interfaces for specific tasks
- Integrate with external tools and services

This guide will walk you through creating, testing, and distributing custom nodes.

## Setting Up the Development Environment

### Prerequisites

- Python 3.9+ (3.11 recommended)
- ComfyUI installed (source installation recommended for development)
- Basic knowledge of Python
- Understanding of ComfyUI's node system

### Development Environment Setup

1. Install ComfyUI from source:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   
   # Set up development environment
   uv venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   
   # Install dependencies
   uv pip install -e ".[dev]"
   ```

2. Create a custom nodes directory:
   ```bash
   mkdir -p custom_nodes/my_custom_node
   cd custom_nodes/my_custom_node
   ```

## Custom Node Structure

A typical custom node package has the following structure:

```
my_custom_node/
├── __init__.py           # Entry point for your node
├── nodes.py              # Node implementation
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── web/                  # [Optional] Frontend components
    ├── js/
    │   └── my_node.js    # Custom UI components
    └── style.css         # Custom styling
```

## Creating Your First Node

### Basic Node Template

Create a file called `nodes.py` with the following content:

```python
# nodes.py
import torch
import numpy as np
from PIL import Image

class MyCustomNode:
    """
    A simple custom node that applies a filter to an image.
    """
    
    # Define the input and output types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
            },
        }
    
    # Define the return types
    RETURN_TYPES = ("IMAGE",)
    # Optional: Define output names (defaults to return types)
    RETURN_NAMES = ("filtered_image",)
    # Define the node category for UI organization
    CATEGORY = "image/filters"
    # Optional: Add a description
    DESCRIPTION = "Applies a custom filter to the input image"
    
    def __init__(self):
        pass
    
    def execute(self, image, intensity):
        # Convert from tensor format to numpy for processing
        # Assuming image is [B, H, W, C] format
        img_np = image.numpy()
        
        # Apply a simple brightness adjustment as an example
        adjusted = np.clip(img_np * intensity, 0, 1)
        
        # Convert back to tensor
        result = torch.from_numpy(adjusted)
        
        return (result,)
```

### Register Your Node

Create an `__init__.py` file to register your node:

```python
# __init__.py
from .nodes import MyCustomNode

NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "My Custom Filter"
}
```

## Node Inputs and Outputs

### Input Types

ComfyUI supports several input types:

- `INT`: Integer values
- `FLOAT`: Floating point values
- `STRING`: Text strings
- `BOOLEAN`: True/False values
- `IMAGE`: Image data
- Custom enum types: A list of string options

### Input Configuration

For numeric inputs, you can provide additional configuration:

```python
"parameter_name": ("FLOAT", {
    "default": 1.0,
    "min": 0.0,
    "max": 10.0,
    "step": 0.1,
    "display": "slider"  # or "number" for a numeric input field
})
```

For dropdown selectors:

```python
"mode": (["option1", "option2", "option3"],)
```

### Output Types

Common output types include:

- `IMAGE`: Processed image data
- `MASK`: Image mask data
- `LATENT`: Latent space representation
- `CONDITIONING`: Conditioning data for samplers
- `MODEL`: Model data

### Multi-Output Nodes

For nodes with multiple outputs:

```python
RETURN_TYPES = ("IMAGE", "MASK")
RETURN_NAMES = ("output_image", "image_mask")
```

## Advanced Node Development

### Handling Batches of Images

To process batches efficiently:

```python
def execute(self, image, intensity):
    # image has shape [B, H, W, C]
    batch_size = image.shape[0]
    result = []
    
    for i in range(batch_size):
        # Process each image in the batch
        img = image[i]
        # Apply processing
        processed = self.process_single_image(img, intensity)
        result.append(processed)
    
    # Stack results back into a batch
    return (torch.stack(result),)
```

### Integrating External Libraries

For nodes that use external libraries:

1. Add requirements to `requirements.txt`:
   ```
   opencv-python>=4.5.0
   scikit-image>=0.19.0
   ```

2. Import and use in your node:
   ```python
   import cv2
   from skimage import filters
   
   class ImageProcessingNode:
       # ...
       def execute(self, image, params):
           # Convert to format for OpenCV
           img_np = (image.numpy() * 255).astype(np.uint8)
           # Process with CV2
           processed = cv2.someFunction(img_np, params)
           # Convert back
           return (torch.from_numpy(processed / 255.0),)
   ```

### Custom UI Components

For advanced UI elements, create a JavaScript file in `web/js/`:

```javascript
// web/js/my_component.js
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MyCustomComponent",
    async setup(app) {
        // Register a custom widget
        app.registerNodeDef("MyCustomNode", {
            color: "#5588AA",
            uiFields: {
                "customParameter": (node, inputName) => {
                    // Create custom UI element
                    const widget = document.createElement("div");
                    widget.innerHTML = `<div class="custom-control">...</div>`;
                    return { element: widget };
                }
            }
        });
    }
});
```

## Distributing Custom Nodes

### Packaging

1. Create a `README.md` with installation and usage instructions
2. Include a `requirements.txt` with dependencies
3. Add example workflows in your documentation
4. Include screenshots of the node in action

### Installation Instructions

Provide clear installation instructions:

```markdown
## Installation

1. Navigate to your ComfyUI custom_nodes directory
2. Clone this repository:
   ```
   git clone https://github.com/username/my-custom-node.git
   ```
3. Install requirements:
   ```
   cd my-custom-node
   pip install -r requirements.txt
   ```
4. Restart ComfyUI
```

### Publishing

1. Publish your code to GitHub
2. Add your node to the [ComfyUI Custom Nodes List](https://github.com/comfyanonymous/ComfyUI-Custom-Nodes)
3. Share in the ComfyUI Discord community

## Best Practices

### Performance

- Optimize tensor operations for speed
- Use batch processing where possible
- Consider adding a "preview" mode for complex operations
- Clean up resources in `__del__` if needed

### Compatibility

- Test with different ComfyUI versions
- Document minimum requirements
- Provide fallbacks for optional dependencies
- Handle different image formats and dimensions

### User Experience

- Use clear, descriptive names for nodes and parameters
- Add tooltips with `DESCRIPTION` and input descriptions
- Include examples in your documentation
- Add visual feedback for long-running operations

### Error Handling

- Validate inputs before processing
- Provide clear error messages
- Handle edge cases gracefully
- Add debug logging for troubleshooting

### Version Management

- Use semantic versioning
- Keep a changelog
- Test thoroughly before releasing updates
- Document breaking changes