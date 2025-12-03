# How to Write Nodes in ComfyUI-Inference-Scaling

This guide will walk you through creating custom nodes for ComfyUI using the modern API system.

## Overview

In this ComfyUI project, nodes are created using the **ComfyAPI** system. Each node is a class that:
1. Inherits from `IO.ComfyNode`
2. Defines its schema (inputs, outputs, metadata) via `define_schema()`
3. Implements its execution logic via `execute()`
4. Is registered through a `ComfyExtension` class

## Basic Structure

### 1. Node Class

Every node is a class that inherits from `IO.ComfyNode`:

```python
from comfy_api.latest import IO, ComfyExtension
from typing_extensions import override
from inspect import cleandoc

class MyCustomNode(IO.ComfyNode):
    """
    Description of what your node does.
    This docstring will appear in the UI.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MyCustomNode",
            display_name="My Custom Node",
            category="mycategory/subcategory",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                # Define inputs here
            ],
            outputs=[
                # Define outputs here
            ],
        )

    @classmethod
    async def execute(cls, ...) -> IO.NodeOutput:
        # Implementation here
        pass
```

### 2. Schema Definition

The `define_schema()` method defines:
- **node_id**: Unique identifier (usually matches class name)
- **display_name**: Name shown in the UI
- **category**: Where it appears in the node menu (use `/` for subcategories)
- **description**: Tooltip/help text
- **inputs**: List of input definitions
- **outputs**: List of output definitions
- **hidden**: Optional hidden inputs (like auth tokens)
- **is_api_node**: Set to `True` for API nodes

### 3. Input Types

Common input types available:

```python
# String input
IO.String.Input(
    "prompt",
    default="",
    multiline=True,  # For longer text
    tooltip="Help text shown on hover",
    optional=True,  # Makes it optional
)

# Integer input
IO.Int.Input(
    "seed",
    default=0,
    min=0,
    max=100,
    step=1,
    display_mode=IO.NumberDisplay.slider,  # or .number
    control_after_generate=True,  # Shows randomize button
    tooltip="Random seed",
)

# Float input
IO.Float.Input(
    "strength",
    default=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    tooltip="Strength value",
)

# Combo/Dropdown input
IO.Combo.Input(
    "model",
    options=["option1", "option2", "option3"],
    default="option1",
    tooltip="Select a model",
)

# Image input
IO.Image.Input(
    "image",
    tooltip="Input image",
    optional=True,
)

# Mask input
IO.Mask.Input(
    "mask",
    tooltip="Mask for inpainting",
    optional=True,
)

# Audio input
IO.Audio.Input(
    "audio",
    tooltip="Input audio",
    optional=True,
)

# Video input
IO.Video.Input(
    "video",
    tooltip="Input video",
    optional=True,
)
```

### 4. Output Types

Common output types:

```python
# Image output
IO.Image.Output()

# Audio output
IO.Audio.Output()

# Video output
IO.Video.Output()

# String output
IO.String.Output()

# Integer output
IO.Int.Output()

# Float output
IO.Float.Output()
```

### 5. Execute Method

The `execute()` method is where your node's logic runs:

```python
@classmethod
async def execute(
    cls,
    # Parameters match input names from define_schema
    prompt: str,
    seed: int = 0,
    image: Optional[torch.Tensor] = None,
    # ... other inputs
) -> IO.NodeOutput:
    """
    Execute the node logic.
    
    Args:
        prompt: Text prompt
        seed: Random seed
        image: Optional image tensor (shape: [B, H, W, C])
        ...
    
    Returns:
        IO.NodeOutput with the result
    """
    # Your implementation here
    
    # For image outputs:
    result_tensor = ...  # torch.Tensor with shape [B, H, W, C]
    return IO.NodeOutput(result_tensor)
    
    # For multiple outputs:
    return IO.NodeOutput(image=result_image, metadata=result_metadata)
```

**Important Notes:**
- The method is `async` - use `await` for async operations
- Input parameters match the names from `define_schema()`
- Image tensors have shape `[Batch, Height, Width, Channels]` (usually RGBA)
- Use `IO.NodeOutput()` to return results

### 6. Extension Registration

To register your nodes, create an extension class:

```python
class MyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MyCustomNode,
            AnotherNode,
            # ... list all your nodes
        ]


async def comfy_entrypoint() -> MyExtension:
    """
    Entry point function that ComfyUI calls to load your extension.
    """
    return MyExtension()
```

## Complete Example

Here's a complete example of a simple image processing node:

```python
from io import BytesIO
import torch
import numpy as np
from PIL import Image
from typing import Optional
from typing_extensions import override
from inspect import cleandoc

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.util import validate_string


class ImageBrightnessNode(IO.ComfyNode):
    """
    Adjusts the brightness of an input image.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageBrightnessNode",
            display_name="Image Brightness",
            category="image/processing",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Input image to adjust",
                ),
                IO.Float.Input(
                    "brightness",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="Brightness multiplier (1.0 = no change)",
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        brightness: float = 1.0,
    ) -> IO.NodeOutput:
        """
        Adjust image brightness.
        
        Args:
            image: Input image tensor [B, H, W, C]
            brightness: Brightness multiplier
            
        Returns:
            Brightness-adjusted image
        """
        # Ensure we have a batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Apply brightness adjustment
        # Clamp values to [0, 1] range
        adjusted = torch.clamp(image * brightness, 0.0, 1.0)
        
        return IO.NodeOutput(adjusted)


class MyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ImageBrightnessNode,
        ]


async def comfy_entrypoint() -> MyExtension:
    return MyExtension()
```

## API Node Example

For API nodes (nodes that call external APIs), you'll typically:

1. Use utility functions from `comfy_api_nodes.util`:
   - `sync_op()` - for synchronous API calls
   - `poll_op()` - for polling async operations
   - `validate_string()` - for input validation
   - `tensor_to_bytesio()` - convert image tensors to bytes
   - `bytesio_to_image_tensor()` - convert bytes to image tensors

2. Include hidden inputs for authentication:
```python
hidden=[
    IO.Hidden.auth_token_comfy_org,
    IO.Hidden.api_key_comfy_org,
    IO.Hidden.unique_id,
],
is_api_node=True,
```

3. Use `ApiEndpoint` for API calls:
```python
from comfy_api_nodes.util import ApiEndpoint, sync_op

response = await sync_op(
    cls,
    ApiEndpoint(path="/api/endpoint", method="POST"),
    response_model=YourResponseModel,
    data=YourRequestModel(...),
    files={...},  # Optional file uploads
    content_type="application/json",
)
```

## File Organization

1. **Create your node file**: `comfy_api_nodes/nodes_yourname.py`
2. **Follow naming conventions**: 
   - Node classes: `YourNodeName` (PascalCase)
   - Extension class: `YourExtension` (PascalCase)
   - File: `nodes_yourname.py` (snake_case)
3. **Import required modules**:
   - `from comfy_api.latest import IO, ComfyExtension`
   - `from typing_extensions import override`
   - `from inspect import cleandoc`

## Testing Your Node

1. **Start ComfyUI**:
   ```bash
   python main.py
   ```

2. **Check the node appears** in the node menu under your specified category

3. **Test the node** by:
   - Adding it to a workflow
   - Connecting inputs
   - Executing the workflow

## Common Patterns

### Working with Images

```python
# Image tensor shape: [Batch, Height, Width, Channels]
# Channels are usually RGBA (4 channels)

# Convert PIL Image to tensor
pil_img = Image.open("image.png").convert("RGBA")
arr = np.asarray(pil_img).astype(np.float32) / 255.0
tensor = torch.from_numpy(arr).unsqueeze(0)  # Add batch dimension

# Convert tensor to PIL Image
tensor = image.squeeze(0).cpu()  # Remove batch, move to CPU
image_np = (tensor.numpy() * 255).astype(np.uint8)
pil_img = Image.fromarray(image_np)
```

### Validation

```python
from comfy_api_nodes.util import validate_string

# Validate string inputs
validate_string(prompt, strip_whitespace=False)

# Check optional inputs
if image is not None:
    # Process image
    pass
```

### Error Handling

```python
if some_condition:
    raise Exception("Error message here")
```

## Tips

1. **Use type hints** - They help with IDE autocomplete and documentation
2. **Add tooltips** - Help users understand what each input does
3. **Use `cleandoc()`** - Cleans up docstrings for display
4. **Make inputs optional** when appropriate - Improves usability
5. **Follow existing patterns** - Look at `nodes_openai.py` or `nodes_stability.py` for examples
6. **Test thoroughly** - Especially edge cases (None inputs, empty strings, etc.)

## Resources

- Existing node examples: `comfy_api_nodes/nodes_*.py`
- ComfyAPI documentation: `comfy_api/latest/`
- Utility functions: `comfy_api_nodes/util/`

## Next Steps

1. Look at existing nodes for reference
2. Start with a simple node
3. Test incrementally
4. Add more features as needed

Happy node writing! 🎨
