"""
Template example node file.
Copy this file and modify it to create your own custom nodes.
"""

from typing import Optional
from typing_extensions import override
from inspect import cleandoc
import torch
import numpy as np
from PIL import Image

from comfy_api.latest import IO, ComfyExtension


class ExampleNode(IO.ComfyNode):
    """
    This is an example node that demonstrates the basic structure.
    Replace this docstring with a description of what your node does.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ExampleNode",
            display_name="Example Node",
            category="example",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "text_input",
                    default="Hello, ComfyUI!",
                    multiline=False,
                    tooltip="A text input field",
                ),
                IO.Int.Input(
                    "number_input",
                    default=42,
                    min=0,
                    max=100,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="A number input with slider",
                ),
                IO.Image.Input(
                    "image_input",
                    tooltip="An optional image input",
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(),
                IO.Int.Output(),
                IO.Image.Output(),
            ],
        )

    @classmethod
    async def execute(
        cls,
        text_input: str,
        number_input: int,
        image_input: Optional[torch.Tensor] = None,
    ) -> IO.NodeOutput:
        """
        Execute the node logic.
        
        Args:
            text_input: The text input value
            number_input: The number input value
            image_input: Optional image tensor [B, H, W, C]
            
        Returns:
            NodeOutput with processed results
        """
        # Process text
        processed_text = f"Processed: {text_input}"
        
        # Process number
        processed_number = number_input * 2
        
        # Process image if provided
        if image_input is not None:
            # Ensure batch dimension exists
            if len(image_input.shape) == 3:
                image_input = image_input.unsqueeze(0)
            
            # Example: invert the image
            processed_image = 1.0 - image_input
        else:
            # Create a default image if none provided
            # Create a simple 256x256 RGBA image
            default_image = torch.ones(1, 256, 256, 4) * 0.5
            processed_image = default_image
        
        # Return multiple outputs
        return IO.NodeOutput(
            string=processed_text,
            int=processed_number,
            image=processed_image,
        )


class SimpleImageNode(IO.ComfyNode):
    """
    A simpler example with just image input/output.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SimpleImageNode",
            display_name="Simple Image Node",
            category="example/image",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Input image",
                ),
                IO.Float.Input(
                    "multiplier",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="Brightness multiplier",
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
        multiplier: float = 1.0,
    ) -> IO.NodeOutput:
        """
        Multiply image brightness.
        
        Args:
            image: Input image tensor [B, H, W, C]
            multiplier: Brightness multiplier
            
        Returns:
            Adjusted image
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Apply multiplier and clamp to valid range
        result = torch.clamp(image * multiplier, 0.0, 1.0)
        
        return IO.NodeOutput(result)


class ExampleExtension(ComfyExtension):
    """
    Extension class that registers all your nodes.
    """
    
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ExampleNode,
            SimpleImageNode,
            # Add more nodes here as you create them
        ]


async def comfy_entrypoint() -> ExampleExtension:
    """
    Entry point function that ComfyUI calls to load your extension.
    This function name must be exactly 'comfy_entrypoint'.
    """
    return ExampleExtension()
