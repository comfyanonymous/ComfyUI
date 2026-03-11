"""Generic fal.ai node -- submit any fal.ai model by ID with JSON input."""

import base64
import json
from io import BytesIO

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.util.client import fal_run
from comfy_api_nodes.util.conversions import bytesio_to_image_tensor
from comfy_api_nodes.util.upload_helpers import upload_image_to_fal


class FalGenericNode(IO.ComfyNode):
    """Submit any fal.ai model by providing a model ID and JSON input.

    Use this node to access any of the 1200+ models on fal.ai with a single
    FAL_API_KEY environment variable.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FalGenericNode",
            display_name="fal.ai Generic Model",
            category="api node/fal.ai",
            description=(
                "Run any fal.ai model by providing its model ID and JSON input. "
                "Browse models at https://fal.ai/models"
            ),
            inputs=[
                IO.String.Input(
                    "model_id",
                    default="fal-ai/flux/dev",
                    tooltip=(
                        "fal.ai model ID, e.g. 'fal-ai/flux/dev', 'fal-ai/kling-video/v2/master/text-to-video'. "
                        "Find model IDs at https://fal.ai/models"
                    ),
                ),
                IO.String.Input(
                    "input_json",
                    multiline=True,
                    default='{"prompt": "a cat in space"}',
                    tooltip="JSON object with the model's input parameters. Check the model's API page for the schema.",
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "Optional image input. If provided, it will be uploaded to fal.ai CDN "
                        "and the URL will be added to the input JSON as 'image_url'."
                    ),
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Random seed. Added to input JSON as 'seed' if > 0.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output("result_json"),
                IO.Image.Output("images"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model_id: str,
        input_json: str,
        image: Input.Image | None = None,
        seed: int = 0,
    ) -> IO.NodeOutput:
        # Parse input JSON
        try:
            data = json.loads(input_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid input JSON: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Input JSON must be a JSON object (dict), not an array or scalar.")

        # Upload image if provided and add URL to input
        if image is not None:
            image_url = await upload_image_to_fal(image)
            data["image_url"] = image_url

        # Add seed if provided
        if seed > 0:
            data["seed"] = seed

        # Run the model via fal.ai queue
        result = await fal_run(cls, model_id, data)

        # Extract images if present in result
        output_images = None
        if "images" in result and isinstance(result["images"], list):
            image_tensors = []
            for img_data in result["images"]:
                if isinstance(img_data, dict) and "url" in img_data:
                    # Download image from fal.ai CDN URL
                    from comfy_api_nodes.util import download_url_to_image_tensor
                    tensor = await download_url_to_image_tensor(img_data["url"])
                    image_tensors.append(tensor)
            if image_tensors:
                output_images = torch.cat(image_tensors, dim=0)
        elif "image" in result and isinstance(result["image"], dict) and "url" in result["image"]:
            from comfy_api_nodes.util import download_url_to_image_tensor
            output_images = await download_url_to_image_tensor(result["image"]["url"])

        if output_images is None:
            output_images = torch.zeros((1, 64, 64, 3))

        # Return result JSON and images
        result_str = json.dumps(result, indent=2, default=str)
        return IO.NodeOutput(result_str, output_images)


class FalExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [FalGenericNode]


async def comfy_entrypoint() -> FalExtension:
    return FalExtension()
