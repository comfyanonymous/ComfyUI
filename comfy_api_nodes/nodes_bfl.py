import torch
from pydantic import BaseModel
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.bfl import (
    BFLFluxExpandImageRequest,
    BFLFluxFillImageRequest,
    BFLFluxKontextProGenerateRequest,
    BFLFluxProGenerateResponse,
    BFLFluxProUltraGenerateRequest,
    BFLFluxStatusResponse,
    BFLStatus,
    Flux2ProGenerateRequest,
)
from comfy_api_nodes.util import (
    download_url_to_image_tensor,
    get_number_of_images,
    resize_mask_to_image,
    tensor_to_base64_string,
    validate_aspect_ratio_string,
    validate_string,
)
from comfy_api_nodes.util.client import fal_run

FAL_FLUX_PRO_ULTRA = "fal-ai/flux-pro/v1.1-ultra"
FAL_FLUX_KONTEXT_PRO = "fal-ai/flux-kontext/pro"
FAL_FLUX_KONTEXT_MAX = "fal-ai/flux-kontext/max"
FAL_FLUX_PRO_EXPAND = "fal-ai/flux-pro/v1/expand"
FAL_FLUX_PRO_FILL = "fal-ai/flux-pro/v1/fill"
FAL_FLUX_2_PRO = "fal-ai/flux-pro/v2"
FAL_FLUX_2_MAX = "fal-ai/flux-pro/v2/max"


def convert_mask_to_image(mask: Input.Image):
    """
    Make mask have the expected amount of dims (4) and channels (3) to be recognized as an image.
    """
    mask = mask.unsqueeze(-1)
    mask = torch.cat([mask] * 3, dim=-1)
    return mask


class FluxProUltraImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProUltraImageNode",
            display_name="Flux 1.1 [pro] Ultra Image",
            category="api node/image/BFL",
            description="Generates images using Flux Pro 1.1 Ultra via api based on prompt and resolution.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.String.Input(
                    "aspect_ratio",
                    default="16:9",
                    tooltip="Aspect ratio of image; must be between 1:4 and 4:1.",
                ),
                IO.Boolean.Input(
                    "raw",
                    default=False,
                    tooltip="When True, generate less processed, more natural-looking images.",
                ),
                IO.Image.Input(
                    "image_prompt",
                    optional=True,
                ),
                IO.Float.Input(
                    "image_prompt_strength",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Blend between the prompt and the image prompt.",
                    optional=True,
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    def validate_inputs(cls, aspect_ratio: str):
        validate_aspect_ratio_string(aspect_ratio, (1, 4), (4, 1))
        return True

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        prompt_upsampling: bool = False,
        raw: bool = False,
        seed: int = 0,
        image_prompt: Input.Image | None = None,
        image_prompt_strength: float = 0.1,
    ) -> IO.NodeOutput:
        if image_prompt is None:
            validate_string(prompt, strip_whitespace=False)
        data = {
            "prompt": prompt,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "aspect_ratio": aspect_ratio,
            "raw": raw,
        }
        if image_prompt is not None:
            data["image_prompt"] = tensor_to_base64_string(image_prompt)
            data["image_prompt_strength"] = round(image_prompt_strength, 2)

        # TODO: Verify fal.ai response schema for Flux Pro Ultra
        result = await fal_run(cls, FAL_FLUX_PRO_ULTRA, data)
        image_url = result.get("images", [{}])[0].get("url") or result.get("sample")
        return IO.NodeOutput(await download_url_to_image_tensor(image_url))


class FluxKontextProImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="api node/image/BFL",
            description="Edits images using Flux.1 Kontext [pro] via api based on prompt and aspect ratio.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation - specify what and how to edit.",
                ),
                IO.String.Input(
                    "aspect_ratio",
                    default="16:9",
                    tooltip="Aspect ratio of image; must be between 1:4 and 4:1.",
                ),
                IO.Float.Input(
                    "guidance",
                    default=3.0,
                    min=0.1,
                    max=99.0,
                    step=0.1,
                    tooltip="Guidance strength for the image generation process",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=1,
                    max=150,
                    tooltip="Number of steps for the image generation process",
                ),
                IO.Int.Input(
                    "seed",
                    default=1234,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Image.Input(
                    "input_image",
                    optional=True,
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    FAL_MODEL = FAL_FLUX_KONTEXT_PRO
    NODE_ID = "FluxKontextProImageNode"
    DISPLAY_NAME = "Flux.1 Kontext [pro] Image"

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        guidance: float,
        steps: int,
        input_image: Input.Image | None = None,
        seed=0,
        prompt_upsampling=False,
    ) -> IO.NodeOutput:
        validate_aspect_ratio_string(aspect_ratio, (1, 4), (4, 1))
        if input_image is None:
            validate_string(prompt, strip_whitespace=False)
        data = {
            "prompt": prompt,
            "prompt_upsampling": prompt_upsampling,
            "guidance": round(guidance, 1),
            "steps": steps,
            "seed": seed,
            "aspect_ratio": aspect_ratio,
        }
        if input_image is not None:
            data["input_image"] = tensor_to_base64_string(input_image)

        # TODO: Verify fal.ai response schema for Flux Kontext
        result = await fal_run(cls, cls.FAL_MODEL, data)
        image_url = result.get("images", [{}])[0].get("url") or result.get("sample")
        return IO.NodeOutput(await download_url_to_image_tensor(image_url))


class FluxKontextMaxImageNode(FluxKontextProImageNode):

    DESCRIPTION = "Edits images using Flux.1 Kontext [max] via api based on prompt and aspect ratio."
    FAL_MODEL = FAL_FLUX_KONTEXT_MAX
    NODE_ID = "FluxKontextMaxImageNode"
    DISPLAY_NAME = "Flux.1 Kontext [max] Image"


class FluxProExpandNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProExpandNode",
            display_name="Flux.1 Expand Image",
            category="api node/image/BFL",
            description="Outpaints image based on prompt.",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Int.Input(
                    "top",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the top of the image",
                ),
                IO.Int.Input(
                    "bottom",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the bottom of the image",
                ),
                IO.Int.Input(
                    "left",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the left of the image",
                ),
                IO.Int.Input(
                    "right",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the right of the image",
                ),
                IO.Float.Input(
                    "guidance",
                    default=60,
                    min=1.5,
                    max=100,
                    tooltip="Guidance strength for the image generation process",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=15,
                    max=50,
                    tooltip="Number of steps for the image generation process",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str,
        prompt_upsampling: bool,
        top: int,
        bottom: int,
        left: int,
        right: int,
        steps: int,
        guidance: float,
        seed=0,
    ) -> IO.NodeOutput:
        data = {
            "prompt": prompt,
            "prompt_upsampling": prompt_upsampling,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "image": tensor_to_base64_string(image),
        }
        # TODO: Verify fal.ai response schema for Flux Expand
        result = await fal_run(cls, FAL_FLUX_PRO_EXPAND, data)
        image_url = result.get("images", [{}])[0].get("url") or result.get("sample")
        return IO.NodeOutput(await download_url_to_image_tensor(image_url))


class FluxProFillNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProFillNode",
            display_name="Flux.1 Fill Image",
            category="api node/image/BFL",
            description="Inpaints image based on mask and prompt.",
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Float.Input(
                    "guidance",
                    default=60,
                    min=1.5,
                    max=100,
                    tooltip="Guidance strength for the image generation process",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=15,
                    max=50,
                    tooltip="Number of steps for the image generation process",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        mask: Input.Image,
        prompt: str,
        prompt_upsampling: bool,
        steps: int,
        guidance: float,
        seed=0,
    ) -> IO.NodeOutput:
        # prepare mask
        mask = resize_mask_to_image(mask, image)
        mask = tensor_to_base64_string(convert_mask_to_image(mask))
        data = {
            "prompt": prompt,
            "prompt_upsampling": prompt_upsampling,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "image": tensor_to_base64_string(image[:, :, :, :3]),
            "mask": mask,
        }
        # TODO: Verify fal.ai response schema for Flux Fill
        result = await fal_run(cls, FAL_FLUX_PRO_FILL, data)
        image_url = result.get("images", [{}])[0].get("url") or result.get("sample")
        return IO.NodeOutput(await download_url_to_image_tensor(image_url))


class Flux2ProImageNode(IO.ComfyNode):

    NODE_ID = "Flux2ProImageNode"
    DISPLAY_NAME = "Flux.2 [pro] Image"
    FAL_MODEL_ID = FAL_FLUX_2_PRO
    PRICE_BADGE_EXPR = """
    (
      $MP := 1024 * 1024;
      $outMP := $max([1, $floor(((widgets.width * widgets.height) + $MP - 1) / $MP)]);
      $outputCost := 0.03 + 0.015 * ($outMP - 1);
      inputs.images.connected
        ? {
            "type":"range_usd",
            "min_usd": $outputCost + 0.015,
            "max_usd": $outputCost + 0.12,
            "format": { "approximate": true }
          }
        : {"type":"usd","usd": $outputCost}
    )
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="api node/image/BFL",
            description="Generates images synchronously based on prompt and resolution.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation or edit",
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=256,
                    max=2048,
                    step=32,
                ),
                IO.Int.Input(
                    "height",
                    default=768,
                    min=256,
                    max=2048,
                    step=32,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=True,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation.",
                    advanced=True,
                ),
                IO.Image.Input("images", optional=True, tooltip="Up to 9 images to be used as references."),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        prompt_upsampling: bool,
        images: Input.Image | None = None,
    ) -> IO.NodeOutput:
        data = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "prompt_upsampling": prompt_upsampling,
        }
        if images is not None:
            if get_number_of_images(images) > 9:
                raise ValueError("The current maximum number of supported images is 9.")
            for image_index in range(images.shape[0]):
                key_name = f"input_image_{image_index + 1}" if image_index else "input_image"
                data[key_name] = tensor_to_base64_string(images[image_index], total_pixels=2048 * 2048)

        # TODO: Verify fal.ai response schema for Flux 2 Pro
        result = await fal_run(cls, cls.FAL_MODEL_ID, data)
        image_url = result.get("images", [{}])[0].get("url") or result.get("sample")
        return IO.NodeOutput(await download_url_to_image_tensor(image_url))


class Flux2MaxImageNode(Flux2ProImageNode):

    NODE_ID = "Flux2MaxImageNode"
    DISPLAY_NAME = "Flux.2 [max] Image"
    FAL_MODEL_ID = FAL_FLUX_2_MAX
    PRICE_BADGE_EXPR = """
    (
      $MP := 1024 * 1024;
      $outMP := $max([1, $floor(((widgets.width * widgets.height) + $MP - 1) / $MP)]);
      $outputCost := 0.07 + 0.03 * ($outMP - 1);

      inputs.images.connected
        ? {
            "type":"range_usd",
            "min_usd": $outputCost + 0.03,
            "max_usd": $outputCost + 0.24,
            "format": { "approximate": true }
          }
        : {"type":"usd","usd": $outputCost}
    )
    """


class BFLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            FluxProUltraImageNode,
            FluxKontextProImageNode,
            FluxKontextMaxImageNode,
            FluxProExpandNode,
            FluxProFillNode,
            Flux2ProImageNode,
            Flux2MaxImageNode,
        ]


async def comfy_entrypoint() -> BFLExtension:
    return BFLExtension()
