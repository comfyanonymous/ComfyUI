import asyncio
import io
from inspect import cleandoc
from typing import Union, Optional
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO
from comfy_api_nodes.apis.bfl_api import (
    BFLStatus,
    BFLFluxExpandImageRequest,
    BFLFluxFillImageRequest,
    BFLFluxCannyImageRequest,
    BFLFluxDepthImageRequest,
    BFLFluxProGenerateRequest,
    BFLFluxKontextProGenerateRequest,
    BFLFluxProUltraGenerateRequest,
    BFLFluxProGenerateResponse,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    downscale_image_tensor,
    validate_aspect_ratio,
    process_image_response,
    resize_mask_to_image,
    validate_string,
)

import numpy as np
from PIL import Image
import aiohttp
import torch
import base64
import time
from server import PromptServer


def convert_mask_to_image(mask: torch.Tensor):
    """
    Make mask have the expected amount of dims (4) and channels (3) to be recognized as an image.
    """
    mask = mask.unsqueeze(-1)
    mask = torch.cat([mask]*3, dim=-1)
    return mask


async def handle_bfl_synchronous_operation(
    operation: SynchronousOperation,
    timeout_bfl_calls=360,
    node_id: Union[str, None] = None,
):
    response_api: BFLFluxProGenerateResponse = await operation.execute()
    return await _poll_until_generated(
        response_api.polling_url, timeout=timeout_bfl_calls, node_id=node_id
    )


async def _poll_until_generated(
    polling_url: str, timeout=360, node_id: Union[str, None] = None
):
    # used bfl-comfy-nodes to verify code implementation:
    # https://github.com/black-forest-labs/bfl-comfy-nodes/tree/main
    start_time = time.time()
    retries_404 = 0
    max_retries_404 = 5
    retry_404_seconds = 2
    retry_202_seconds = 2
    retry_pending_seconds = 1

    async with aiohttp.ClientSession() as session:
        # NOTE: should True loop be replaced with checking if workflow has been interrupted?
        while True:
            if node_id:
                time_elapsed = time.time() - start_time
                PromptServer.instance.send_progress_text(
                    f"Generating ({time_elapsed:.0f}s)", node_id
                )

            async with session.get(polling_url) as response:
                if response.status == 200:
                    result = await response.json()
                    if result["status"] == BFLStatus.ready:
                        img_url = result["result"]["sample"]
                        if node_id:
                            PromptServer.instance.send_progress_text(
                                f"Result URL: {img_url}", node_id
                            )
                        async with session.get(img_url) as img_resp:
                            return process_image_response(await img_resp.content.read())
                    elif result["status"] in [
                        BFLStatus.request_moderated,
                        BFLStatus.content_moderated,
                    ]:
                        status = result["status"]
                        raise Exception(
                            f"BFL API did not return an image due to: {status}."
                        )
                    elif result["status"] == BFLStatus.error:
                        raise Exception(f"BFL API encountered an error: {result}.")
                    elif result["status"] == BFLStatus.pending:
                        await asyncio.sleep(retry_pending_seconds)
                        continue
                elif response.status == 404:
                    if retries_404 < max_retries_404:
                        retries_404 += 1
                        await asyncio.sleep(retry_404_seconds)
                        continue
                    raise Exception(
                        f"BFL API could not find task after {max_retries_404} tries."
                    )
                elif response.status == 202:
                    await asyncio.sleep(retry_202_seconds)
                elif time.time() - start_time > timeout:
                    raise Exception(
                        f"BFL API experienced a timeout; could not return request under {timeout} seconds."
                    )
                else:
                    raise Exception(f"BFL API encountered an error: {response.json()}")

def convert_image_to_base64(image: torch.Tensor):
    scaled_image = downscale_image_tensor(image, total_pixels=2048 * 2048)
    # remove batch dimension if present
    if len(scaled_image.shape) > 3:
        scaled_image = scaled_image[0]
    image_np = (scaled_image.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return base64.b64encode(img_byte_arr.getvalue()).decode()


class FluxProUltraImageNode(IO.ComfyNode):
    """
    Generates images using Flux Pro 1.1 Ultra via api based on prompt and resolution.
    """

    MINIMUM_RATIO = 1 / 4
    MAXIMUM_RATIO = 4 / 1
    MINIMUM_RATIO_STR = "1:4"
    MAXIMUM_RATIO_STR = "4:1"

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProUltraImageNode",
            display_name="Flux 1.1 [pro] Ultra Image",
            category="api node/image/BFL",
            description=cleandoc(cls.__doc__ or ""),
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
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
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
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    def validate_inputs(cls, aspect_ratio: str):
        try:
            validate_aspect_ratio(
                aspect_ratio,
                minimum_ratio=cls.MINIMUM_RATIO,
                maximum_ratio=cls.MAXIMUM_RATIO,
                minimum_ratio_str=cls.MINIMUM_RATIO_STR,
                maximum_ratio_str=cls.MAXIMUM_RATIO_STR,
            )
        except Exception as e:
            return str(e)
        return True

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        prompt_upsampling=False,
        raw=False,
        seed=0,
        image_prompt=None,
        image_prompt_strength=0.1,
    ) -> IO.NodeOutput:
        if image_prompt is None:
            validate_string(prompt, strip_whitespace=False)
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.1-ultra/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxProUltraGenerateRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxProUltraGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                seed=seed,
                aspect_ratio=validate_aspect_ratio(
                    aspect_ratio,
                    minimum_ratio=cls.MINIMUM_RATIO,
                    maximum_ratio=cls.MAXIMUM_RATIO,
                    minimum_ratio_str=cls.MINIMUM_RATIO_STR,
                    maximum_ratio_str=cls.MAXIMUM_RATIO_STR,
                ),
                raw=raw,
                image_prompt=(
                    image_prompt
                    if image_prompt is None
                    else convert_image_to_base64(image_prompt)
                ),
                image_prompt_strength=(
                    None if image_prompt is None else round(image_prompt_strength, 2)
                ),
            ),
            auth_kwargs={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
        )
        output_image = await handle_bfl_synchronous_operation(operation, node_id=cls.hidden.unique_id)
        return IO.NodeOutput(output_image)


class FluxKontextProImageNode(IO.ComfyNode):
    """
    Edits images using Flux.1 Kontext [pro] via api based on prompt and aspect ratio.
    """

    MINIMUM_RATIO = 1 / 4
    MAXIMUM_RATIO = 4 / 1
    MINIMUM_RATIO_STR = "1:4"
    MAXIMUM_RATIO_STR = "4:1"

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="api node/image/BFL",
            description=cleandoc(cls.__doc__ or ""),
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
                ),
                IO.Image.Input(
                    "input_image",
                    optional=True,
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    BFL_PATH = "/proxy/bfl/flux-kontext-pro/generate"
    NODE_ID = "FluxKontextProImageNode"
    DISPLAY_NAME = "Flux.1 Kontext [pro] Image"

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        guidance: float,
        steps: int,
        input_image: Optional[torch.Tensor]=None,
        seed=0,
        prompt_upsampling=False,
    ) -> IO.NodeOutput:
        aspect_ratio = validate_aspect_ratio(
            aspect_ratio,
            minimum_ratio=cls.MINIMUM_RATIO,
            maximum_ratio=cls.MAXIMUM_RATIO,
            minimum_ratio_str=cls.MINIMUM_RATIO_STR,
            maximum_ratio_str=cls.MAXIMUM_RATIO_STR,
        )
        if input_image is None:
            validate_string(prompt, strip_whitespace=False)
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=cls.BFL_PATH,
                method=HttpMethod.POST,
                request_model=BFLFluxKontextProGenerateRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxKontextProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                guidance=round(guidance, 1),
                steps=steps,
                seed=seed,
                aspect_ratio=aspect_ratio,
                input_image=(
                    input_image
                    if input_image is None
                    else convert_image_to_base64(input_image)
                )
            ),
            auth_kwargs={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
        )
        output_image = await handle_bfl_synchronous_operation(operation, node_id=cls.hidden.unique_id)
        return IO.NodeOutput(output_image)


class FluxKontextMaxImageNode(FluxKontextProImageNode):
    """
    Edits images using Flux.1 Kontext [max] via api based on prompt and aspect ratio.
    """

    DESCRIPTION = cleandoc(__doc__ or "")
    BFL_PATH = "/proxy/bfl/flux-kontext-max/generate"
    NODE_ID = "FluxKontextMaxImageNode"
    DISPLAY_NAME = "Flux.1 Kontext [max] Image"


class FluxProImageNode(IO.ComfyNode):
    """
    Generates images synchronously based on prompt and resolution.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProImageNode",
            display_name="Flux 1.1 [pro] Image",
            category="api node/image/BFL",
            description=cleandoc(cls.__doc__ or ""),
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
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=256,
                    max=1440,
                    step=32,
                ),
                IO.Int.Input(
                    "height",
                    default=768,
                    min=256,
                    max=1440,
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
                IO.Image.Input(
                    "image_prompt",
                    optional=True,
                ),
                # "image_prompt_strength": (
                #     IO.FLOAT,
                #     {
                #         "default": 0.1,
                #         "min": 0.0,
                #         "max": 1.0,
                #         "step": 0.01,
                #         "tooltip": "Blend between the prompt and the image prompt.",
                #     },
                # ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        prompt_upsampling,
        width: int,
        height: int,
        seed=0,
        image_prompt=None,
        # image_prompt_strength=0.1,
    ) -> IO.NodeOutput:
        image_prompt = (
                    image_prompt
                    if image_prompt is None
                    else convert_image_to_base64(image_prompt)
                )

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.1/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxProGenerateRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                width=width,
                height=height,
                seed=seed,
                image_prompt=image_prompt,
            ),
            auth_kwargs={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
        )
        output_image = await handle_bfl_synchronous_operation(operation, node_id=cls.hidden.unique_id)
        return IO.NodeOutput(output_image)


class FluxProExpandNode(IO.ComfyNode):
    """
    Outpaints image based on prompt.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProExpandNode",
            display_name="Flux.1 Expand Image",
            category="api node/image/BFL",
            description=cleandoc(cls.__doc__ or ""),
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
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
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
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
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
        image = convert_image_to_base64(image)

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.0-expand/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxExpandImageRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxExpandImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                steps=steps,
                guidance=guidance,
                seed=seed,
                image=image,
            ),
            auth_kwargs={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
        )
        output_image = await handle_bfl_synchronous_operation(operation, node_id=cls.hidden.unique_id)
        return IO.NodeOutput(output_image)



class FluxProFillNode(IO.ComfyNode):
    """
    Inpaints image based on mask and prompt.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProFillNode",
            display_name="Flux.1 Fill Image",
            category="api node/image/BFL",
            description=cleandoc(cls.__doc__ or ""),
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
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
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
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        prompt_upsampling: bool,
        steps: int,
        guidance: float,
        seed=0,
    ) -> IO.NodeOutput:
        # prepare mask
        mask = resize_mask_to_image(mask, image)
        mask = convert_image_to_base64(convert_mask_to_image(mask))
        # make sure image will have alpha channel removed
        image = convert_image_to_base64(image[:, :, :, :3])

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.0-fill/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxFillImageRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxFillImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                steps=steps,
                guidance=guidance,
                seed=seed,
                image=image,
                mask=mask,
            ),
            auth_kwargs={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
        )
        output_image = await handle_bfl_synchronous_operation(operation, node_id=cls.hidden.unique_id)
        return IO.NodeOutput(output_image)


class FluxProCannyNode(IO.ComfyNode):
    """
    Generate image using a control image (canny).
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProCannyNode",
            display_name="Flux.1 Canny Control Image",
            category="api node/image/BFL",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("control_image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
                ),
                IO.Float.Input(
                    "canny_low_threshold",
                    default=0.1,
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    tooltip="Low threshold for Canny edge detection; ignored if skip_processing is True",
                ),
                IO.Float.Input(
                    "canny_high_threshold",
                    default=0.4,
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    tooltip="High threshold for Canny edge detection; ignored if skip_processing is True",
                ),
                IO.Boolean.Input(
                    "skip_preprocessing",
                    default=False,
                    tooltip="Whether to skip preprocessing; set to True if control_image already is canny-fied, False if it is a raw image.",
                ),
                IO.Float.Input(
                    "guidance",
                    default=30,
                    min=1,
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
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        control_image: torch.Tensor,
        prompt: str,
        prompt_upsampling: bool,
        canny_low_threshold: float,
        canny_high_threshold: float,
        skip_preprocessing: bool,
        steps: int,
        guidance: float,
        seed=0,
    ) -> IO.NodeOutput:
        control_image = convert_image_to_base64(control_image[:, :, :, :3])
        preprocessed_image = None

        # scale canny threshold between 0-500, to match BFL's API
        def scale_value(value: float, min_val=0, max_val=500):
            return min_val + value * (max_val - min_val)
        canny_low_threshold = int(round(scale_value(canny_low_threshold)))
        canny_high_threshold = int(round(scale_value(canny_high_threshold)))


        if skip_preprocessing:
            preprocessed_image = control_image
            control_image = None
            canny_low_threshold = None
            canny_high_threshold = None

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.0-canny/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxCannyImageRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxCannyImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                steps=steps,
                guidance=guidance,
                seed=seed,
                control_image=control_image,
                canny_low_threshold=canny_low_threshold,
                canny_high_threshold=canny_high_threshold,
                preprocessed_image=preprocessed_image,
            ),
            auth_kwargs={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
        )
        output_image = await handle_bfl_synchronous_operation(operation, node_id=cls.hidden.unique_id)
        return IO.NodeOutput(output_image)


class FluxProDepthNode(IO.ComfyNode):
    """
    Generate image using a control image (depth).
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProDepthNode",
            display_name="Flux.1 Depth Control Image",
            category="api node/image/BFL",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("control_image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
                ),
                IO.Boolean.Input(
                    "skip_preprocessing",
                    default=False,
                    tooltip="Whether to skip preprocessing; set to True if control_image already is depth-ified, False if it is a raw image.",
                ),
                IO.Float.Input(
                    "guidance",
                    default=15,
                    min=1,
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
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        control_image: torch.Tensor,
        prompt: str,
        prompt_upsampling: bool,
        skip_preprocessing: bool,
        steps: int,
        guidance: float,
        seed=0,
    ) -> IO.NodeOutput:
        control_image = convert_image_to_base64(control_image[:,:,:,:3])
        preprocessed_image = None

        if skip_preprocessing:
            preprocessed_image = control_image
            control_image = None

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.0-depth/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxDepthImageRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxDepthImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                steps=steps,
                guidance=guidance,
                seed=seed,
                control_image=control_image,
                preprocessed_image=preprocessed_image,
            ),
            auth_kwargs={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
        )
        output_image = await handle_bfl_synchronous_operation(operation, node_id=cls.hidden.unique_id)
        return IO.NodeOutput(output_image)


class BFLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            FluxProUltraImageNode,
            # FluxProImageNode,
            FluxKontextProImageNode,
            FluxKontextMaxImageNode,
            FluxProExpandNode,
            FluxProFillNode,
            FluxProCannyNode,
            FluxProDepthNode,
        ]


async def comfy_entrypoint() -> BFLExtension:
    return BFLExtension()
