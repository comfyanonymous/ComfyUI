from inspect import cleandoc
from typing import Optional

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.bfl_api import (
    BFLFluxExpandImageRequest,
    BFLFluxFillImageRequest,
    BFLFluxKontextProGenerateRequest,
    BFLFluxProGenerateRequest,
    BFLFluxProGenerateResponse,
    BFLFluxProUltraGenerateRequest,
    BFLFluxStatusResponse,
    BFLStatus,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    poll_op,
    resize_mask_to_image,
    sync_op,
    tensor_to_base64_string,
    validate_aspect_ratio_string,
    validate_string,
)


def convert_mask_to_image(mask: torch.Tensor):
    """
    Make mask have the expected amount of dims (4) and channels (3) to be recognized as an image.
    """
    mask = mask.unsqueeze(-1)
    mask = torch.cat([mask] * 3, dim=-1)
    return mask


class FluxProUltraImageNode(IO.ComfyNode):
    """
    Generates images using Flux Pro 1.1 Ultra via api based on prompt and resolution.
    """

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
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
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
        image_prompt: Optional[torch.Tensor] = None,
        image_prompt_strength: float = 0.1,
    ) -> IO.NodeOutput:
        if image_prompt is None:
            validate_string(prompt, strip_whitespace=False)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/flux-pro-1.1-ultra/generate", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxProUltraGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                seed=seed,
                aspect_ratio=aspect_ratio,
                raw=raw,
                image_prompt=(image_prompt if image_prompt is None else tensor_to_base64_string(image_prompt)),
                image_prompt_strength=(None if image_prompt is None else round(image_prompt_strength, 2)),
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class FluxKontextProImageNode(IO.ComfyNode):
    """
    Edits images using Flux.1 Kontext [pro] via api based on prompt and aspect ratio.
    """

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
        input_image: Optional[torch.Tensor] = None,
        seed=0,
        prompt_upsampling=False,
    ) -> IO.NodeOutput:
        validate_aspect_ratio_string(aspect_ratio, (1, 4), (4, 1))
        if input_image is None:
            validate_string(prompt, strip_whitespace=False)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=cls.BFL_PATH, method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxKontextProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                guidance=round(guidance, 1),
                steps=steps,
                seed=seed,
                aspect_ratio=aspect_ratio,
                input_image=(input_image if input_image is None else tensor_to_base64_string(input_image)),
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


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
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
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
        image_prompt = image_prompt if image_prompt is None else tensor_to_base64_string(image_prompt)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.1/generate",
                method="POST",
            ),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                width=width,
                height=height,
                seed=seed,
                image_prompt=image_prompt,
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


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
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
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
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/flux-pro-1.0-expand/generate", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxExpandImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                steps=steps,
                guidance=guidance,
                seed=seed,
                image=tensor_to_base64_string(image),
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


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
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
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
        mask = tensor_to_base64_string(convert_mask_to_image(mask))
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/flux-pro-1.0-fill/generate", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxFillImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                steps=steps,
                guidance=guidance,
                seed=seed,
                image=tensor_to_base64_string(image[:, :, :, :3]),  # make sure image will have alpha channel removed
                mask=mask,
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


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
        ]


async def comfy_entrypoint() -> BFLExtension:
    return BFLExtension()
