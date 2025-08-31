from inspect import cleandoc
from typing import Optional
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io as comfy_io
from comfy_api_nodes.apis.stability_api import (
    StabilityUpscaleConservativeRequest,
    StabilityUpscaleCreativeRequest,
    StabilityAsyncResponse,
    StabilityResultsGetResponse,
    StabilityStable3_5Request,
    StabilityStableUltraRequest,
    StabilityStableUltraResponse,
    StabilityAspectRatio,
    Stability_SD3_5_Model,
    Stability_SD3_5_GenerationMode,
    get_stability_style_presets,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    bytesio_to_image_tensor,
    tensor_to_bytesio,
    validate_string,
)

import torch
import base64
from io import BytesIO
from enum import Enum


class StabilityPollStatus(str, Enum):
    finished = "finished"
    in_progress = "in_progress"
    failed = "failed"


def get_async_dummy_status(x: StabilityResultsGetResponse):
    if x.name is not None or x.errors is not None:
        return StabilityPollStatus.failed
    elif x.finish_reason is not None:
        return StabilityPollStatus.finished
    return StabilityPollStatus.in_progress


class StabilityStableImageUltraNode(comfy_io.ComfyNode):
    """
    Generates images synchronously based on prompt and resolution.
    """

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="StabilityStableImageUltraNode",
            display_name="Stability AI Stable Image Ultra",
            category="api node/image/Stability AI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What you wish to see in the output image. A strong, descriptive prompt that clearly defines" +
                                    "elements, colors, and subjects will lead to better results. " +
                                    "To control the weight of a given word use the format `(word:weight)`," +
                                    "where `word` is the word you'd like to control the weight of and `weight`" +
                                    "is a value between 0 and 1. For example: `The sky was a crisp (blue:0.3) and (green:0.8)`" +
                                    "would convey a sky that was blue and green, but more green than blue.",
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=[x.value for x in StabilityAspectRatio],
                    default=StabilityAspectRatio.ratio_1_1.value,
                    tooltip="Aspect ratio of generated image.",
                ),
                comfy_io.Combo.Input(
                    "style_preset",
                    options=get_stability_style_presets(),
                    tooltip="Optional desired style of generated image.",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967294,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                comfy_io.Image.Input(
                    "image",
                    optional=True,
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    default="",
                    tooltip="A blurb of text describing what you do not wish to see in the output image. This is an advanced feature.",
                    force_input=True,
                    optional=True,
                ),
                comfy_io.Float.Input(
                    "image_denoise",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Denoise of input image; 0.0 yields image identical to input, 1.0 is as if no image was provided at all.",
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Image.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        style_preset: str,
        seed: int,
        image: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        image_denoise: Optional[float] = 0.5,
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        # prepare image binary if image present
        image_binary = None
        if image is not None:
            image_binary = tensor_to_bytesio(image, total_pixels=1504*1504).read()
        else:
            image_denoise = None

        if not negative_prompt:
            negative_prompt = None
        if style_preset == "None":
            style_preset = None

        files = {
            "image": image_binary
        }

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/stability/v2beta/stable-image/generate/ultra",
                method=HttpMethod.POST,
                request_model=StabilityStableUltraRequest,
                response_model=StabilityStableUltraResponse,
            ),
            request=StabilityStableUltraRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                seed=seed,
                strength=image_denoise,
                style_preset=style_preset,
            ),
            files=files,
            content_type="multipart/form-data",
            auth_kwargs=auth,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stable Image Ultra generation failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return comfy_io.NodeOutput(returned_image)


class StabilityStableImageSD_3_5Node(comfy_io.ComfyNode):
    """
    Generates images synchronously based on prompt and resolution.
    """

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="StabilityStableImageSD_3_5Node",
            display_name="Stability AI Stable Diffusion 3.5 Image",
            category="api node/image/Stability AI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results.",
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=[x.value for x in Stability_SD3_5_Model],
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=[x.value for x in StabilityAspectRatio],
                    default=StabilityAspectRatio.ratio_1_1.value,
                    tooltip="Aspect ratio of generated image.",
                ),
                comfy_io.Combo.Input(
                    "style_preset",
                    options=get_stability_style_presets(),
                    tooltip="Optional desired style of generated image.",
                ),
                comfy_io.Float.Input(
                    "cfg_scale",
                    default=4.0,
                    min=1.0,
                    max=10.0,
                    step=0.1,
                    tooltip="How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967294,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                comfy_io.Image.Input(
                    "image",
                    optional=True,
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    default="",
                    tooltip="Keywords of what you do not wish to see in the output image. This is an advanced feature.",
                    force_input=True,
                    optional=True,
                ),
                comfy_io.Float.Input(
                    "image_denoise",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Denoise of input image; 0.0 yields image identical to input, 1.0 is as if no image was provided at all.",
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Image.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        aspect_ratio: str,
        style_preset: str,
        seed: int,
        cfg_scale: float,
        image: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        image_denoise: Optional[float] = 0.5,
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        # prepare image binary if image present
        image_binary = None
        mode = Stability_SD3_5_GenerationMode.text_to_image
        if image is not None:
            image_binary = tensor_to_bytesio(image, total_pixels=1504*1504).read()
            mode = Stability_SD3_5_GenerationMode.image_to_image
            aspect_ratio = None
        else:
            image_denoise = None

        if not negative_prompt:
            negative_prompt = None
        if style_preset == "None":
            style_preset = None

        files = {
            "image": image_binary
        }

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/stability/v2beta/stable-image/generate/sd3",
                method=HttpMethod.POST,
                request_model=StabilityStable3_5Request,
                response_model=StabilityStableUltraResponse,
            ),
            request=StabilityStable3_5Request(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                seed=seed,
                strength=image_denoise,
                style_preset=style_preset,
                cfg_scale=cfg_scale,
                model=model,
                mode=mode,
            ),
            files=files,
            content_type="multipart/form-data",
            auth_kwargs=auth,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stable Diffusion 3.5 Image generation failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return comfy_io.NodeOutput(returned_image)


class StabilityUpscaleConservativeNode(comfy_io.ComfyNode):
    """
    Upscale image with minimal alterations to 4K resolution.
    """

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="StabilityUpscaleConservativeNode",
            display_name="Stability AI Upscale Conservative",
            category="api node/image/Stability AI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.Image.Input("image"),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results.",
                ),
                comfy_io.Float.Input(
                    "creativity",
                    default=0.35,
                    min=0.2,
                    max=0.5,
                    step=0.01,
                    tooltip="Controls the likelihood of creating additional details not heavily conditioned by the init image.",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967294,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    default="",
                    tooltip="Keywords of what you do not wish to see in the output image. This is an advanced feature.",
                    force_input=True,
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Image.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        creativity: float,
        seed: int,
        negative_prompt: str = "",
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        image_binary = tensor_to_bytesio(image, total_pixels=1024*1024).read()

        if not negative_prompt:
            negative_prompt = None

        files = {
            "image": image_binary
        }

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/stability/v2beta/stable-image/upscale/conservative",
                method=HttpMethod.POST,
                request_model=StabilityUpscaleConservativeRequest,
                response_model=StabilityStableUltraResponse,
            ),
            request=StabilityUpscaleConservativeRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                creativity=round(creativity,2),
                seed=seed,
            ),
            files=files,
            content_type="multipart/form-data",
            auth_kwargs=auth,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stability Upscale Conservative generation failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return comfy_io.NodeOutput(returned_image)


class StabilityUpscaleCreativeNode(comfy_io.ComfyNode):
    """
    Upscale image with minimal alterations to 4K resolution.
    """

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="StabilityUpscaleCreativeNode",
            display_name="Stability AI Upscale Creative",
            category="api node/image/Stability AI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.Image.Input("image"),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results.",
                ),
                comfy_io.Float.Input(
                    "creativity",
                    default=0.3,
                    min=0.1,
                    max=0.5,
                    step=0.01,
                    tooltip="Controls the likelihood of creating additional details not heavily conditioned by the init image.",
                ),
                comfy_io.Combo.Input(
                    "style_preset",
                    options=get_stability_style_presets(),
                    tooltip="Optional desired style of generated image.",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967294,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    default="",
                    tooltip="Keywords of what you do not wish to see in the output image. This is an advanced feature.",
                    force_input=True,
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Image.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        creativity: float,
        style_preset: str,
        seed: int,
        negative_prompt: str = "",
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        image_binary = tensor_to_bytesio(image, total_pixels=1024*1024).read()

        if not negative_prompt:
            negative_prompt = None
        if style_preset == "None":
            style_preset = None

        files = {
            "image": image_binary
        }

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/stability/v2beta/stable-image/upscale/creative",
                method=HttpMethod.POST,
                request_model=StabilityUpscaleCreativeRequest,
                response_model=StabilityAsyncResponse,
            ),
            request=StabilityUpscaleCreativeRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                creativity=round(creativity,2),
                style_preset=style_preset,
                seed=seed,
            ),
            files=files,
            content_type="multipart/form-data",
            auth_kwargs=auth,
        )
        response_api = await operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/stability/v2beta/results/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=StabilityResultsGetResponse,
            ),
            poll_interval=3,
            completed_statuses=[StabilityPollStatus.finished],
            failed_statuses=[StabilityPollStatus.failed],
            status_extractor=lambda x: get_async_dummy_status(x),
            auth_kwargs=auth,
            node_id=cls.hidden.unique_id,
        )
        response_poll: StabilityResultsGetResponse = await operation.execute()

        if response_poll.finish_reason != "SUCCESS":
            raise Exception(f"Stability Upscale Creative generation failed: {response_poll.finish_reason}.")

        image_data = base64.b64decode(response_poll.result)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return comfy_io.NodeOutput(returned_image)


class StabilityUpscaleFastNode(comfy_io.ComfyNode):
    """
    Quickly upscales an image via Stability API call to 4x its original size; intended for upscaling low-quality/compressed images.
    """

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="StabilityUpscaleFastNode",
            display_name="Stability AI Upscale Fast",
            category="api node/image/Stability AI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.Image.Input("image"),
            ],
            outputs=[
                comfy_io.Image.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(cls, image: torch.Tensor) -> comfy_io.NodeOutput:
        image_binary = tensor_to_bytesio(image, total_pixels=4096*4096).read()

        files = {
            "image": image_binary
        }

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/stability/v2beta/stable-image/upscale/fast",
                method=HttpMethod.POST,
                request_model=EmptyRequest,
                response_model=StabilityStableUltraResponse,
            ),
            request=EmptyRequest(),
            files=files,
            content_type="multipart/form-data",
            auth_kwargs=auth,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stability Upscale Fast failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return comfy_io.NodeOutput(returned_image)


class StabilityExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            StabilityStableImageUltraNode,
            StabilityStableImageSD_3_5Node,
            StabilityUpscaleConservativeNode,
            StabilityUpscaleCreativeNode,
            StabilityUpscaleFastNode,
        ]


async def comfy_entrypoint() -> StabilityExtension:
    return StabilityExtension()
