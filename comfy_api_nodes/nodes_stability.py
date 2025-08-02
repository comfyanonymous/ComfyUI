from inspect import cleandoc
from comfy.comfy_types.node_typing import IO
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


class StabilityStableImageUltraNode:
    """
    Generates images synchronously based on prompt and resolution.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Stability AI"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What you wish to see in the output image. A strong, descriptive prompt that clearly defines" +
                                    "What you wish to see in the output image. A strong, descriptive prompt that clearly defines" +
                                    "elements, colors, and subjects will lead to better results. " +
                                    "To control the weight of a given word use the format `(word:weight)`," +
                                    "where `word` is the word you'd like to control the weight of and `weight`" +
                                    "is a value between 0 and 1. For example: `The sky was a crisp (blue:0.3) and (green:0.8)`" +
                                    "would convey a sky that was blue and green, but more green than blue."
                    },
                ),
                "aspect_ratio": ([x.value for x in StabilityAspectRatio],
                    {
                        "default": StabilityAspectRatio.ratio_1_1,
                        "tooltip": "Aspect ratio of generated image.",
                    },
                ),
                "style_preset": (get_stability_style_presets(),
                    {
                        "tooltip": "Optional desired style of generated image.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967294,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "optional": {
                "image": (IO.IMAGE,),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "A blurb of text describing what you do not wish to see in the output image. This is an advanced feature."
                    },
                ),
                "image_denoise": (
                    IO.FLOAT,
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoise of input image; 0.0 yields image identical to input, 1.0 is as if no image was provided at all.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(self, prompt: str, aspect_ratio: str, style_preset: str, seed: int,
                 negative_prompt: str=None, image: torch.Tensor = None, image_denoise: float=None,
                 **kwargs):
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
            auth_kwargs=kwargs,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stable Image Ultra generation failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return (returned_image,)


class StabilityStableImageSD_3_5Node:
    """
    Generates images synchronously based on prompt and resolution.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Stability AI"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results."
                    },
                ),
                "model": ([x.value for x in Stability_SD3_5_Model],),
                "aspect_ratio": ([x.value for x in StabilityAspectRatio],
                    {
                        "default": StabilityAspectRatio.ratio_1_1,
                        "tooltip": "Aspect ratio of generated image.",
                    },
                ),
                "style_preset": (get_stability_style_presets(),
                    {
                        "tooltip": "Optional desired style of generated image.",
                    },
                ),
                "cfg_scale": (
                    IO.FLOAT,
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967294,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "optional": {
                "image": (IO.IMAGE,),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Keywords of what you do not wish to see in the output image. This is an advanced feature."
                    },
                ),
                "image_denoise": (
                    IO.FLOAT,
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoise of input image; 0.0 yields image identical to input, 1.0 is as if no image was provided at all.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(self, model: str, prompt: str, aspect_ratio: str, style_preset: str, seed: int, cfg_scale: float,
                 negative_prompt: str=None, image: torch.Tensor = None, image_denoise: float=None,
                 **kwargs):
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
            auth_kwargs=kwargs,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stable Diffusion 3.5 Image generation failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return (returned_image,)


class StabilityUpscaleConservativeNode:
    """
    Upscale image with minimal alterations to 4K resolution.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Stability AI"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results."
                    },
                ),
                "creativity": (
                    IO.FLOAT,
                    {
                        "default": 0.35,
                        "min": 0.2,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Controls the likelihood of creating additional details not heavily conditioned by the init image.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967294,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Keywords of what you do not wish to see in the output image. This is an advanced feature."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(self, image: torch.Tensor, prompt: str, creativity: float, seed: int, negative_prompt: str=None,
                 **kwargs):
        validate_string(prompt, strip_whitespace=False)
        image_binary = tensor_to_bytesio(image, total_pixels=1024*1024).read()

        if not negative_prompt:
            negative_prompt = None

        files = {
            "image": image_binary
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
            auth_kwargs=kwargs,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stability Upscale Conservative generation failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return (returned_image,)


class StabilityUpscaleCreativeNode:
    """
    Upscale image with minimal alterations to 4K resolution.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Stability AI"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results."
                    },
                ),
                "creativity": (
                    IO.FLOAT,
                    {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Controls the likelihood of creating additional details not heavily conditioned by the init image.",
                    },
                ),
                "style_preset": (get_stability_style_presets(),
                    {
                        "tooltip": "Optional desired style of generated image.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967294,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Keywords of what you do not wish to see in the output image. This is an advanced feature."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(self, image: torch.Tensor, prompt: str, creativity: float, style_preset: str, seed: int, negative_prompt: str=None,
                 **kwargs):
        validate_string(prompt, strip_whitespace=False)
        image_binary = tensor_to_bytesio(image, total_pixels=1024*1024).read()

        if not negative_prompt:
            negative_prompt = None
        if style_preset == "None":
            style_preset = None

        files = {
            "image": image_binary
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
            auth_kwargs=kwargs,
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
            auth_kwargs=kwargs,
        )
        response_poll: StabilityResultsGetResponse = await operation.execute()

        if response_poll.finish_reason != "SUCCESS":
            raise Exception(f"Stability Upscale Creative generation failed: {response_poll.finish_reason}.")

        image_data = base64.b64decode(response_poll.result)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return (returned_image,)


class StabilityUpscaleFastNode:
    """
    Quickly upscales an image via Stability API call to 4x its original size; intended for upscaling low-quality/compressed images.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Stability AI"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
            },
            "optional": {
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(self, image: torch.Tensor, **kwargs):
        image_binary = tensor_to_bytesio(image, total_pixels=4096*4096).read()

        files = {
            "image": image_binary
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
            auth_kwargs=kwargs,
        )
        response_api = await operation.execute()

        if response_api.finish_reason != "SUCCESS":
            raise Exception(f"Stability Upscale Fast failed: {response_api.finish_reason}.")

        image_data = base64.b64decode(response_api.image)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))

        return (returned_image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "StabilityStableImageUltraNode": StabilityStableImageUltraNode,
    "StabilityStableImageSD_3_5Node": StabilityStableImageSD_3_5Node,
    "StabilityUpscaleConservativeNode": StabilityUpscaleConservativeNode,
    "StabilityUpscaleCreativeNode": StabilityUpscaleCreativeNode,
    "StabilityUpscaleFastNode": StabilityUpscaleFastNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StabilityStableImageUltraNode": "Stability AI Stable Image Ultra",
    "StabilityStableImageSD_3_5Node": "Stability AI Stable Diffusion 3.5 Image",
    "StabilityUpscaleConservativeNode": "Stability AI Upscale Conservative",
    "StabilityUpscaleCreativeNode": "Stability AI Upscale Creative",
    "StabilityUpscaleFastNode": "Stability AI Upscale Fast",
}
