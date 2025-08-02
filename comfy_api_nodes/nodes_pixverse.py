from inspect import cleandoc
from typing import Optional
from comfy_api_nodes.apis.pixverse_api import (
    PixverseTextVideoRequest,
    PixverseImageVideoRequest,
    PixverseTransitionVideoRequest,
    PixverseImageUploadResponse,
    PixverseVideoResponse,
    PixverseGenerationStatusResponse,
    PixverseAspectRatio,
    PixverseQuality,
    PixverseDuration,
    PixverseMotionMode,
    PixverseStatus,
    PixverseIO,
    pixverse_templates,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    tensor_to_bytesio,
    validate_string,
)
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile

import torch
import aiohttp
from io import BytesIO


AVERAGE_DURATION_T2V = 32
AVERAGE_DURATION_I2V = 30
AVERAGE_DURATION_T2T = 52


def get_video_url_from_response(
    response: PixverseGenerationStatusResponse,
) -> Optional[str]:
    if response.Resp is None or response.Resp.url is None:
        return None
    return str(response.Resp.url)


async def upload_image_to_pixverse(image: torch.Tensor, auth_kwargs=None):
    # first, upload image to Pixverse and get image id to use in actual generation call
    files = {"image": tensor_to_bytesio(image)}
    operation = SynchronousOperation(
        endpoint=ApiEndpoint(
            path="/proxy/pixverse/image/upload",
            method=HttpMethod.POST,
            request_model=EmptyRequest,
            response_model=PixverseImageUploadResponse,
        ),
        request=EmptyRequest(),
        files=files,
        content_type="multipart/form-data",
        auth_kwargs=auth_kwargs,
    )
    response_upload: PixverseImageUploadResponse = await operation.execute()

    if response_upload.Resp is None:
        raise Exception(
            f"PixVerse image upload request failed: '{response_upload.ErrMsg}'"
        )

    return response_upload.Resp.img_id


class PixverseTemplateNode:
    """
    Select template for PixVerse Video generation.
    """

    RETURN_TYPES = (PixverseIO.TEMPLATE,)
    RETURN_NAMES = ("pixverse_template",)
    FUNCTION = "create_template"
    CATEGORY = "api node/video/PixVerse"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template": (list(pixverse_templates.keys()),),
            }
        }

    def create_template(self, template: str):
        template_id = pixverse_templates.get(template, None)
        if template_id is None:
            raise Exception(f"Template '{template}' is not recognized.")
        # just return the integer
        return (template_id,)


class PixverseTextToVideoNode(ComfyNodeABC):
    """
    Generates videos based on prompt and output_size.
    """

    RETURN_TYPES = (IO.VIDEO,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/video/PixVerse"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the video generation",
                    },
                ),
                "aspect_ratio": ([ratio.value for ratio in PixverseAspectRatio],),
                "quality": (
                    [resolution.value for resolution in PixverseQuality],
                    {
                        "default": PixverseQuality.res_540p,
                    },
                ),
                "duration_seconds": ([dur.value for dur in PixverseDuration],),
                "motion_mode": ([mode.value for mode in PixverseMotionMode],),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                        "tooltip": "Seed for video generation.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
                "pixverse_template": (
                    PixverseIO.TEMPLATE,
                    {
                        "tooltip": "An optional template to influence style of generation, created by the PixVerse Template node."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        aspect_ratio: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
        pixverse_template: int = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        # 1080p is limited to 5 seconds duration
        # only normal motion_mode supported for 1080p or for non-5 second duration
        if quality == PixverseQuality.res_1080p:
            motion_mode = PixverseMotionMode.normal
            duration_seconds = PixverseDuration.dur_5
        elif duration_seconds != PixverseDuration.dur_5:
            motion_mode = PixverseMotionMode.normal

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/pixverse/video/text/generate",
                method=HttpMethod.POST,
                request_model=PixverseTextVideoRequest,
                response_model=PixverseVideoResponse,
            ),
            request=PixverseTextVideoRequest(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                quality=quality,
                duration=duration_seconds,
                motion_mode=motion_mode,
                negative_prompt=negative_prompt if negative_prompt else None,
                template_id=pixverse_template,
                seed=seed,
            ),
            auth_kwargs=kwargs,
        )
        response_api = await operation.execute()

        if response_api.Resp is None:
            raise Exception(f"PixVerse request failed: '{response_api.ErrMsg}'")

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/pixverse/video/result/{response_api.Resp.video_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=PixverseGenerationStatusResponse,
            ),
            completed_statuses=[PixverseStatus.successful],
            failed_statuses=[
                PixverseStatus.contents_moderation,
                PixverseStatus.failed,
                PixverseStatus.deleted,
            ],
            status_extractor=lambda x: x.Resp.status,
            auth_kwargs=kwargs,
            node_id=unique_id,
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_T2V,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.Resp.url) as vid_response:
                return (VideoFromFile(BytesIO(await vid_response.content.read())),)


class PixverseImageToVideoNode(ComfyNodeABC):
    """
    Generates videos based on prompt and output_size.
    """

    RETURN_TYPES = (IO.VIDEO,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/video/PixVerse"

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
                        "tooltip": "Prompt for the video generation",
                    },
                ),
                "quality": (
                    [resolution.value for resolution in PixverseQuality],
                    {
                        "default": PixverseQuality.res_540p,
                    },
                ),
                "duration_seconds": ([dur.value for dur in PixverseDuration],),
                "motion_mode": ([mode.value for mode in PixverseMotionMode],),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                        "tooltip": "Seed for video generation.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
                "pixverse_template": (
                    PixverseIO.TEMPLATE,
                    {
                        "tooltip": "An optional template to influence style of generation, created by the PixVerse Template node."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
        prompt: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
        pixverse_template: int = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        img_id = await upload_image_to_pixverse(image, auth_kwargs=kwargs)

        # 1080p is limited to 5 seconds duration
        # only normal motion_mode supported for 1080p or for non-5 second duration
        if quality == PixverseQuality.res_1080p:
            motion_mode = PixverseMotionMode.normal
            duration_seconds = PixverseDuration.dur_5
        elif duration_seconds != PixverseDuration.dur_5:
            motion_mode = PixverseMotionMode.normal

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/pixverse/video/img/generate",
                method=HttpMethod.POST,
                request_model=PixverseImageVideoRequest,
                response_model=PixverseVideoResponse,
            ),
            request=PixverseImageVideoRequest(
                img_id=img_id,
                prompt=prompt,
                quality=quality,
                duration=duration_seconds,
                motion_mode=motion_mode,
                negative_prompt=negative_prompt if negative_prompt else None,
                template_id=pixverse_template,
                seed=seed,
            ),
            auth_kwargs=kwargs,
        )
        response_api = await operation.execute()

        if response_api.Resp is None:
            raise Exception(f"PixVerse request failed: '{response_api.ErrMsg}'")

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/pixverse/video/result/{response_api.Resp.video_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=PixverseGenerationStatusResponse,
            ),
            completed_statuses=[PixverseStatus.successful],
            failed_statuses=[
                PixverseStatus.contents_moderation,
                PixverseStatus.failed,
                PixverseStatus.deleted,
            ],
            status_extractor=lambda x: x.Resp.status,
            auth_kwargs=kwargs,
            node_id=unique_id,
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_I2V,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.Resp.url) as vid_response:
                return (VideoFromFile(BytesIO(await vid_response.content.read())),)


class PixverseTransitionVideoNode(ComfyNodeABC):
    """
    Generates videos based on prompt and output_size.
    """

    RETURN_TYPES = (IO.VIDEO,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/video/PixVerse"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "first_frame": (IO.IMAGE,),
                "last_frame": (IO.IMAGE,),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the video generation",
                    },
                ),
                "quality": (
                    [resolution.value for resolution in PixverseQuality],
                    {
                        "default": PixverseQuality.res_540p,
                    },
                ),
                "duration_seconds": ([dur.value for dur in PixverseDuration],),
                "motion_mode": ([mode.value for mode in PixverseMotionMode],),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                        "tooltip": "Seed for video generation.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        prompt: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        first_frame_id = await upload_image_to_pixverse(first_frame, auth_kwargs=kwargs)
        last_frame_id = await upload_image_to_pixverse(last_frame, auth_kwargs=kwargs)

        # 1080p is limited to 5 seconds duration
        # only normal motion_mode supported for 1080p or for non-5 second duration
        if quality == PixverseQuality.res_1080p:
            motion_mode = PixverseMotionMode.normal
            duration_seconds = PixverseDuration.dur_5
        elif duration_seconds != PixverseDuration.dur_5:
            motion_mode = PixverseMotionMode.normal

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/pixverse/video/transition/generate",
                method=HttpMethod.POST,
                request_model=PixverseTransitionVideoRequest,
                response_model=PixverseVideoResponse,
            ),
            request=PixverseTransitionVideoRequest(
                first_frame_img=first_frame_id,
                last_frame_img=last_frame_id,
                prompt=prompt,
                quality=quality,
                duration=duration_seconds,
                motion_mode=motion_mode,
                negative_prompt=negative_prompt if negative_prompt else None,
                seed=seed,
            ),
            auth_kwargs=kwargs,
        )
        response_api = await operation.execute()

        if response_api.Resp is None:
            raise Exception(f"PixVerse request failed: '{response_api.ErrMsg}'")

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/pixverse/video/result/{response_api.Resp.video_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=PixverseGenerationStatusResponse,
            ),
            completed_statuses=[PixverseStatus.successful],
            failed_statuses=[
                PixverseStatus.contents_moderation,
                PixverseStatus.failed,
                PixverseStatus.deleted,
            ],
            status_extractor=lambda x: x.Resp.status,
            auth_kwargs=kwargs,
            node_id=unique_id,
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_T2V,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.Resp.url) as vid_response:
                return (VideoFromFile(BytesIO(await vid_response.content.read())),)


NODE_CLASS_MAPPINGS = {
    "PixverseTextToVideoNode": PixverseTextToVideoNode,
    "PixverseImageToVideoNode": PixverseImageToVideoNode,
    "PixverseTransitionVideoNode": PixverseTransitionVideoNode,
    "PixverseTemplateNode": PixverseTemplateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixverseTextToVideoNode": "PixVerse Text to Video",
    "PixverseImageToVideoNode": "PixVerse Image to Video",
    "PixverseTransitionVideoNode": "PixVerse Transition Video",
    "PixverseTemplateNode": "PixVerse Template",
}
