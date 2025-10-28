from inspect import cleandoc
from typing import Optional
from typing_extensions import override
from io import BytesIO
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
from comfy_api_nodes.util import validate_string, tensor_to_bytesio
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ComfyExtension, IO

import torch
import aiohttp


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
    operation = SynchronousOperation(
        endpoint=ApiEndpoint(
            path="/proxy/pixverse/image/upload",
            method=HttpMethod.POST,
            request_model=EmptyRequest,
            response_model=PixverseImageUploadResponse,
        ),
        request=EmptyRequest(),
        files={"image": tensor_to_bytesio(image)},
        content_type="multipart/form-data",
        auth_kwargs=auth_kwargs,
    )
    response_upload: PixverseImageUploadResponse = await operation.execute()

    if response_upload.Resp is None:
        raise Exception(f"PixVerse image upload request failed: '{response_upload.ErrMsg}'")

    return response_upload.Resp.img_id


class PixverseTemplateNode(IO.ComfyNode):
    """
    Select template for PixVerse Video generation.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseTemplateNode",
            display_name="PixVerse Template",
            category="api node/video/PixVerse",
            inputs=[
                IO.Combo.Input("template", options=list(pixverse_templates.keys())),
            ],
            outputs=[IO.Custom(PixverseIO.TEMPLATE).Output(display_name="pixverse_template")],
        )

    @classmethod
    def execute(cls, template: str) -> IO.NodeOutput:
        template_id = pixverse_templates.get(template, None)
        if template_id is None:
            raise Exception(f"Template '{template}' is not recognized.")
        return IO.NodeOutput(template_id)


class PixverseTextToVideoNode(IO.ComfyNode):
    """
    Generates videos based on prompt and output_size.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseTextToVideoNode",
            display_name="PixVerse Text to Video",
            category="api node/video/PixVerse",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=PixverseAspectRatio,
                ),
                IO.Combo.Input(
                    "quality",
                    options=PixverseQuality,
                    default=PixverseQuality.res_540p,
                ),
                IO.Combo.Input(
                    "duration_seconds",
                    options=PixverseDuration,
                ),
                IO.Combo.Input(
                    "motion_mode",
                    options=PixverseMotionMode,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for video generation.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
                IO.Custom(PixverseIO.TEMPLATE).Input(
                    "pixverse_template",
                    tooltip="An optional template to influence style of generation, created by the PixVerse Template node.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
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
        aspect_ratio: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
        pixverse_template: int = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        # 1080p is limited to 5 seconds duration
        # only normal motion_mode supported for 1080p or for non-5 second duration
        if quality == PixverseQuality.res_1080p:
            motion_mode = PixverseMotionMode.normal
            duration_seconds = PixverseDuration.dur_5
        elif duration_seconds != PixverseDuration.dur_5:
            motion_mode = PixverseMotionMode.normal

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
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
            auth_kwargs=auth,
            node_id=cls.hidden.unique_id,
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_T2V,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.Resp.url) as vid_response:
                return IO.NodeOutput(VideoFromFile(BytesIO(await vid_response.content.read())))


class PixverseImageToVideoNode(IO.ComfyNode):
    """
    Generates videos based on prompt and output_size.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseImageToVideoNode",
            display_name="PixVerse Image to Video",
            category="api node/video/PixVerse",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "quality",
                    options=PixverseQuality,
                    default=PixverseQuality.res_540p,
                ),
                IO.Combo.Input(
                    "duration_seconds",
                    options=PixverseDuration,
                ),
                IO.Combo.Input(
                    "motion_mode",
                    options=PixverseMotionMode,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for video generation.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
                IO.Custom(PixverseIO.TEMPLATE).Input(
                    "pixverse_template",
                    tooltip="An optional template to influence style of generation, created by the PixVerse Template node.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
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
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
        pixverse_template: int = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        img_id = await upload_image_to_pixverse(image, auth_kwargs=auth)

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
            auth_kwargs=auth,
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
            auth_kwargs=auth,
            node_id=cls.hidden.unique_id,
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_I2V,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.Resp.url) as vid_response:
                return IO.NodeOutput(VideoFromFile(BytesIO(await vid_response.content.read())))


class PixverseTransitionVideoNode(IO.ComfyNode):
    """
    Generates videos based on prompt and output_size.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseTransitionVideoNode",
            display_name="PixVerse Transition Video",
            category="api node/video/PixVerse",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("first_frame"),
                IO.Image.Input("last_frame"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "quality",
                    options=PixverseQuality,
                    default=PixverseQuality.res_540p,
                ),
                IO.Combo.Input(
                    "duration_seconds",
                    options=PixverseDuration,
                ),
                IO.Combo.Input(
                    "motion_mode",
                    options=PixverseMotionMode,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for video generation.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
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
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        prompt: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        first_frame_id = await upload_image_to_pixverse(first_frame, auth_kwargs=auth)
        last_frame_id = await upload_image_to_pixverse(last_frame, auth_kwargs=auth)

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
            auth_kwargs=auth,
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
            auth_kwargs=auth,
            node_id=cls.hidden.unique_id,
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_T2V,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.Resp.url) as vid_response:
                return IO.NodeOutput(VideoFromFile(BytesIO(await vid_response.content.read())))


class PixVerseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PixverseTextToVideoNode,
            PixverseImageToVideoNode,
            PixverseTransitionVideoNode,
            PixverseTemplateNode,
        ]


async def comfy_entrypoint() -> PixVerseExtension:
    return PixVerseExtension()
