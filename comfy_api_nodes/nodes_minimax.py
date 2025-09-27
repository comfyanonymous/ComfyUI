from inspect import cleandoc
from typing import Optional
import logging
import torch

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io as comfy_io
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api_nodes.apis import (
    MinimaxVideoGenerationRequest,
    MinimaxVideoGenerationResponse,
    MinimaxFileRetrieveResponse,
    MinimaxTaskResultResponse,
    SubjectReferenceItem,
    MiniMaxModel,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    download_url_to_bytesio,
    upload_images_to_comfyapi,
    validate_string,
)
from server import PromptServer


I2V_AVERAGE_DURATION = 114
T2V_AVERAGE_DURATION = 234


async def _generate_mm_video(
    *,
    auth: dict[str, str],
    node_id: str,
    prompt_text: str,
    seed: int,
    model: str,
    image: Optional[torch.Tensor] = None,   # used for ImageToVideo
    subject: Optional[torch.Tensor] = None, # used for SubjectToVideo
    average_duration: Optional[int] = None,
) -> comfy_io.NodeOutput:
    if image is None:
        validate_string(prompt_text, field_name="prompt_text")
    # upload image, if passed in
    image_url = None
    if image is not None:
        image_url = (await upload_images_to_comfyapi(image, max_images=1, auth_kwargs=auth))[0]

    # TODO: figure out how to deal with subject properly, API returns invalid params when using S2V-01 model
    subject_reference = None
    if subject is not None:
        subject_url = (await upload_images_to_comfyapi(subject, max_images=1, auth_kwargs=auth))[0]
        subject_reference = [SubjectReferenceItem(image=subject_url)]


    video_generate_operation = SynchronousOperation(
        endpoint=ApiEndpoint(
            path="/proxy/minimax/video_generation",
            method=HttpMethod.POST,
            request_model=MinimaxVideoGenerationRequest,
            response_model=MinimaxVideoGenerationResponse,
        ),
        request=MinimaxVideoGenerationRequest(
            model=MiniMaxModel(model),
            prompt=prompt_text,
            callback_url=None,
            first_frame_image=image_url,
            subject_reference=subject_reference,
            prompt_optimizer=None,
        ),
        auth_kwargs=auth,
    )
    response = await video_generate_operation.execute()

    task_id = response.task_id
    if not task_id:
        raise Exception(f"MiniMax generation failed: {response.base_resp}")

    video_generate_operation = PollingOperation(
        poll_endpoint=ApiEndpoint(
            path="/proxy/minimax/query/video_generation",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=MinimaxTaskResultResponse,
            query_params={"task_id": task_id},
        ),
        completed_statuses=["Success"],
        failed_statuses=["Fail"],
        status_extractor=lambda x: x.status.value,
        estimated_duration=average_duration,
        node_id=node_id,
        auth_kwargs=auth,
    )
    task_result = await video_generate_operation.execute()

    file_id = task_result.file_id
    if file_id is None:
        raise Exception("Request was not successful. Missing file ID.")
    file_retrieve_operation = SynchronousOperation(
        endpoint=ApiEndpoint(
            path="/proxy/minimax/files/retrieve",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=MinimaxFileRetrieveResponse,
            query_params={"file_id": int(file_id)},
        ),
        request=EmptyRequest(),
        auth_kwargs=auth,
    )
    file_result = await file_retrieve_operation.execute()

    file_url = file_result.file.download_url
    if file_url is None:
        raise Exception(
            f"No video was found in the response. Full response: {file_result.model_dump()}"
        )
    logging.info("Generated video URL: %s", file_url)
    if node_id:
        if hasattr(file_result.file, "backup_download_url"):
            message = f"Result URL: {file_url}\nBackup URL: {file_result.file.backup_download_url}"
        else:
            message = f"Result URL: {file_url}"
        PromptServer.instance.send_progress_text(message, node_id)

    # Download and return as VideoFromFile
    video_io = await download_url_to_bytesio(file_url)
    if video_io is None:
        error_msg = f"Failed to download video from {file_url}"
        logging.error(error_msg)
        raise Exception(error_msg)
    return comfy_io.NodeOutput(VideoFromFile(video_io))


class MinimaxTextToVideoNode(comfy_io.ComfyNode):
    """
    Generates videos synchronously based on a prompt, and optional parameters using MiniMax's API.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="MinimaxTextToVideoNode",
            display_name="MiniMax Text to Video",
            category="api node/video/MiniMax",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=["T2V-01", "T2V-01-Director"],
                    default="T2V-01",
                    tooltip="Model to use for video generation",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
            ],
            outputs=[comfy_io.Video.Output()],
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
        prompt_text: str,
        model: str = "T2V-01",
        seed: int = 0,
    ) -> comfy_io.NodeOutput:
        return await _generate_mm_video(
            auth={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
            node_id=cls.hidden.unique_id,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=None,
            subject=None,
            average_duration=T2V_AVERAGE_DURATION,
        )


class MinimaxImageToVideoNode(comfy_io.ComfyNode):
    """
    Generates videos synchronously based on an image and prompt, and optional parameters using MiniMax's API.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="MinimaxImageToVideoNode",
            display_name="MiniMax Image to Video",
            category="api node/video/MiniMax",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.Image.Input(
                    "image",
                    tooltip="Image to use as first frame of video generation",
                ),
                comfy_io.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=["I2V-01-Director", "I2V-01", "I2V-01-live"],
                    default="I2V-01",
                    tooltip="Model to use for video generation",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
            ],
            outputs=[comfy_io.Video.Output()],
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
        prompt_text: str,
        model: str = "I2V-01",
        seed: int = 0,
    ) -> comfy_io.NodeOutput:
        return await _generate_mm_video(
            auth={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
            node_id=cls.hidden.unique_id,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=image,
            subject=None,
            average_duration=I2V_AVERAGE_DURATION,
        )


class MinimaxSubjectToVideoNode(comfy_io.ComfyNode):
    """
    Generates videos synchronously based on an image and prompt, and optional parameters using MiniMax's API.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="MinimaxSubjectToVideoNode",
            display_name="MiniMax Subject to Video",
            category="api node/video/MiniMax",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.Image.Input(
                    "subject",
                    tooltip="Image of subject to reference for video generation",
                ),
                comfy_io.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=["S2V-01"],
                    default="S2V-01",
                    tooltip="Model to use for video generation",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
            ],
            outputs=[comfy_io.Video.Output()],
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
        subject: torch.Tensor,
        prompt_text: str,
        model: str = "S2V-01",
        seed: int = 0,
    ) -> comfy_io.NodeOutput:
        return await _generate_mm_video(
            auth={
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
            node_id=cls.hidden.unique_id,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=None,
            subject=subject,
            average_duration=T2V_AVERAGE_DURATION,
        )


class MinimaxHailuoVideoNode(comfy_io.ComfyNode):
    """Generates videos from prompt, with optional start frame using the new MiniMax Hailuo-02 model."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="MinimaxHailuoVideoNode",
            display_name="MiniMax Hailuo Video",
            category="api node/video/MiniMax",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                comfy_io.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation.",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
                comfy_io.Image.Input(
                    "first_frame_image",
                    tooltip="Optional image to use as the first frame to generate a video.",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "prompt_optimizer",
                    default=True,
                    tooltip="Optimize prompt to improve generation quality when needed.",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "duration",
                    options=[6, 10],
                    default=6,
                    tooltip="The length of the output video in seconds.",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "resolution",
                    options=["768P", "1080P"],
                    default="768P",
                    tooltip="The dimensions of the video display. 1080p is 1920x1080, 768p is 1366x768.",
                    optional=True,
                ),
            ],
            outputs=[comfy_io.Video.Output()],
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
        prompt_text: str,
        seed: int = 0,
        first_frame_image: Optional[torch.Tensor] = None,  # used for ImageToVideo
        prompt_optimizer: bool = True,
        duration: int = 6,
        resolution: str = "768P",
        model: str = "MiniMax-Hailuo-02",
    ) -> comfy_io.NodeOutput:
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        if first_frame_image is None:
            validate_string(prompt_text, field_name="prompt_text")

        if model == "MiniMax-Hailuo-02" and resolution.upper() == "1080P" and duration != 6:
            raise Exception(
                "When model is MiniMax-Hailuo-02 and resolution is 1080P, duration is limited to 6 seconds."
            )

        # upload image, if passed in
        image_url = None
        if first_frame_image is not None:
            image_url = (await upload_images_to_comfyapi(first_frame_image, max_images=1, auth_kwargs=auth))[0]

        video_generate_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/minimax/video_generation",
                method=HttpMethod.POST,
                request_model=MinimaxVideoGenerationRequest,
                response_model=MinimaxVideoGenerationResponse,
            ),
            request=MinimaxVideoGenerationRequest(
                model=MiniMaxModel(model),
                prompt=prompt_text,
                callback_url=None,
                first_frame_image=image_url,
                prompt_optimizer=prompt_optimizer,
                duration=duration,
                resolution=resolution,
            ),
            auth_kwargs=auth,
        )
        response = await video_generate_operation.execute()

        task_id = response.task_id
        if not task_id:
            raise Exception(f"MiniMax generation failed: {response.base_resp}")

        average_duration = 120 if resolution == "768P" else 240
        video_generate_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path="/proxy/minimax/query/video_generation",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=MinimaxTaskResultResponse,
                query_params={"task_id": task_id},
            ),
            completed_statuses=["Success"],
            failed_statuses=["Fail"],
            status_extractor=lambda x: x.status.value,
            estimated_duration=average_duration,
            node_id=cls.hidden.unique_id,
            auth_kwargs=auth,
        )
        task_result = await video_generate_operation.execute()

        file_id = task_result.file_id
        if file_id is None:
            raise Exception("Request was not successful. Missing file ID.")
        file_retrieve_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/minimax/files/retrieve",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=MinimaxFileRetrieveResponse,
                query_params={"file_id": int(file_id)},
            ),
            request=EmptyRequest(),
            auth_kwargs=auth,
        )
        file_result = await file_retrieve_operation.execute()

        file_url = file_result.file.download_url
        if file_url is None:
            raise Exception(
                f"No video was found in the response. Full response: {file_result.model_dump()}"
            )
        logging.info(f"Generated video URL: {file_url}")
        if cls.hidden.unique_id:
            if hasattr(file_result.file, "backup_download_url"):
                message = f"Result URL: {file_url}\nBackup URL: {file_result.file.backup_download_url}"
            else:
                message = f"Result URL: {file_url}"
            PromptServer.instance.send_progress_text(message, cls.hidden.unique_id)

        video_io = await download_url_to_bytesio(file_url)
        if video_io is None:
            error_msg = f"Failed to download video from {file_url}"
            logging.error(error_msg)
            raise Exception(error_msg)
        return comfy_io.NodeOutput(VideoFromFile(video_io))


class MinimaxExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            MinimaxTextToVideoNode,
            MinimaxImageToVideoNode,
            # MinimaxSubjectToVideoNode,
            MinimaxHailuoVideoNode,
        ]


async def comfy_entrypoint() -> MinimaxExtension:
    return MinimaxExtension()
