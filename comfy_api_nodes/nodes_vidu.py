from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.vidu import (
    SubjectReference,
    TaskCreationRequest,
    TaskCreationResponse,
    TaskResult,
    TaskStatusResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_images_aspect_ratio_closeness,
    validate_string,
)

VIDU_TEXT_TO_VIDEO = "/proxy/vidu/text2video"
VIDU_IMAGE_TO_VIDEO = "/proxy/vidu/img2video"
VIDU_REFERENCE_VIDEO = "/proxy/vidu/reference2video"
VIDU_START_END_VIDEO = "/proxy/vidu/start-end2video"
VIDU_GET_GENERATION_STATUS = "/proxy/vidu/tasks/%s/creations"


async def execute_task(
    cls: type[IO.ComfyNode],
    vidu_endpoint: str,
    payload: TaskCreationRequest,
) -> list[TaskResult]:
    task_creation_response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=vidu_endpoint, method="POST"),
        response_model=TaskCreationResponse,
        data=payload,
    )
    if task_creation_response.state == "failed":
        raise RuntimeError(f"Vidu request failed. Code: {task_creation_response.code}")
    response = await poll_op(
        cls,
        ApiEndpoint(path=VIDU_GET_GENERATION_STATUS % task_creation_response.task_id),
        response_model=TaskStatusResponse,
        status_extractor=lambda r: r.state,
        progress_extractor=lambda r: r.progress,
        max_poll_attempts=320,
    )
    if not response.creations:
        raise RuntimeError(
            f"Vidu request does not contain results. State: {response.state}, Error Code: {response.err_code}"
        )
    return response.creations


class ViduTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduTextToVideoNode",
            display_name="Vidu Text To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from a text prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16", "1:1"],
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if not prompt:
            raise ValueError("The prompt field is required and cannot be empty.")
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        results = await execute_task(cls, VIDU_TEXT_TO_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduImageToVideoNode",
            display_name="Vidu Image To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from image and optional prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.Image.Input(
                    "image",
                    tooltip="An image to be used as the start frame of the generated video",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) > 1:
            raise ValueError("Only one input image is allowed.")
        validate_image_aspect_ratio(image, (1, 4), (4, 1))
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = await upload_images_to_comfyapi(
            cls,
            image,
            max_images=1,
            mime_type="image/png",
        )
        results = await execute_task(cls, VIDU_IMAGE_TO_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduReferenceVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduReferenceVideoNode",
            display_name="Vidu Reference To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from multiple images and a prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.Image.Input(
                    "images",
                    tooltip="Images to use as references to generate a video with consistent subjects (max 7 images).",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16", "1:1"],
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        images: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if not prompt:
            raise ValueError("The prompt field is required and cannot be empty.")
        a = get_number_of_images(images)
        if a > 7:
            raise ValueError("Too many images, maximum allowed is 7.")
        for image in images:
            validate_image_aspect_ratio(image, (1, 4), (4, 1))
            validate_image_dimensions(image, min_width=128, min_height=128)
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = await upload_images_to_comfyapi(
            cls,
            images,
            max_images=7,
            mime_type="image/png",
        )
        results = await execute_task(cls, VIDU_REFERENCE_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduStartEndToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduStartEndToVideoNode",
            display_name="Vidu Start End To Video Generation",
            category="api node/video/Vidu",
            description="Generate a video from start and end frames and a prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.Image.Input(
                    "first_frame",
                    tooltip="Start frame",
                ),
                IO.Image.Input(
                    "end_frame",
                    tooltip="End frame",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        first_frame: Input.Image,
        end_frame: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        validate_images_aspect_ratio_closeness(first_frame, end_frame, min_rel=0.8, max_rel=1.25, strict=False)
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = [
            (await upload_images_to_comfyapi(cls, frame, max_images=1, mime_type="image/png"))[0]
            for frame in (first_frame, end_frame)
        ]
        results = await execute_task(cls, VIDU_START_END_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2TextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2TextToVideoNode",
            display_name="Vidu2 Text-to-Video Generation",
            category="api node/video/Vidu",
            description="Generate video from a text prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq2"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation, with a maximum length of 2000 characters.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=1,
                    max=10,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "3:4", "4:3", "1:1"]),
                IO.Combo.Input("resolution", options=["720p", "1080p"]),
                IO.Boolean.Input(
                    "background_music",
                    default=False,
                    tooltip="Whether to add background music to the generated video.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution"]),
                expr="""
                (
                  $is1080 := widgets.resolution = "1080p";
                  $base := $is1080 ? 0.1 : 0.075;
                  $perSec := $is1080 ? 0.05 : 0.025;
                  {"type":"usd","usd": $base + $perSec * (widgets.duration - 1)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        background_music: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2000)
        results = await execute_task(
            cls,
            VIDU_TEXT_TO_VIDEO,
            TaskCreationRequest(
                model=model,
                prompt=prompt,
                duration=duration,
                seed=seed,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                bgm=background_music,
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2ImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2ImageToVideoNode",
            display_name="Vidu2 Image-to-Video Generation",
            category="api node/video/Vidu",
            description="Generate a video from an image and an optional prompt.",
            inputs=[
                IO.Combo.Input("model", options=["viduq2-pro-fast", "viduq2-pro", "viduq2-turbo"]),
                IO.Image.Input(
                    "image",
                    tooltip="An image to be used as the start frame of the generated video.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="An optional text prompt for video generation (max 2000 characters).",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=1,
                    max=10,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720p", "1080p"],
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution"]),
                expr="""
                (
                  $m := widgets.model;
                  $d := widgets.duration;
                  $is1080 := widgets.resolution = "1080p";
                  $contains($m, "pro-fast")
                    ? (
                        $base := $is1080 ? 0.08 : 0.04;
                        $perSec := $is1080 ? 0.02 : 0.01;
                        {"type":"usd","usd": $base + $perSec * ($d - 1)}
                      )
                    : $contains($m, "pro")
                      ? (
                          $base := $is1080 ? 0.275 : 0.075;
                          $perSec := $is1080 ? 0.075 : 0.05;
                          {"type":"usd","usd": $base + $perSec * ($d - 1)}
                        )
                      : $contains($m, "turbo")
                        ? (
                            $is1080
                              ? {"type":"usd","usd": 0.175 + 0.05 * ($d - 1)}
                              : (
                                  $d <= 1 ? {"type":"usd","usd": 0.04}
                                  : $d <= 2 ? {"type":"usd","usd": 0.05}
                                  : {"type":"usd","usd": 0.05 + 0.05 * ($d - 2)}
                                )
                          )
                        : {"type":"usd","usd": 0.04}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) > 1:
            raise ValueError("Only one input image is allowed.")
        validate_image_aspect_ratio(image, (1, 4), (4, 1))
        validate_string(prompt, max_length=2000)
        results = await execute_task(
            cls,
            VIDU_IMAGE_TO_VIDEO,
            TaskCreationRequest(
                model=model,
                prompt=prompt,
                duration=duration,
                seed=seed,
                resolution=resolution,
                movement_amplitude=movement_amplitude,
                images=await upload_images_to_comfyapi(
                    cls,
                    image,
                    max_images=1,
                    mime_type="image/png",
                ),
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2ReferenceVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2ReferenceVideoNode",
            display_name="Vidu2 Reference-to-Video Generation",
            category="api node/video/Vidu",
            description="Generate a video from multiple reference images and a prompt.",
            inputs=[
                IO.Combo.Input("model", options=["viduq2"]),
                IO.Autogrow.Input(
                    "subjects",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_images"),
                        names=["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7"],
                        min=1,
                    ),
                    tooltip="For each subject, provide up to 3 reference images (7 images total across all subjects). "
                    "Reference them in prompts via @subject{subject_id}.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="When enabled, the video will include generated speech and background music "
                    "based on the prompt.",
                ),
                IO.Boolean.Input(
                    "audio",
                    default=False,
                    tooltip="When enabled video will contain generated speech and background music based on the prompt.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=1,
                    max=10,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "4:3", "3:4", "1:1"]),
                IO.Combo.Input("resolution", options=["720p", "1080p"]),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["audio", "duration", "resolution"]),
                expr="""
                (
                  $is1080 := widgets.resolution = "1080p";
                  $base := $is1080 ? 0.375 : 0.125;
                  $perSec := $is1080 ? 0.05 : 0.025;
                  $audioCost := widgets.audio = true ? 0.075 : 0;
                  {"type":"usd","usd": $base + $perSec * (widgets.duration - 1) + $audioCost}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        subjects: IO.Autogrow.Type,
        prompt: str,
        audio: bool,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2000)
        total_images = 0
        for i in subjects:
            if get_number_of_images(subjects[i]) > 3:
                raise ValueError("Maximum number of images per subject is 3.")
            for im in subjects[i]:
                total_images += 1
                validate_image_aspect_ratio(im, (1, 4), (4, 1))
                validate_image_dimensions(im, min_width=128, min_height=128)
        if total_images > 7:
            raise ValueError("Too many reference images; the maximum allowed is 7.")
        subjects_param: list[SubjectReference] = []
        for i in subjects:
            subjects_param.append(
                SubjectReference(
                    id=i,
                    images=await upload_images_to_comfyapi(
                        cls,
                        subjects[i],
                        max_images=3,
                        mime_type="image/png",
                        wait_label=f"Uploading reference images for {i}",
                    ),
                ),
            )
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            audio=audio,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
            subjects=subjects_param,
        )
        results = await execute_task(cls, VIDU_REFERENCE_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2StartEndToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2StartEndToVideoNode",
            display_name="Vidu2 Start/End Frame-to-Video Generation",
            category="api node/video/Vidu",
            description="Generate a video from a start frame, an end frame, and a prompt.",
            inputs=[
                IO.Combo.Input("model", options=["viduq2-pro-fast", "viduq2-pro", "viduq2-turbo"]),
                IO.Image.Input("first_frame"),
                IO.Image.Input("end_frame"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Prompt description (max 2000 characters).",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=2,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input("resolution", options=["720p", "1080p"]),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution"]),
                expr="""
                (
                  $m := widgets.model;
                  $d := widgets.duration;
                  $is1080 := widgets.resolution = "1080p";
                  $contains($m, "pro-fast")
                    ? (
                        $base := $is1080 ? 0.08 : 0.04;
                        $perSec := $is1080 ? 0.02 : 0.01;
                        {"type":"usd","usd": $base + $perSec * ($d - 1)}
                      )
                    : $contains($m, "pro")
                      ? (
                          $base := $is1080 ? 0.275 : 0.075;
                          $perSec := $is1080 ? 0.075 : 0.05;
                          {"type":"usd","usd": $base + $perSec * ($d - 1)}
                        )
                      : $contains($m, "turbo")
                        ? (
                            $is1080
                              ? {"type":"usd","usd": 0.175 + 0.05 * ($d - 1)}
                              : (
                                  $d <= 2 ? {"type":"usd","usd": 0.05}
                                  : {"type":"usd","usd": 0.05 + 0.05 * ($d - 2)}
                                )
                          )
                        : {"type":"usd","usd": 0.04}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        first_frame: Input.Image,
        end_frame: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        validate_string(prompt, max_length=2000)
        if get_number_of_images(first_frame) > 1:
            raise ValueError("Only one input image is allowed for `first_frame`.")
        if get_number_of_images(end_frame) > 1:
            raise ValueError("Only one input image is allowed for `end_frame`.")
        validate_images_aspect_ratio_closeness(first_frame, end_frame, min_rel=0.8, max_rel=1.25, strict=False)
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
            images=[
                (await upload_images_to_comfyapi(cls, frame, max_images=1, mime_type="image/png"))[0]
                for frame in (first_frame, end_frame)
            ],
        )
        results = await execute_task(cls, VIDU_START_END_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ViduTextToVideoNode,
            ViduImageToVideoNode,
            ViduReferenceVideoNode,
            ViduStartEndToVideoNode,
            Vidu2TextToVideoNode,
            Vidu2ImageToVideoNode,
            Vidu2ReferenceVideoNode,
            Vidu2StartEndToVideoNode,
        ]


async def comfy_entrypoint() -> ViduExtension:
    return ViduExtension()
