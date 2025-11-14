"""Kling API Nodes

For source of truth on the allowed permutations of request fields, please reference:
- [Compatibility Table](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
"""

from __future__ import annotations
from typing import Optional, TypeVar
import math
import logging

from typing_extensions import override

import torch

from comfy_api_nodes.apis import (
    KlingCameraControl,
    KlingCameraConfig,
    KlingCameraControlType,
    KlingVideoGenDuration,
    KlingVideoGenMode,
    KlingVideoGenAspectRatio,
    KlingVideoGenModelName,
    KlingText2VideoRequest,
    KlingText2VideoResponse,
    KlingImage2VideoRequest,
    KlingImage2VideoResponse,
    KlingVideoExtendRequest,
    KlingVideoExtendResponse,
    KlingLipSyncVoiceLanguage,
    KlingLipSyncInputObject,
    KlingLipSyncRequest,
    KlingLipSyncResponse,
    KlingVirtualTryOnModelName,
    KlingVirtualTryOnRequest,
    KlingVirtualTryOnResponse,
    KlingVideoResult,
    KlingImageResult,
    KlingImageGenerationsRequest,
    KlingImageGenerationsResponse,
    KlingImageGenImageReferenceType,
    KlingImageGenModelName,
    KlingImageGenAspectRatio,
    KlingVideoEffectsRequest,
    KlingVideoEffectsResponse,
    KlingDualCharacterEffectsScene,
    KlingSingleImageEffectsScene,
    KlingDualCharacterEffectInput,
    KlingSingleImageEffectInput,
    KlingCharacterEffectModelName,
    KlingSingleImageEffectModelName,
)
from comfy_api_nodes.util import (
    validate_image_dimensions,
    validate_image_aspect_ratio,
    validate_video_dimensions,
    validate_video_duration,
    tensor_to_base64_string,
    validate_string,
    upload_audio_to_comfyapi,
    download_url_to_image_tensor,
    upload_video_to_comfyapi,
    download_url_to_video_output,
    sync_op,
    ApiEndpoint,
    poll_op,
)
from comfy_api.input_impl import VideoFromFile
from comfy_api.input.basic_types import AudioInput
from comfy_api.input.video_types import VideoInput
from comfy_api.latest import ComfyExtension, IO

KLING_API_VERSION = "v1"
PATH_TEXT_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/text2video"
PATH_IMAGE_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/image2video"
PATH_VIDEO_EXTEND = f"/proxy/kling/{KLING_API_VERSION}/videos/video-extend"
PATH_LIP_SYNC = f"/proxy/kling/{KLING_API_VERSION}/videos/lip-sync"
PATH_VIDEO_EFFECTS = f"/proxy/kling/{KLING_API_VERSION}/videos/effects"
PATH_CHARACTER_IMAGE = f"/proxy/kling/{KLING_API_VERSION}/images/generations"
PATH_VIRTUAL_TRY_ON = f"/proxy/kling/{KLING_API_VERSION}/images/kolors-virtual-try-on"
PATH_IMAGE_GENERATIONS = f"/proxy/kling/{KLING_API_VERSION}/images/generations"

MAX_PROMPT_LENGTH_T2V = 2500
MAX_PROMPT_LENGTH_I2V = 500
MAX_PROMPT_LENGTH_IMAGE_GEN = 500
MAX_NEGATIVE_PROMPT_LENGTH_IMAGE_GEN = 200
MAX_PROMPT_LENGTH_LIP_SYNC = 120

AVERAGE_DURATION_T2V = 319
AVERAGE_DURATION_I2V = 164
AVERAGE_DURATION_LIP_SYNC = 455
AVERAGE_DURATION_VIRTUAL_TRY_ON = 19
AVERAGE_DURATION_IMAGE_GEN = 32
AVERAGE_DURATION_VIDEO_EFFECTS = 320
AVERAGE_DURATION_VIDEO_EXTEND = 320

R = TypeVar("R")


MODE_TEXT2VIDEO = {
    "standard mode / 5s duration / kling-v1": ("std", "5", "kling-v1"),
    "standard mode / 10s duration / kling-v1": ("std", "10", "kling-v1"),
    "pro mode / 5s duration / kling-v1": ("pro", "5", "kling-v1"),
    "pro mode / 10s duration / kling-v1": ("pro", "10", "kling-v1"),
    "standard mode / 5s duration / kling-v1-6": ("std", "5", "kling-v1-6"),
    "standard mode / 10s duration / kling-v1-6": ("std", "10", "kling-v1-6"),
    "pro mode / 5s duration / kling-v2-master": ("pro", "5", "kling-v2-master"),
    "pro mode / 10s duration / kling-v2-master": ("pro", "10", "kling-v2-master"),
    "standard mode / 5s duration / kling-v2-master": ("std", "5", "kling-v2-master"),
    "standard mode / 10s duration / kling-v2-master": ("std", "10", "kling-v2-master"),
    "pro mode / 5s duration / kling-v2-1-master": ("pro", "5", "kling-v2-1-master"),
    "pro mode / 10s duration / kling-v2-1-master": ("pro", "10", "kling-v2-1-master"),
    "pro mode / 5s duration / kling-v2-5-turbo": ("pro", "5", "kling-v2-5-turbo"),
    "pro mode / 10s duration / kling-v2-5-turbo": ("pro", "10", "kling-v2-5-turbo"),
}
"""
Mapping of mode strings to their corresponding (mode, duration, model_name) tuples.
Only includes config combos that support the `image_tail` request field.

See: [Kling API Docs Capability Map](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
"""


MODE_START_END_FRAME = {
    "standard mode / 5s duration / kling-v1": ("std", "5", "kling-v1"),
    "pro mode / 5s duration / kling-v1": ("pro", "5", "kling-v1"),
    "pro mode / 5s duration / kling-v1-5": ("pro", "5", "kling-v1-5"),
    "pro mode / 10s duration / kling-v1-5": ("pro", "10", "kling-v1-5"),
    "pro mode / 5s duration / kling-v1-6": ("pro", "5", "kling-v1-6"),
    "pro mode / 10s duration / kling-v1-6": ("pro", "10", "kling-v1-6"),
    "pro mode / 5s duration / kling-v2-1": ("pro", "5", "kling-v2-1"),
    "pro mode / 10s duration / kling-v2-1": ("pro", "10", "kling-v2-1"),
}
"""
Returns a mapping of mode strings to their corresponding (mode, duration, model_name) tuples.
Only includes config combos that support the `image_tail` request field.

See: [Kling API Docs Capability Map](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
"""


VOICES_CONFIG = {
    # English voices
    "Melody": ("girlfriend_4_speech02", "en"),
    "Sunny": ("genshin_vindi2", "en"),
    "Sage": ("zhinen_xuesheng", "en"),
    "Ace": ("AOT", "en"),
    "Blossom": ("ai_shatang", "en"),
    "Peppy": ("genshin_klee2", "en"),
    "Dove": ("genshin_kirara", "en"),
    "Shine": ("ai_kaiya", "en"),
    "Anchor": ("oversea_male1", "en"),
    "Lyric": ("ai_chenjiahao_712", "en"),
    "Tender": ("chat1_female_new-3", "en"),
    "Siren": ("chat_0407_5-1", "en"),
    "Zippy": ("cartoon-boy-07", "en"),
    "Bud": ("uk_boy1", "en"),
    "Sprite": ("cartoon-girl-01", "en"),
    "Candy": ("PeppaPig_platform", "en"),
    "Beacon": ("ai_huangzhong_712", "en"),
    "Rock": ("ai_huangyaoshi_712", "en"),
    "Titan": ("ai_laoguowang_712", "en"),
    "Grace": ("chengshu_jiejie", "en"),
    "Helen": ("you_pingjing", "en"),
    "Lore": ("calm_story1", "en"),
    "Crag": ("uk_man2", "en"),
    "Prattle": ("laopopo_speech02", "en"),
    "Hearth": ("heainainai_speech02", "en"),
    "The Reader": ("reader_en_m-v1", "en"),
    "Commercial Lady": ("commercial_lady_en_f-v1", "en"),
    # Chinese voices
    "阳光少年": ("genshin_vindi2", "zh"),
    "懂事小弟": ("zhinen_xuesheng", "zh"),
    "运动少年": ("tiyuxi_xuedi", "zh"),
    "青春少女": ("ai_shatang", "zh"),
    "温柔小妹": ("genshin_klee2", "zh"),
    "元气少女": ("genshin_kirara", "zh"),
    "阳光男生": ("ai_kaiya", "zh"),
    "幽默小哥": ("tiexin_nanyou", "zh"),
    "文艺小哥": ("ai_chenjiahao_712", "zh"),
    "甜美邻家": ("girlfriend_1_speech02", "zh"),
    "温柔姐姐": ("chat1_female_new-3", "zh"),
    "职场女青": ("girlfriend_2_speech02", "zh"),
    "活泼男童": ("cartoon-boy-07", "zh"),
    "俏皮女童": ("cartoon-girl-01", "zh"),
    "稳重老爸": ("ai_huangyaoshi_712", "zh"),
    "温柔妈妈": ("you_pingjing", "zh"),
    "严肃上司": ("ai_laoguowang_712", "zh"),
    "优雅贵妇": ("chengshu_jiejie", "zh"),
    "慈祥爷爷": ("zhuxi_speech02", "zh"),
    "唠叨爷爷": ("uk_oldman3", "zh"),
    "唠叨奶奶": ("laopopo_speech02", "zh"),
    "和蔼奶奶": ("heainainai_speech02", "zh"),
    "东北老铁": ("dongbeilaotie_speech02", "zh"),
    "重庆小伙": ("chongqingxiaohuo_speech02", "zh"),
    "四川妹子": ("chuanmeizi_speech02", "zh"),
    "潮汕大叔": ("chaoshandashu_speech02", "zh"),
    "台湾男生": ("ai_taiwan_man2_speech02", "zh"),
    "西安掌柜": ("xianzhanggui_speech02", "zh"),
    "天津姐姐": ("tianjinjiejie_speech02", "zh"),
    "新闻播报男": ("diyinnansang_DB_CN_M_04-v2", "zh"),
    "译制片男": ("yizhipiannan-v1", "zh"),
    "撒娇女友": ("tianmeixuemei-v1", "zh"),
    "刀片烟嗓": ("daopianyansang-v1", "zh"),
    "乖巧正太": ("mengwa-v1", "zh"),
}


def is_valid_camera_control_configs(configs: list[float]) -> bool:
    """Verifies that at least one camera control configuration is non-zero."""
    return any(not math.isclose(value, 0.0) for value in configs)


def is_valid_task_creation_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the initial response contains a task ID."""
    return bool(response.data.task_id)


def is_valid_video_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the response contains a task result with at least one video."""
    return (
        response.data is not None
        and response.data.task_result is not None
        and response.data.task_result.videos is not None
        and len(response.data.task_result.videos) > 0
    )


def is_valid_image_response(response: KlingVirtualTryOnResponse) -> bool:
    """Verifies that the response contains a task result with at least one image."""
    return (
        response.data is not None
        and response.data.task_result is not None
        and response.data.task_result.images is not None
        and len(response.data.task_result.images) > 0
    )


def validate_prompts(prompt: str, negative_prompt: str, max_length: int) -> bool:
    """Verifies that the positive prompt is not empty and that neither promt is too long."""
    if not prompt:
        raise ValueError("Positive prompt is empty")
    if len(prompt) > max_length:
        raise ValueError(f"Positive prompt is too long: {len(prompt)} characters")
    if negative_prompt and len(negative_prompt) > max_length:
        raise ValueError(
            f"Negative prompt is too long: {len(negative_prompt)} characters"
        )
    return True


def validate_task_creation_response(response) -> None:
    """Validates that the Kling task creation request was successful."""
    if not is_valid_task_creation_response(response):
        error_msg = f"Kling initial request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
        logging.error(error_msg)
        raise Exception(error_msg)


def validate_video_result_response(response) -> None:
    """Validates that the Kling task result contains a video."""
    if not is_valid_video_response(response):
        error_msg = f"Kling task {response.data.task_id} succeeded but no video data found in response."
        logging.error("Error: %s.\nResponse: %s", error_msg, response)
        raise Exception(error_msg)


def validate_image_result_response(response) -> None:
    """Validates that the Kling task result contains an image."""
    if not is_valid_image_response(response):
        error_msg = f"Kling task {response.data.task_id} succeeded but no image data found in response."
        logging.error("Error: %s.\nResponse: %s", error_msg, response)
        raise Exception(error_msg)


def validate_input_image(image: torch.Tensor) -> None:
    """
    Validates the input image adheres to the expectations of the Kling API:
    - The image resolution should not be less than 300*300px
    - The aspect ratio of the image should be between 1:2.5 ~ 2.5:1

    See: https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo
    """
    validate_image_dimensions(image, min_width=300, min_height=300)
    validate_image_aspect_ratio(image, (1, 2.5), (2.5, 1))


def get_video_from_response(response) -> KlingVideoResult:
    """Returns the first video object from the Kling video generation task result.
    Will raise an error if the response is not valid.
    """
    video = response.data.task_result.videos[0]
    logging.info(
        "Kling task %s succeeded. Video URL: %s", response.data.task_id, video.url
    )
    return video


def get_video_url_from_response(response) -> Optional[str]:
    """Returns the first video url from the Kling video generation task result.
    Will not raise an error if the response is not valid.
    """
    if response and is_valid_video_response(response):
        return str(get_video_from_response(response).url)
    else:
        return None


def get_images_from_response(response) -> list[KlingImageResult]:
    """Returns the list of image objects from the Kling image generation task result.
    Will raise an error if the response is not valid.
    """
    images = response.data.task_result.images
    logging.info("Kling task %s succeeded. Images: %s", response.data.task_id, images)
    return images


def get_images_urls_from_response(response) -> Optional[str]:
    """Returns the list of image urls from the Kling image generation task result.
    Will not raise an error if the response is not valid. If there is only one image, returns the url as a string. If there are multiple images, returns a list of urls.
    """
    if response and is_valid_image_response(response):
        images = get_images_from_response(response)
        image_urls = [str(image.url) for image in images]
        return "\n".join(image_urls)
    else:
        return None


async def image_result_to_node_output(
    images: list[KlingImageResult],
) -> torch.Tensor:
    """
    Converts a KlingImageResult to a tuple containing a [B, H, W, C] tensor.
    If multiple images are returned, they will be stacked along the batch dimension.
    """
    if len(images) == 1:
        return await download_url_to_image_tensor(str(images[0].url))
    else:
        return torch.cat([await download_url_to_image_tensor(str(image.url)) for image in images])


async def execute_text2video(
    cls: type[IO.ComfyNode],
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    model_name: str,
    model_mode: str,
    duration: str,
    aspect_ratio: str,
    camera_control: Optional[KlingCameraControl] = None,
) -> IO.NodeOutput:
    validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_T2V)
    task_creation_response = await sync_op(
        cls,
        ApiEndpoint(path=PATH_TEXT_TO_VIDEO, method="POST"),
        response_model=KlingText2VideoResponse,
        data=KlingText2VideoRequest(
            prompt=prompt if prompt else None,
            negative_prompt=negative_prompt if negative_prompt else None,
            duration=KlingVideoGenDuration(duration),
            mode=KlingVideoGenMode(model_mode),
            model_name=KlingVideoGenModelName(model_name),
            cfg_scale=cfg_scale,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            camera_control=camera_control,
        ),
    )

    validate_task_creation_response(task_creation_response)

    task_id = task_creation_response.data.task_id
    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_TEXT_TO_VIDEO}/{task_id}"),
        response_model=KlingText2VideoResponse,
        estimated_duration=AVERAGE_DURATION_T2V,
        status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
    )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


async def execute_image2video(
    cls: type[IO.ComfyNode],
    start_frame: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    model_name: str,
    cfg_scale: float,
    model_mode: str,
    aspect_ratio: str,
    duration: str,
    camera_control: Optional[KlingCameraControl] = None,
    end_frame: Optional[torch.Tensor] = None,
) -> IO.NodeOutput:
    validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_I2V)
    validate_input_image(start_frame)

    if camera_control is not None:
        # Camera control type for image 2 video is always `simple`
        camera_control.type = KlingCameraControlType.simple

    if model_mode == "std" and model_name == KlingVideoGenModelName.kling_v2_5_turbo.value:
        model_mode = "pro"  # October 5: currently "std" mode is not supported for this model

    task_creation_response = await sync_op(
        cls,
        ApiEndpoint(path=PATH_IMAGE_TO_VIDEO, method="POST"),
        response_model=KlingImage2VideoResponse,
        data=KlingImage2VideoRequest(
            model_name=KlingVideoGenModelName(model_name),
            image=tensor_to_base64_string(start_frame),
            image_tail=(
                tensor_to_base64_string(end_frame)
                if end_frame is not None
                else None
            ),
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            cfg_scale=cfg_scale,
            mode=KlingVideoGenMode(model_mode),
            duration=KlingVideoGenDuration(duration),
            camera_control=camera_control,
        ),
    )

    validate_task_creation_response(task_creation_response)
    task_id = task_creation_response.data.task_id

    final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{PATH_IMAGE_TO_VIDEO}/{task_id}"),
            response_model=KlingImage2VideoResponse,
            estimated_duration=AVERAGE_DURATION_I2V,
            status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
        )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


async def execute_video_effect(
    cls: type[IO.ComfyNode],
    dual_character: bool,
    effect_scene: KlingDualCharacterEffectsScene | KlingSingleImageEffectsScene,
    model_name: str,
    duration: KlingVideoGenDuration,
    image_1: torch.Tensor,
    image_2: Optional[torch.Tensor] = None,
    model_mode: Optional[KlingVideoGenMode] = None,
) -> tuple[VideoFromFile, str, str]:
    if dual_character:
        request_input_field = KlingDualCharacterEffectInput(
            model_name=model_name,
            mode=model_mode,
            images=[
                tensor_to_base64_string(image_1),
                tensor_to_base64_string(image_2),
            ],
            duration=duration,
        )
    else:
        request_input_field = KlingSingleImageEffectInput(
            model_name=model_name,
            image=tensor_to_base64_string(image_1),
            duration=duration,
        )

    task_creation_response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=PATH_VIDEO_EFFECTS, method="POST"),
        response_model=KlingVideoEffectsResponse,
        data=KlingVideoEffectsRequest(
            effect_scene=effect_scene,
            input=request_input_field,
        ),
    )

    validate_task_creation_response(task_creation_response)
    task_id = task_creation_response.data.task_id

    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_VIDEO_EFFECTS}/{task_id}"),
        response_model=KlingVideoEffectsResponse,
        estimated_duration=AVERAGE_DURATION_VIDEO_EFFECTS,
        status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
    )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration)


async def execute_lipsync(
    cls: type[IO.ComfyNode],
    video: VideoInput,
    audio: Optional[AudioInput] = None,
    voice_language: Optional[str] = None,
    model_mode: Optional[str] = None,
    text: Optional[str] = None,
    voice_speed: Optional[float] = None,
    voice_id: Optional[str] = None,
) -> IO.NodeOutput:
    if text:
        validate_string(text, field_name="Text", max_length=MAX_PROMPT_LENGTH_LIP_SYNC)
    validate_video_dimensions(video, 720, 1920)
    validate_video_duration(video, 2, 10)

    # Upload video to Comfy API and get download URL
    video_url = await upload_video_to_comfyapi(cls, video)
    logging.info("Uploaded video to Comfy API. URL: %s", video_url)

    # Upload the audio file to Comfy API and get download URL
    if audio:
        audio_url = await upload_audio_to_comfyapi(cls, audio)
        logging.info("Uploaded audio to Comfy API. URL: %s", audio_url)
    else:
        audio_url = None

    task_creation_response = await sync_op(
        cls,
        ApiEndpoint(PATH_LIP_SYNC, "POST"),
        response_model=KlingLipSyncResponse,
        data=KlingLipSyncRequest(
            input=KlingLipSyncInputObject(
                video_url=video_url,
                mode=model_mode,
                text=text,
                voice_language=voice_language,
                voice_speed=voice_speed,
                audio_type="url",
                audio_url=audio_url,
                voice_id=voice_id,
            ),
        ),
    )

    validate_task_creation_response(task_creation_response)
    task_id = task_creation_response.data.task_id

    final_response = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_LIP_SYNC}/{task_id}"),
        response_model=KlingLipSyncResponse,
        estimated_duration=AVERAGE_DURATION_LIP_SYNC,
        status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
    )
    validate_video_result_response(final_response)

    video = get_video_from_response(final_response)
    return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


class KlingCameraControls(IO.ComfyNode):
    """Kling Camera Controls Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingCameraControls",
            display_name="Kling Camera Controls",
            category="api node/video/Kling",
            description="Allows specifying configuration options for Kling Camera Controls and motion control effects.",
            inputs=[
                IO.Combo.Input("camera_control_type", options=KlingCameraControlType),
                IO.Float.Input(
                    "horizontal_movement",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's movement along horizontal axis (x-axis). Negative indicates left, positive indicates right",
                ),
                IO.Float.Input(
                    "vertical_movement",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's movement along vertical axis (y-axis). Negative indicates downward, positive indicates upward.",
                ),
                IO.Float.Input(
                    "pan",
                    default=0.5,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's rotation in vertical plane (x-axis). Negative indicates downward rotation, positive indicates upward rotation.",
                ),
                IO.Float.Input(
                    "tilt",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's rotation in horizontal plane (y-axis). Negative indicates left rotation, positive indicates right rotation.",
                ),
                IO.Float.Input(
                    "roll",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls camera's rolling amount (z-axis). Negative indicates counterclockwise, positive indicates clockwise.",
                ),
                IO.Float.Input(
                    "zoom",
                    default=0.0,
                    min=-10.0,
                    max=10.0,
                    step=0.25,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Controls change in camera's focal length. Negative indicates narrower field of view, positive indicates wider field of view.",
                ),
            ],
            outputs=[IO.Custom("CAMERA_CONTROL").Output(display_name="camera_control")],
        )

    @classmethod
    def validate_inputs(
        cls,
        horizontal_movement: float,
        vertical_movement: float,
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
    ) -> bool | str:
        if not is_valid_camera_control_configs(
            [
                horizontal_movement,
                vertical_movement,
                pan,
                tilt,
                roll,
                zoom,
            ]
        ):
            return "Invalid camera control configs: at least one of the values must be non-zero"
        return True

    @classmethod
    def execute(
        cls,
        camera_control_type: str,
        horizontal_movement: float,
        vertical_movement: float,
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
    ) -> IO.NodeOutput:
        return IO.NodeOutput(
            KlingCameraControl(
                type=KlingCameraControlType(camera_control_type),
                config=KlingCameraConfig(
                    horizontal=horizontal_movement,
                    vertical=vertical_movement,
                    pan=pan,
                    roll=roll,
                    tilt=tilt,
                    zoom=zoom,
                ),
            )
        )


class KlingTextToVideoNode(IO.ComfyNode):
    """Kling Text to Video Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        modes = list(MODE_TEXT2VIDEO.keys())
        return IO.Schema(
            node_id="KlingTextToVideoNode",
            display_name="Kling Text to Video",
            category="api node/video/Kling",
            description="Kling Text to Video Node",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=1.0, min=0.0, max=1.0),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default="16:9",
                ),
                IO.Combo.Input(
                    "mode",
                    options=modes,
                    default=modes[4],
                    tooltip="The configuration to use for the video generation following the format: mode / duration / model_name.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        negative_prompt: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
    ) -> IO.NodeOutput:
        model_mode, duration, model_name = MODE_TEXT2VIDEO[mode]
        return await execute_text2video(
            cls,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            model_mode=model_mode,
            aspect_ratio=aspect_ratio,
            model_name=model_name,
            duration=duration,
        )


class KlingCameraControlT2VNode(IO.ComfyNode):
    """
    Kling Text to Video Camera Control Node. This node is a text to video node, but it supports controlling the camera.
    Duration, mode, and model_name request fields are hard-coded because camera control is only supported in pro mode with the kling-v1-5 model at 5s duration as of 2025-05-02.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingCameraControlT2VNode",
            display_name="Kling Text to Video (Camera Control)",
            category="api node/video/Kling",
            description="Transform text into cinematic videos with professional camera movements that simulate real-world cinematography. Control virtual camera actions including zoom, rotation, pan, tilt, and first-person view, while maintaining focus on your original text.",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=0.75, min=0.0, max=1.0),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default="16:9",
                ),
                IO.Custom("CAMERA_CONTROL").Input(
                    "camera_control",
                    tooltip="Can be created using the Kling Camera Controls node. Controls the camera movement and motion during the video generation.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        camera_control: Optional[KlingCameraControl] = None,
    ) -> IO.NodeOutput:
        return await execute_text2video(
            cls,
            model_name=KlingVideoGenModelName.kling_v1,
            cfg_scale=cfg_scale,
            model_mode=KlingVideoGenMode.std,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            duration=KlingVideoGenDuration.field_5,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_control=camera_control,
        )


class KlingImage2VideoNode(IO.ComfyNode):
    """Kling Image to Video Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingImage2VideoNode",
            display_name="Kling Image to Video",
            category="api node/video/Kling",
            description="Kling Image to Video Node",
            inputs=[
                IO.Image.Input("start_frame", tooltip="The reference image used to generate the video."),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Combo.Input(
                    "model_name",
                    options=KlingVideoGenModelName,
                    default="kling-v2-master",
                ),
                IO.Float.Input("cfg_scale", default=0.8, min=0.0, max=1.0),
                IO.Combo.Input("mode", options=KlingVideoGenMode, default=KlingVideoGenMode.std),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default=KlingVideoGenAspectRatio.field_16_9,
                ),
                IO.Combo.Input("duration", options=KlingVideoGenDuration, default=KlingVideoGenDuration.field_5),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        start_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        model_name: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
        duration: str,
        camera_control: Optional[KlingCameraControl] = None,
        end_frame: Optional[torch.Tensor] = None,
    ) -> IO.NodeOutput:
        return await execute_image2video(
            cls,
            start_frame=start_frame,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            model_mode=mode,
            duration=duration,
            camera_control=camera_control,
            end_frame=end_frame,
        )


class KlingCameraControlI2VNode(IO.ComfyNode):
    """
    Kling Image to Video Camera Control Node. This node is a image to video node, but it supports controlling the camera.
    Duration, mode, and model_name request fields are hard-coded because camera control is only supported in pro mode with the kling-v1-5 model at 5s duration as of 2025-05-02.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingCameraControlI2VNode",
            display_name="Kling Image to Video (Camera Control)",
            category="api node/video/Kling",
            description="Transform still images into cinematic videos with professional camera movements that simulate real-world cinematography. Control virtual camera actions including zoom, rotation, pan, tilt, and first-person view, while maintaining focus on your original image.",
            inputs=[
                IO.Image.Input(
                    "start_frame",
                    tooltip="Reference Image - URL or Base64 encoded string, cannot exceed 10MB, resolution not less than 300*300px, aspect ratio between 1:2.5 ~ 2.5:1. Base64 should not include data:image prefix.",
                ),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=0.75, min=0.0, max=1.0),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=KlingVideoGenAspectRatio,
                    default=KlingVideoGenAspectRatio.field_16_9,
                ),
                IO.Custom("CAMERA_CONTROL").Input(
                    "camera_control",
                    tooltip="Can be created using the Kling Camera Controls node. Controls the camera movement and motion during the video generation.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        start_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        camera_control: KlingCameraControl,
    ) -> IO.NodeOutput:
        return await execute_image2video(
            cls,
            model_name=KlingVideoGenModelName.kling_v1_5,
            start_frame=start_frame,
            cfg_scale=cfg_scale,
            model_mode=KlingVideoGenMode.pro,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            duration=KlingVideoGenDuration.field_5,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_control=camera_control,
        )


class KlingStartEndFrameNode(IO.ComfyNode):
    """
    Kling First Last Frame Node. This node allows creation of a video from a first and last frame. It calls the normal image to video endpoint, but only allows the subset of input options that support the `image_tail` request field.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        modes = list(MODE_START_END_FRAME.keys())
        return IO.Schema(
            node_id="KlingStartEndFrameNode",
            display_name="Kling Start-End Frame to Video",
            category="api node/video/Kling",
            description="Generate a video sequence that transitions between your provided start and end images. The node creates all frames in between, producing a smooth transformation from the first frame to the last.",
            inputs=[
                IO.Image.Input(
                    "start_frame",
                    tooltip="Reference Image - URL or Base64 encoded string, cannot exceed 10MB, resolution not less than 300*300px, aspect ratio between 1:2.5 ~ 2.5:1. Base64 should not include data:image prefix.",
                ),
                IO.Image.Input(
                    "end_frame",
                    tooltip="Reference Image - End frame control. URL or Base64 encoded string, cannot exceed 10MB, resolution not less than 300*300px. Base64 should not include data:image prefix.",
                ),
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Float.Input("cfg_scale", default=0.5, min=0.0, max=1.0),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=[i.value for i in KlingVideoGenAspectRatio],
                    default="16:9",
                ),
                IO.Combo.Input(
                    "mode",
                    options=modes,
                    default=modes[2],
                    tooltip="The configuration to use for the video generation following the format: mode / duration / model_name.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        mode: str,
    ) -> IO.NodeOutput:
        mode, duration, model_name = MODE_START_END_FRAME[mode]
        return await execute_image2video(
            cls,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            start_frame=start_frame,
            cfg_scale=cfg_scale,
            model_mode=mode,
            aspect_ratio=aspect_ratio,
            duration=duration,
            end_frame=end_frame,
        )


class KlingVideoExtendNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingVideoExtendNode",
            display_name="Kling Video Extend",
            category="api node/video/Kling",
            description="Kling Video Extend Node. Extend videos made by other Kling nodes. The video_id is created by using other Kling Nodes.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Positive text prompt for guiding the video extension",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    tooltip="Negative text prompt for elements to avoid in the extended video",
                ),
                IO.Float.Input("cfg_scale", default=0.5, min=0.0, max=1.0),
                IO.String.Input(
                    "video_id",
                    force_input=True,
                    tooltip="The ID of the video to be extended. Supports videos generated by text-to-video, image-to-video, and previous video extension operations. Cannot exceed 3 minutes total duration after extension.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        negative_prompt: str,
        cfg_scale: float,
        video_id: str,
    ) -> IO.NodeOutput:
        validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_T2V)
        task_creation_response = await sync_op(
            cls,
            ApiEndpoint(path=PATH_VIDEO_EXTEND, method="POST"),
            response_model=KlingVideoExtendResponse,
            data=KlingVideoExtendRequest(
                prompt=prompt if prompt else None,
                negative_prompt=negative_prompt if negative_prompt else None,
                cfg_scale=cfg_scale,
                video_id=video_id,
            ),
        )

        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{PATH_VIDEO_EXTEND}/{task_id}"),
            response_model=KlingVideoExtendResponse,
            estimated_duration=AVERAGE_DURATION_VIDEO_EXTEND,
            status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
        )
        validate_video_result_response(final_response)

        video = get_video_from_response(final_response)
        return IO.NodeOutput(await download_url_to_video_output(str(video.url)), str(video.id), str(video.duration))


class KlingDualCharacterVideoEffectNode(IO.ComfyNode):
    """Kling Dual Character Video Effect Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingDualCharacterVideoEffectNode",
            display_name="Kling Dual Character Video Effects",
            category="api node/video/Kling",
            description="Achieve different special effects when generating a video based on the effect_scene. First image will be positioned on left side, second on right side of the composite.",
            inputs=[
                IO.Image.Input("image_left", tooltip="Left side image"),
                IO.Image.Input("image_right", tooltip="Right side image"),
                IO.Combo.Input(
                    "effect_scene",
                    options=[i.value for i in KlingDualCharacterEffectsScene],
                ),
                IO.Combo.Input(
                    "model_name",
                    options=[i.value for i in KlingCharacterEffectModelName],
                    default="kling-v1",
                ),
                IO.Combo.Input(
                    "mode",
                    options=[i.value for i in KlingVideoGenMode],
                    default="std",
                ),
                IO.Combo.Input(
                    "duration",
                    options=[i.value for i in KlingVideoGenDuration],
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="duration"),
            ],
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
        image_left: torch.Tensor,
        image_right: torch.Tensor,
        effect_scene: KlingDualCharacterEffectsScene,
        model_name: KlingCharacterEffectModelName,
        mode: KlingVideoGenMode,
        duration: KlingVideoGenDuration,
    ) -> IO.NodeOutput:
        video, _, duration = await execute_video_effect(
            cls,
            dual_character=True,
            effect_scene=effect_scene,
            model_name=model_name,
            model_mode=mode,
            duration=duration,
            image_1=image_left,
            image_2=image_right,
        )
        return IO.NodeOutput(video, duration)


class KlingSingleImageVideoEffectNode(IO.ComfyNode):
    """Kling Single Image Video Effect Node"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingSingleImageVideoEffectNode",
            display_name="Kling Video Effects",
            category="api node/video/Kling",
            description="Achieve different special effects when generating a video based on the effect_scene.",
            inputs=[
                IO.Image.Input("image", tooltip=" Reference Image. URL or Base64 encoded string (without data:image prefix). File size cannot exceed 10MB, resolution not less than 300*300px, aspect ratio between 1:2.5 ~ 2.5:1"),
                IO.Combo.Input(
                    "effect_scene",
                    options=[i.value for i in KlingSingleImageEffectsScene],
                ),
                IO.Combo.Input(
                    "model_name",
                    options=[i.value for i in KlingSingleImageEffectModelName],
                ),
                IO.Combo.Input(
                    "duration",
                    options=[i.value for i in KlingVideoGenDuration],
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        effect_scene: KlingSingleImageEffectsScene,
        model_name: KlingSingleImageEffectModelName,
        duration: KlingVideoGenDuration,
    ) -> IO.NodeOutput:
        return IO.NodeOutput(
            *(
                await execute_video_effect(
                    cls,
                    dual_character=False,
                    effect_scene=effect_scene,
                    model_name=model_name,
                    duration=duration,
                    image_1=image,
                )
            )
        )


class KlingLipSyncAudioToVideoNode(IO.ComfyNode):
    """Kling Lip Sync Audio to Video Node. Syncs mouth movements in a video file to the audio content of an audio file."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingLipSyncAudioToVideoNode",
            display_name="Kling Lip Sync Video with Audio",
            category="api node/video/Kling",
            description="Kling Lip Sync Audio to Video Node. Syncs mouth movements in a video file to the audio content of an audio file. When using, ensure that the audio contains clearly distinguishable vocals and that the video contains a distinct face. The audio file should not be larger than 5MB. The video file should not be larger than 100MB, should have height/width between 720px and 1920px, and should be between 2s and 10s in length.",
            inputs=[
                IO.Video.Input("video"),
                IO.Audio.Input("audio"),
                IO.Combo.Input(
                    "voice_language",
                    options=[i.value for i in KlingLipSyncVoiceLanguage],
                    default="en",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        video: VideoInput,
        audio: AudioInput,
        voice_language: str,
    ) -> IO.NodeOutput:
        return await execute_lipsync(
            cls,
            video=video,
            audio=audio,
            voice_language=voice_language,
            model_mode="audio2video",
        )


class KlingLipSyncTextToVideoNode(IO.ComfyNode):
    """Kling Lip Sync Text to Video Node. Syncs mouth movements in a video file to a text prompt."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingLipSyncTextToVideoNode",
            display_name="Kling Lip Sync Video with Text",
            category="api node/video/Kling",
            description="Kling Lip Sync Text to Video Node. Syncs mouth movements in a video file to a text prompt. The video file should not be larger than 100MB, should have height/width between 720px and 1920px, and should be between 2s and 10s in length.",
            inputs=[
                IO.Video.Input("video"),
                IO.String.Input(
                    "text",
                    multiline=True,
                    tooltip="Text Content for Lip-Sync Video Generation. Required when mode is text2video. Maximum length is 120 characters.",
                ),
                IO.Combo.Input(
                    "voice",
                    options=list(VOICES_CONFIG.keys()),
                    default="Melody",
                ),
                IO.Float.Input(
                    "voice_speed",
                    default=1,
                    min=0.8,
                    max=2.0,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Speech Rate. Valid range: 0.8~2.0, accurate to one decimal place.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(display_name="video_id"),
                IO.String.Output(display_name="duration"),
            ],
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
        video: VideoInput,
        text: str,
        voice: str,
        voice_speed: float,
    ) -> IO.NodeOutput:
        voice_id, voice_language = VOICES_CONFIG[voice]
        return await execute_lipsync(
            cls,
            video=video,
            text=text,
            voice_language=voice_language,
            voice_id=voice_id,
            voice_speed=voice_speed,
            model_mode="text2video",
        )


class KlingVirtualTryOnNode(IO.ComfyNode):
    """Kling Virtual Try On Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingVirtualTryOnNode",
            display_name="Kling Virtual Try On",
            category="api node/image/Kling",
            description="Kling Virtual Try On Node. Input a human image and a cloth image to try on the cloth on the human. You can merge multiple clothing item pictures into one image with a white background.",
            inputs=[
                IO.Image.Input("human_image"),
                IO.Image.Input("cloth_image"),
                IO.Combo.Input(
                    "model_name",
                    options=[i.value for i in KlingVirtualTryOnModelName],
                    default="kolors-virtual-try-on-v1",
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
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
        human_image: torch.Tensor,
        cloth_image: torch.Tensor,
        model_name: KlingVirtualTryOnModelName,
    ) -> IO.NodeOutput:
        task_creation_response = await sync_op(
            cls,
            ApiEndpoint(path=PATH_VIRTUAL_TRY_ON, method="POST"),
            response_model=KlingVirtualTryOnResponse,
            data=KlingVirtualTryOnRequest(
                human_image=tensor_to_base64_string(human_image),
                cloth_image=tensor_to_base64_string(cloth_image),
                model_name=model_name,
            ),
        )

        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{PATH_VIRTUAL_TRY_ON}/{task_id}"),
            response_model=KlingVirtualTryOnResponse,
            estimated_duration=AVERAGE_DURATION_VIRTUAL_TRY_ON,
            status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
        )
        validate_image_result_response(final_response)

        images = get_images_from_response(final_response)
        return IO.NodeOutput(await image_result_to_node_output(images))


class KlingImageGenerationNode(IO.ComfyNode):
    """Kling Image Generation Node. Generate an image from a text prompt with an optional reference image."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="KlingImageGenerationNode",
            display_name="Kling Image Generation",
            category="api node/image/Kling",
            description="Kling Image Generation Node. Generate an image from a text prompt with an optional reference image.",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Positive text prompt"),
                IO.String.Input("negative_prompt", multiline=True, tooltip="Negative text prompt"),
                IO.Combo.Input(
                    "image_type",
                    options=[i.value for i in KlingImageGenImageReferenceType],
                ),
                IO.Float.Input(
                    "image_fidelity",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Reference intensity for user-uploaded images",
                ),
                IO.Float.Input(
                    "human_fidelity",
                    default=0.45,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Subject reference similarity",
                ),
                IO.Combo.Input(
                    "model_name",
                    options=[i.value for i in KlingImageGenModelName],
                    default="kling-v1",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=[i.value for i in KlingImageGenAspectRatio],
                    default="16:9",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=9,
                    tooltip="Number of generated images",
                ),
                IO.Image.Input("image", optional=True),
            ],
            outputs=[
                IO.Image.Output(),
            ],
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
        model_name: KlingImageGenModelName,
        prompt: str,
        negative_prompt: str,
        image_type: KlingImageGenImageReferenceType,
        image_fidelity: float,
        human_fidelity: float,
        n: int,
        aspect_ratio: KlingImageGenAspectRatio,
        image: Optional[torch.Tensor] = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, field_name="prompt", min_length=1, max_length=MAX_PROMPT_LENGTH_IMAGE_GEN)
        validate_string(negative_prompt, field_name="negative_prompt", max_length=MAX_PROMPT_LENGTH_IMAGE_GEN)

        if image is None:
            image_type = None
        elif model_name == KlingImageGenModelName.kling_v1:
            raise ValueError(f"The model {KlingImageGenModelName.kling_v1.value} does not support reference images.")
        else:
            image = tensor_to_base64_string(image)

        task_creation_response = await sync_op(
            cls,
            ApiEndpoint(path=PATH_IMAGE_GENERATIONS, method="POST"),
            response_model=KlingImageGenerationsResponse,
            data=KlingImageGenerationsRequest(
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                image_reference=image_type,
                image_fidelity=image_fidelity,
                human_fidelity=human_fidelity,
                n=n,
                aspect_ratio=aspect_ratio,
            ),
        )

        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{PATH_IMAGE_GENERATIONS}/{task_id}"),
            response_model=KlingImageGenerationsResponse,
            estimated_duration=AVERAGE_DURATION_IMAGE_GEN,
            status_extractor=lambda r: (r.data.task_status.value if r.data and r.data.task_status else None),
        )
        validate_image_result_response(final_response)

        images = get_images_from_response(final_response)
        return IO.NodeOutput(await image_result_to_node_output(images))


class KlingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            KlingCameraControls,
            KlingTextToVideoNode,
            KlingImage2VideoNode,
            KlingCameraControlI2VNode,
            KlingCameraControlT2VNode,
            KlingStartEndFrameNode,
            KlingVideoExtendNode,
            KlingLipSyncAudioToVideoNode,
            KlingLipSyncTextToVideoNode,
            KlingVirtualTryOnNode,
            KlingImageGenerationNode,
            KlingSingleImageVideoEffectNode,
            KlingDualCharacterVideoEffectNode,
        ]


async def comfy_entrypoint() -> KlingExtension:
    return KlingExtension()
