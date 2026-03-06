import logging
import math

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.bytedance import (
    RECOMMENDED_PRESETS,
    RECOMMENDED_PRESETS_SEEDREAM_4,
    VIDEO_TASKS_EXECUTION_TIME,
    Image2VideoTaskCreationRequest,
    ImageTaskCreationResponse,
    Seedream4Options,
    Seedream4TaskCreationRequest,
    TaskCreationResponse,
    TaskImageContent,
    TaskImageContentUrl,
    TaskStatusResponse,
    TaskTextContent,
    Text2ImageTaskCreationRequest,
    Text2VideoTaskCreationRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    image_tensor_pair_to_batch,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
)

BYTEPLUS_IMAGE_ENDPOINT = "/proxy/byteplus/api/v3/images/generations"

SEEDREAM_MODELS = {
    "seedream 5.0 lite": "seedream-5-0-260128",
    "seedream-4-5-251128": "seedream-4-5-251128",
    "seedream-4-0-250828": "seedream-4-0-250828",
}

# Long-running tasks endpoints(e.g., video)
BYTEPLUS_TASK_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"
BYTEPLUS_TASK_STATUS_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"  # + /{task_id}


def get_image_url_from_response(response: ImageTaskCreationResponse) -> str:
    if response.error:
        error_msg = f"ByteDance request failed. Code: {response.error['code']}, message: {response.error['message']}"
        logging.info(error_msg)
        raise RuntimeError(error_msg)
    logging.info("ByteDance task succeeded, image URL: %s", response.data[0]["url"])
    return response.data[0]["url"]


class ByteDanceImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageNode",
            display_name="ByteDance Image",
            category="api node/image/ByteDance",
            description="Generate images using ByteDance models via api based on prompt",
            inputs=[
                IO.Combo.Input("model", options=["seedream-3-0-t2i-250415"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below",
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "height",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=2.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the image',
                    optional=True,
                    advanced=True,
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
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.03}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        size_preset: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height
            if not (512 <= w <= 2048) or not (512 <= h <= 2048):
                raise ValueError(
                    f"Custom size out of range: {w}x{h}. " "Both width and height must be between 512 and 2048 pixels."
                )

        payload = Text2ImageTaskCreationRequest(
            model=model,
            prompt=prompt,
            size=f"{w}x{h}",
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            data=payload,
            response_model=ImageTaskCreationResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class ByteDanceSeedreamNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamNode",
            display_name="ByteDance Seedream 4.5 & 5.0",
            category="api node/image/ByteDance",
            description="Unified text-to-image generation and precise single-sentence editing at up to 4K resolution.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=list(SEEDREAM_MODELS.keys()),
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for creating or editing an image.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Input image(s) for image-to-image generation. "
                    "Reference image(s) for single or multi-reference generation.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS_SEEDREAM_4],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below.",
                ),
                IO.Int.Input(
                    "width",
                    default=2048,
                    min=1024,
                    max=6240,
                    step=2,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Int.Input(
                    "height",
                    default=2048,
                    min=1024,
                    max=4992,
                    step=2,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Combo.Input(
                    "sequential_image_generation",
                    options=["disabled", "auto"],
                    tooltip="Group image generation mode. "
                    "'disabled' generates a single image. "
                    "'auto' lets the model decide whether to generate multiple related images "
                    "(e.g., story scenes, character variations).",
                    optional=True,
                ),
                IO.Int.Input(
                    "max_images",
                    default=1,
                    min=1,
                    max=15,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Maximum number of images to generate when sequential_image_generation='auto'. "
                    "Total images (input + generated) cannot exceed 15.",
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
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the image.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "fail_on_partial",
                    default=True,
                    tooltip="If enabled, abort execution if any requested images are missing or return an error.",
                    optional=True,
                    advanced=True,
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $price := $contains(widgets.model, "5.0 lite") ? 0.035 :
                            $contains(widgets.model, "4-5") ? 0.04 : 0.03;
                  {
                    "type":"usd",
                    "usd": $price,
                    "format": { "suffix":" x images/Run", "approximate": true }
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        image: Input.Image | None = None,
        size_preset: str = RECOMMENDED_PRESETS_SEEDREAM_4[0][0],
        width: int = 2048,
        height: int = 2048,
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        seed: int = 0,
        watermark: bool = False,
        fail_on_partial: bool = True,
    ) -> IO.NodeOutput:
        model = SEEDREAM_MODELS[model]
        validate_string(prompt, strip_whitespace=True, min_length=1)
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS_SEEDREAM_4:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height

        out_num_pixels = w * h
        mp_provided = out_num_pixels / 1_000_000.0
        if ("seedream-4-5" in model or "seedream-5-0" in model) and out_num_pixels < 3686400:
            raise ValueError(
                f"Minimum image resolution for the selected model is 3.68MP, "
                f"but {mp_provided:.2f}MP provided."
            )
        if "seedream-4-0" in model and out_num_pixels < 921600:
            raise ValueError(
                f"Minimum image resolution that the selected model can generate is 0.92MP, "
                f"but {mp_provided:.2f}MP provided."
            )
        max_pixels = 10_404_496 if "seedream-5-0" in model else 16_777_216
        if out_num_pixels > max_pixels:
            raise ValueError(
                f"Maximum image resolution for the selected model is {max_pixels / 1_000_000:.2f}MP, "
                f"but {mp_provided:.2f}MP provided."
            )
        n_input_images = get_number_of_images(image) if image is not None else 0
        max_num_of_images = 14 if model == "seedream-5-0-260128" else 10
        if n_input_images > max_num_of_images:
            raise ValueError(
                f"Maximum of {max_num_of_images} reference images are supported, but {n_input_images} received."
            )
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(
                "The maximum number of generated images plus the number of reference images cannot exceed 15."
            )
        reference_images_urls = []
        if n_input_images:
            for i in image:
                validate_image_aspect_ratio(i, (1, 3), (3, 1))
            reference_images_urls = await upload_images_to_comfyapi(
                cls,
                image,
                max_images=n_input_images,
                mime_type="image/png",
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            response_model=ImageTaskCreationResponse,
            data=Seedream4TaskCreationRequest(
                model=model,
                prompt=prompt,
                image=reference_images_urls,
                size=f"{w}x{h}",
                seed=seed,
                sequential_image_generation=sequential_image_generation,
                sequential_image_generation_options=Seedream4Options(max_images=max_images),
                watermark=watermark,
                output_format="png" if model == "seedream-5-0-260128" else None,
            ),
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))
        urls = [str(d["url"]) for d in response.data if isinstance(d, dict) and "url" in d]
        if fail_on_partial and len(urls) < len(response.data):
            raise RuntimeError(f"Only {len(urls)} of {len(response.data)} images were generated before error.")
        return IO.NodeOutput(torch.cat([await download_url_to_image_tensor(i) for i in urls]))


class ByteDanceTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceTextToVideoNode",
            display_name="ByteDance Text to Video",
            category="api node/video/ByteDance",
            description="Generate video using ByteDance models via api based on prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[
                        "seedance-1-5-pro-251215",
                        "seedance-1-0-pro-250528",
                        "seedance-1-0-lite-t2v-250428",
                        "seedance-1-0-pro-fast-251015",
                    ],
                    default="seedance-1-0-pro-fast-251015",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="This parameter is ignored for any model except seedance-1-5-pro.",
                    optional=True,
                    advanced=True,
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
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        if model == "seedance-1-5-pro-251215" and duration < 4:
            raise ValueError("Minimum supported duration for Seedance 1.5 Pro is 4 seconds.")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])

        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )
        return await process_video_task(
            cls,
            payload=Text2VideoTaskCreationRequest(
                model=model,
                content=[TaskTextContent(text=prompt)],
                generate_audio=generate_audio if model == "seedance-1-5-pro-251215" else None,
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageToVideoNode",
            display_name="ByteDance Image to Video",
            category="api node/video/ByteDance",
            description="Generate video using ByteDance models via api based on image and prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[
                        "seedance-1-5-pro-251215",
                        "seedance-1-0-pro-250528",
                        "seedance-1-0-lite-i2v-250428",
                        "seedance-1-0-pro-fast-251015",
                    ],
                    default="seedance-1-0-pro-fast-251015",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="This parameter is ignored for any model except seedance-1-5-pro.",
                    optional=True,
                    advanced=True,
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
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        image: Input.Image,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        if model == "seedance-1-5-pro-251215" and duration < 4:
            raise ValueError("Minimum supported duration for Seedance 1.5 Pro is 4 seconds.")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
        validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1))[0]
        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )

        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(
                model=model,
                content=[TaskTextContent(text=prompt), TaskImageContent(image_url=TaskImageContentUrl(url=image_url))],
                generate_audio=generate_audio if model == "seedance-1-5-pro-251215" else None,
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceFirstLastFrameNode",
            display_name="ByteDance First-Last-Frame to Video",
            category="api node/video/ByteDance",
            description="Generate video using prompt and first and last frames.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["seedance-1-5-pro-251215", "seedance-1-0-pro-250528", "seedance-1-0-lite-i2v-250428"],
                    default="seedance-1-0-lite-i2v-250428",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Last frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="This parameter is ignored for any model except seedance-1-5-pro.",
                    optional=True,
                    advanced=True,
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
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        first_frame: Input.Image,
        last_frame: Input.Image,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        if model == "seedance-1-5-pro-251215" and duration < 4:
            raise ValueError("Minimum supported duration for Seedance 1.5 Pro is 4 seconds.")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        for i in (first_frame, last_frame):
            validate_image_dimensions(i, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio(i, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        download_urls = await upload_images_to_comfyapi(
            cls,
            image_tensor_pair_to_batch(first_frame, last_frame),
            max_images=2,
            mime_type="image/png",
        )

        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )

        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(
                model=model,
                content=[
                    TaskTextContent(text=prompt),
                    TaskImageContent(image_url=TaskImageContentUrl(url=str(download_urls[0])), role="first_frame"),
                    TaskImageContent(image_url=TaskImageContentUrl(url=str(download_urls[1])), role="last_frame"),
                ],
                generate_audio=generate_audio if model == "seedance-1-5-pro-251215" else None,
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceImageReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageReferenceNode",
            display_name="ByteDance Reference Images to Video",
            category="api node/video/ByteDance",
            description="Generate video using prompt and reference images.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["seedance-1-0-pro-250528", "seedance-1-0-lite-i2v-250428"],
                    default="seedance-1-0-lite-i2v-250428",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="One to four images.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
                    optional=True,
                    advanced=True,
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
                  $priceByModel := {
                    "seedance-1-0-pro": {
                      "480p":[0.23,0.24],
                      "720p":[0.51,0.56]
                    },
                    "seedance-1-0-lite": {
                      "480p":[0.17,0.18],
                      "720p":[0.37,0.41]
                    }
                  };
                  $model := widgets.model;
                  $modelKey :=
                    $contains($model, "seedance-1-0-pro")  ? "seedance-1-0-pro" :
                    "seedance-1-0-lite";
                  $resolution := widgets.resolution;
                  $resKey :=
                    $contains($resolution, "720") ? "720p" :
                    "480p";
                  $modelPrices := $lookup($priceByModel, $modelKey);
                  $baseRange := $lookup($modelPrices, $resKey);
                  $min10s := $baseRange[0];
                  $max10s := $baseRange[1];
                  $scale := widgets.duration / 10;
                  $minCost := $min10s * $scale;
                  $maxCost := $max10s * $scale;
                  ($minCost = $maxCost)
                    ? {"type":"usd","usd": $minCost}
                    : {"type":"range_usd","min_usd": $minCost, "max_usd": $maxCost}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        images: Input.Image,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "watermark"])
        for image in images:
            validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        image_urls = await upload_images_to_comfyapi(cls, images, max_images=4, mime_type="image/png")
        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--watermark {str(watermark).lower()}"
        )
        x = [
            TaskTextContent(text=prompt),
            *[TaskImageContent(image_url=TaskImageContentUrl(url=str(i)), role="reference_image") for i in image_urls],
        ]
        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(model=model, content=x),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


async def process_video_task(
    cls: type[IO.ComfyNode],
    payload: Text2VideoTaskCreationRequest | Image2VideoTaskCreationRequest,
    estimated_duration: int | None,
) -> IO.NodeOutput:
    initial_response = await sync_op(
        cls,
        ApiEndpoint(path=BYTEPLUS_TASK_ENDPOINT, method="POST"),
        data=payload,
        response_model=TaskCreationResponse,
    )
    response = await poll_op(
        cls,
        ApiEndpoint(path=f"{BYTEPLUS_TASK_STATUS_ENDPOINT}/{initial_response.id}"),
        status_extractor=lambda r: r.status,
        estimated_duration=estimated_duration,
        response_model=TaskStatusResponse,
    )
    return IO.NodeOutput(await download_url_to_video_output(response.content.video_url))


def raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    for i in text_params:
        if f"--{i} " in prompt:
            raise ValueError(
                f"--{i} is not allowed in the prompt, use the appropriated widget input to change this value."
            )


PRICE_BADGE_VIDEO = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution", "generate_audio"]),
    expr="""
    (
      $priceByModel := {
        "seedance-1-5-pro": {
          "480p":[0.12,0.12],
          "720p":[0.26,0.26],
          "1080p":[0.58,0.59]
        },
        "seedance-1-0-pro": {
          "480p":[0.23,0.24],
          "720p":[0.51,0.56],
          "1080p":[1.18,1.22]
        },
        "seedance-1-0-pro-fast": {
          "480p":[0.09,0.1],
          "720p":[0.21,0.23],
          "1080p":[0.47,0.49]
        },
        "seedance-1-0-lite": {
          "480p":[0.17,0.18],
          "720p":[0.37,0.41],
          "1080p":[0.85,0.88]
        }
      };
      $model := widgets.model;
      $modelKey :=
        $contains($model, "seedance-1-5-pro")      ? "seedance-1-5-pro" :
        $contains($model, "seedance-1-0-pro-fast") ? "seedance-1-0-pro-fast" :
        $contains($model, "seedance-1-0-pro")      ? "seedance-1-0-pro" :
        "seedance-1-0-lite";
      $resolution := widgets.resolution;
      $resKey :=
        $contains($resolution, "1080") ? "1080p" :
        $contains($resolution, "720")  ? "720p" :
        "480p";
      $modelPrices := $lookup($priceByModel, $modelKey);
      $baseRange := $lookup($modelPrices, $resKey);
      $min10s := $baseRange[0];
      $max10s := $baseRange[1];
      $scale := widgets.duration / 10;
      $audioMultiplier := ($modelKey = "seedance-1-5-pro" and widgets.generate_audio) ? 2 : 1;
      $minCost := $min10s * $scale * $audioMultiplier;
      $maxCost := $max10s * $scale * $audioMultiplier;
      ($minCost = $maxCost)
        ? {"type":"usd","usd": $minCost, "format": { "approximate": true }}
        : {"type":"range_usd","min_usd": $minCost, "max_usd": $maxCost, "format": { "approximate": true }}
    )
    """,
)


class ByteDanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ByteDanceImageNode,
            ByteDanceSeedreamNode,
            ByteDanceTextToVideoNode,
            ByteDanceImageToVideoNode,
            ByteDanceFirstLastFrameNode,
            ByteDanceImageReferenceNode,
        ]


async def comfy_entrypoint() -> ByteDanceExtension:
    return ByteDanceExtension()
