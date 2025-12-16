import builtins
from io import BytesIO

import aiohttp
import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis import topaz_api
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_fs_object_size,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_container_format_is_mp4,
)

UPSCALER_MODELS_MAP = {
    "Starlight (Astra) Fast": "slf-1",
    "Starlight (Astra) Creative": "slc-1",
}
UPSCALER_VALUES_MAP = {
    "FullHD (1080p)": 1920,
    "4K (2160p)": 3840,
}


class TopazImageEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TopazImageEnhance",
            display_name="Topaz Image Enhance",
            category="api node/image/Topaz",
            description="Industry-standard upscaling and image enhancement.",
            inputs=[
                IO.Combo.Input("model", options=["Reimagine"]),
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Optional text prompt for creative upscaling guidance.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "subject_detection",
                    options=["All", "Foreground", "Background"],
                    optional=True,
                ),
                IO.Boolean.Input(
                    "face_enhancement",
                    default=True,
                    optional=True,
                    tooltip="Enhance faces (if present) during processing.",
                ),
                IO.Float.Input(
                    "face_enhancement_creativity",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                    tooltip="Set the creativity level for face enhancement.",
                ),
                IO.Float.Input(
                    "face_enhancement_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                    tooltip="Controls how sharp enhanced faces are relative to the background.",
                ),
                IO.Boolean.Input(
                    "crop_to_fill",
                    default=False,
                    optional=True,
                    tooltip="By default, the image is letterboxed when the output aspect ratio differs. "
                    "Enable to crop the image to fill the output dimensions.",
                ),
                IO.Int.Input(
                    "output_width",
                    default=0,
                    min=0,
                    max=32000,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                    tooltip="Zero value means to calculate automatically (usually it will be original size or output_height if specified).",
                ),
                IO.Int.Input(
                    "output_height",
                    default=0,
                    min=0,
                    max=32000,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                    tooltip="Zero value means to output in the same height as original or output width.",
                ),
                IO.Int.Input(
                    "creativity",
                    default=3,
                    min=1,
                    max=9,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                    optional=True,
                ),
                IO.Boolean.Input(
                    "face_preservation",
                    default=True,
                    optional=True,
                    tooltip="Preserve subjects' facial identity.",
                ),
                IO.Boolean.Input(
                    "color_preservation",
                    default=True,
                    optional=True,
                    tooltip="Preserve the original colors.",
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
        model: str,
        image: torch.Tensor,
        prompt: str = "",
        subject_detection: str = "All",
        face_enhancement: bool = True,
        face_enhancement_creativity: float = 1.0,
        face_enhancement_strength: float = 0.8,
        crop_to_fill: bool = False,
        output_width: int = 0,
        output_height: int = 0,
        creativity: int = 3,
        face_preservation: bool = True,
        color_preservation: bool = True,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Only one input image is supported.")
        download_url = await upload_images_to_comfyapi(cls, image, max_images=1, mime_type="image/png")
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/topaz/image/v1/enhance-gen/async", method="POST"),
            response_model=topaz_api.ImageAsyncTaskResponse,
            data=topaz_api.ImageEnhanceRequest(
                model=model,
                prompt=prompt,
                subject_detection=subject_detection,
                face_enhancement=face_enhancement,
                face_enhancement_creativity=face_enhancement_creativity,
                face_enhancement_strength=face_enhancement_strength,
                crop_to_fill=crop_to_fill,
                output_width=output_width if output_width else None,
                output_height=output_height if output_height else None,
                creativity=creativity,
                face_preservation=str(face_preservation).lower(),
                color_preservation=str(color_preservation).lower(),
                source_url=download_url[0],
                output_format="png",
            ),
            content_type="multipart/form-data",
        )

        await poll_op(
            cls,
            poll_endpoint=ApiEndpoint(path=f"/proxy/topaz/image/v1/status/{initial_response.process_id}"),
            response_model=topaz_api.ImageStatusResponse,
            status_extractor=lambda x: x.status,
            progress_extractor=lambda x: getattr(x, "progress", 0),
            price_extractor=lambda x: x.credits * 0.08,
            poll_interval=8.0,
            max_poll_attempts=160,
            estimated_duration=60,
        )

        results = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/topaz/image/v1/download/{initial_response.process_id}"),
            response_model=topaz_api.ImageDownloadResponse,
            monitor_progress=False,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(results.download_url))


class TopazVideoEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TopazVideoEnhance",
            display_name="Topaz Video Enhance",
            category="api node/video/Topaz",
            description="Breathe new life into video with powerful upscaling and recovery technology.",
            inputs=[
                IO.Video.Input("video"),
                IO.Boolean.Input("upscaler_enabled", default=True),
                IO.Combo.Input("upscaler_model", options=list(UPSCALER_MODELS_MAP.keys())),
                IO.Combo.Input("upscaler_resolution", options=list(UPSCALER_VALUES_MAP.keys())),
                IO.Combo.Input(
                    "upscaler_creativity",
                    options=["low", "middle", "high"],
                    default="low",
                    tooltip="Creativity level (applies only to Starlight (Astra) Creative).",
                    optional=True,
                ),
                IO.Boolean.Input("interpolation_enabled", default=False, optional=True),
                IO.Combo.Input("interpolation_model", options=["apo-8"], default="apo-8", optional=True),
                IO.Int.Input(
                    "interpolation_slowmo",
                    default=1,
                    min=1,
                    max=16,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Slow-motion factor applied to the input video. "
                    "For example, 2 makes the output twice as slow and doubles the duration.",
                    optional=True,
                ),
                IO.Int.Input(
                    "interpolation_frame_rate",
                    default=60,
                    min=15,
                    max=240,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Output frame rate.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "interpolation_duplicate",
                    default=False,
                    tooltip="Analyze the input for duplicate frames and remove them.",
                    optional=True,
                ),
                IO.Float.Input(
                    "interpolation_duplicate_threshold",
                    default=0.01,
                    min=0.001,
                    max=0.1,
                    step=0.001,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Detection sensitivity for duplicate frames.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "dynamic_compression_level",
                    options=["Low", "Mid", "High"],
                    default="Low",
                    tooltip="CQP level.",
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
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        upscaler_enabled: bool,
        upscaler_model: str,
        upscaler_resolution: str,
        upscaler_creativity: str = "low",
        interpolation_enabled: bool = False,
        interpolation_model: str = "apo-8",
        interpolation_slowmo: int = 1,
        interpolation_frame_rate: int = 60,
        interpolation_duplicate: bool = False,
        interpolation_duplicate_threshold: float = 0.01,
        dynamic_compression_level: str = "Low",
    ) -> IO.NodeOutput:
        if upscaler_enabled is False and interpolation_enabled is False:
            raise ValueError("There is nothing to do: both upscaling and interpolation are disabled.")
        validate_container_format_is_mp4(video)
        src_width, src_height = video.get_dimensions()
        src_frame_rate = int(video.get_frame_rate())
        duration_sec = video.get_duration()
        src_video_stream = video.get_stream_source()
        target_width = src_width
        target_height = src_height
        target_frame_rate = src_frame_rate
        filters = []
        if upscaler_enabled:
            target_width = UPSCALER_VALUES_MAP[upscaler_resolution]
            target_height = UPSCALER_VALUES_MAP[upscaler_resolution]
            filters.append(
                topaz_api.VideoEnhancementFilter(
                    model=UPSCALER_MODELS_MAP[upscaler_model],
                    creativity=(upscaler_creativity if UPSCALER_MODELS_MAP[upscaler_model] == "slc-1" else None),
                    isOptimizedMode=(True if UPSCALER_MODELS_MAP[upscaler_model] == "slc-1" else None),
                ),
            )
        if interpolation_enabled:
            target_frame_rate = interpolation_frame_rate
            filters.append(
                topaz_api.VideoFrameInterpolationFilter(
                    model=interpolation_model,
                    slowmo=interpolation_slowmo,
                    fps=interpolation_frame_rate,
                    duplicate=interpolation_duplicate,
                    duplicate_threshold=interpolation_duplicate_threshold,
                ),
            )
        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/topaz/video/", method="POST"),
            response_model=topaz_api.CreateVideoResponse,
            data=topaz_api.CreateVideoRequest(
                source=topaz_api.CreateCreateVideoRequestSource(
                    container="mp4",
                    size=get_fs_object_size(src_video_stream),
                    duration=int(duration_sec),
                    frameCount=video.get_frame_count(),
                    frameRate=src_frame_rate,
                    resolution=topaz_api.Resolution(width=src_width, height=src_height),
                ),
                filters=filters,
                output=topaz_api.OutputInformationVideo(
                    resolution=topaz_api.Resolution(width=target_width, height=target_height),
                    frameRate=target_frame_rate,
                    audioCodec="AAC",
                    audioTransfer="Copy",
                    dynamicCompressionLevel=dynamic_compression_level,
                ),
            ),
            wait_label="Creating task",
            final_label_on_success="Task created",
        )
        upload_res = await sync_op(
            cls,
            ApiEndpoint(
                path=f"/proxy/topaz/video/{initial_res.requestId}/accept",
                method="PATCH",
            ),
            response_model=topaz_api.VideoAcceptResponse,
            wait_label="Preparing upload",
            final_label_on_success="Upload started",
        )
        if len(upload_res.urls) > 1:
            raise NotImplementedError(
                "Large files are not currently supported. Please open an issue in the ComfyUI repository."
            )
        async with aiohttp.ClientSession(headers={"Content-Type": "video/mp4"}) as session:
            if isinstance(src_video_stream, BytesIO):
                src_video_stream.seek(0)
                async with session.put(upload_res.urls[0], data=src_video_stream, raise_for_status=True) as res:
                    upload_etag = res.headers["Etag"]
            else:
                with builtins.open(src_video_stream, "rb") as video_file:
                    async with session.put(upload_res.urls[0], data=video_file, raise_for_status=True) as res:
                        upload_etag = res.headers["Etag"]
        await sync_op(
            cls,
            ApiEndpoint(
                path=f"/proxy/topaz/video/{initial_res.requestId}/complete-upload",
                method="PATCH",
            ),
            response_model=topaz_api.VideoCompleteUploadResponse,
            data=topaz_api.VideoCompleteUploadRequest(
                uploadResults=[
                    topaz_api.VideoCompleteUploadRequestPart(
                        partNum=1,
                        eTag=upload_etag,
                    ),
                ],
            ),
            wait_label="Finalizing upload",
            final_label_on_success="Upload completed",
        )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/topaz/video/{initial_res.requestId}/status"),
            response_model=topaz_api.VideoStatusResponse,
            status_extractor=lambda x: x.status,
            progress_extractor=lambda x: getattr(x, "progress", 0),
            price_extractor=lambda x: (x.estimates.cost[0] * 0.08 if x.estimates and x.estimates.cost[0] else None),
            poll_interval=10.0,
            max_poll_attempts=320,
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.download.url))


class TopazExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TopazImageEnhance,
            TopazVideoEnhance,
        ]


async def comfy_entrypoint() -> TopazExtension:
    return TopazExtension()
