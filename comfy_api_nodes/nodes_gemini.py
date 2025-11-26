"""
API Nodes for Gemini Multimodal LLM Usage via Remote API
See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
"""

import base64
import os
from enum import Enum
from io import BytesIO
from typing import Literal

import torch
from typing_extensions import override

import folder_paths
from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api.util import VideoCodec, VideoContainer
from comfy_api_nodes.apis.gemini_api import (
    GeminiContent,
    GeminiFileData,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiImageConfig,
    GeminiImageGenerateContentRequest,
    GeminiImageGenerationConfig,
    GeminiInlineData,
    GeminiMimeType,
    GeminiPart,
    GeminiRole,
    Modality,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_to_base64_string,
    bytesio_to_image_tensor,
    get_number_of_images,
    sync_op,
    tensor_to_base64_string,
    upload_images_to_comfyapi,
    validate_string,
    video_to_base64_string,
)

GEMINI_BASE_ENDPOINT = "/proxy/vertexai/gemini"
GEMINI_MAX_INPUT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class GeminiModel(str, Enum):
    """
    Gemini Model Names allowed by comfy-api
    """

    gemini_2_5_pro_preview_05_06 = "gemini-2.5-pro-preview-05-06"
    gemini_2_5_flash_preview_04_17 = "gemini-2.5-flash-preview-04-17"
    gemini_2_5_pro = "gemini-2.5-pro"
    gemini_2_5_flash = "gemini-2.5-flash"
    gemini_3_0_pro = "gemini-3-pro-preview"


class GeminiImageModel(str, Enum):
    """
    Gemini Image Model Names allowed by comfy-api
    """

    gemini_2_5_flash_image_preview = "gemini-2.5-flash-image-preview"
    gemini_2_5_flash_image = "gemini-2.5-flash-image"


async def create_image_parts(
    cls: type[IO.ComfyNode],
    images: torch.Tensor,
    image_limit: int = 0,
) -> list[GeminiPart]:
    image_parts: list[GeminiPart] = []
    if image_limit < 0:
        raise ValueError("image_limit must be greater than or equal to 0 when creating Gemini image parts.")
    total_images = get_number_of_images(images)
    if total_images <= 0:
        raise ValueError("No images provided to create_image_parts; at least one image is required.")

    # If image_limit == 0 --> use all images; otherwise clamp to image_limit.
    effective_max = total_images if image_limit == 0 else min(total_images, image_limit)

    # Number of images we'll send as URLs (fileData)
    num_url_images = min(effective_max, 10)  # Vertex API max number of image links
    reference_images_urls = await upload_images_to_comfyapi(
        cls,
        images,
        max_images=num_url_images,
    )
    for reference_image_url in reference_images_urls:
        image_parts.append(
            GeminiPart(
                fileData=GeminiFileData(
                    mimeType=GeminiMimeType.image_png,
                    fileUri=reference_image_url,
                )
            )
        )
    for idx in range(num_url_images, effective_max):
        image_parts.append(
            GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.image_png,
                    data=tensor_to_base64_string(images[idx]),
                )
            )
        )
    return image_parts


def get_parts_by_type(response: GeminiGenerateContentResponse, part_type: Literal["text"] | str) -> list[GeminiPart]:
    """
    Filter response parts by their type.

    Args:
        response: The API response from Gemini.
        part_type: Type of parts to extract ("text" or a MIME type).

    Returns:
        List of response parts matching the requested type.
    """
    if response.candidates is None:
        if response.promptFeedback and response.promptFeedback.blockReason:
            feedback = response.promptFeedback
            raise ValueError(
                f"Gemini API blocked the request. Reason: {feedback.blockReason} ({feedback.blockReasonMessage})"
            )
        raise ValueError(
            "Gemini API returned no response candidates. If you are using the `IMAGE` modality, "
            "try changing it to `IMAGE+TEXT` to view the model's reasoning and understand why image generation failed."
        )
    parts = []
    for part in response.candidates[0].content.parts:
        if part_type == "text" and hasattr(part, "text") and part.text:
            parts.append(part)
        elif hasattr(part, "inlineData") and part.inlineData and part.inlineData.mimeType == part_type:
            parts.append(part)
        # Skip parts that don't match the requested type
    return parts


def get_text_from_response(response: GeminiGenerateContentResponse) -> str:
    """
    Extract and concatenate all text parts from the response.

    Args:
        response: The API response from Gemini.

    Returns:
        Combined text from all text parts in the response.
    """
    parts = get_parts_by_type(response, "text")
    return "\n".join([part.text for part in parts])


def get_image_from_response(response: GeminiGenerateContentResponse) -> torch.Tensor:
    image_tensors: list[torch.Tensor] = []
    parts = get_parts_by_type(response, "image/png")
    for part in parts:
        image_data = base64.b64decode(part.inlineData.data)
        returned_image = bytesio_to_image_tensor(BytesIO(image_data))
        image_tensors.append(returned_image)
    if len(image_tensors) == 0:
        return torch.zeros((1, 1024, 1024, 4))
    return torch.cat(image_tensors, dim=0)


def calculate_tokens_price(response: GeminiGenerateContentResponse) -> float | None:
    if not response.modelVersion:
        return None
    # Define prices (Cost per 1,000,000 tokens), see https://cloud.google.com/vertex-ai/generative-ai/pricing
    if response.modelVersion in ("gemini-2.5-pro-preview-05-06", "gemini-2.5-pro"):
        input_tokens_price = 1.25
        output_text_tokens_price = 10.0
        output_image_tokens_price = 0.0
    elif response.modelVersion in (
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash",
    ):
        input_tokens_price = 0.30
        output_text_tokens_price = 2.50
        output_image_tokens_price = 0.0
    elif response.modelVersion in (
        "gemini-2.5-flash-image-preview",
        "gemini-2.5-flash-image",
    ):
        input_tokens_price = 0.30
        output_text_tokens_price = 2.50
        output_image_tokens_price = 30.0
    elif response.modelVersion == "gemini-3-pro-preview":
        input_tokens_price = 2
        output_text_tokens_price = 12.0
        output_image_tokens_price = 0.0
    elif response.modelVersion == "gemini-3-pro-image-preview":
        input_tokens_price = 2
        output_text_tokens_price = 12.0
        output_image_tokens_price = 120.0
    else:
        return None
    final_price = response.usageMetadata.promptTokenCount * input_tokens_price
    if response.usageMetadata.candidatesTokensDetails:
        for i in response.usageMetadata.candidatesTokensDetails:
            if i.modality == Modality.IMAGE:
                final_price += output_image_tokens_price * i.tokenCount  # for Nano Banana models
            else:
                final_price += output_text_tokens_price * i.tokenCount
    if response.usageMetadata.thoughtsTokenCount:
        final_price += output_text_tokens_price * response.usageMetadata.thoughtsTokenCount
    return final_price / 1_000_000.0


class GeminiNode(IO.ComfyNode):
    """
    Node to generate text responses from a Gemini model.

    This node allows users to interact with Google's Gemini AI models, providing
    multimodal inputs (text, images, audio, video, files) to generate coherent
    text responses. The node works with the latest Gemini models, handling the
    API communication and response parsing.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiNode",
            display_name="Google Gemini",
            category="api node/text/Gemini",
            description="Generate text responses with Google's Gemini AI model. "
            "You can provide multiple types of inputs (text, images, audio, video) "
            "as context for generating more relevant and meaningful responses.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text inputs to the model, used to generate a response. "
                    "You can include detailed instructions, questions, or context for the model.",
                ),
                IO.Combo.Input(
                    "model",
                    options=GeminiModel,
                    default=GeminiModel.gemini_2_5_pro,
                    tooltip="The Gemini model to use for generating responses.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional image(s) to use as context for the model. "
                    "To include multiple images, you can use the Batch Images node.",
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Optional audio to use as context for the model.",
                ),
                IO.Video.Input(
                    "video",
                    optional=True,
                    tooltip="Optional video to use as context for the model.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
            ],
            outputs=[
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    def create_video_parts(cls, video_input: Input.Video) -> list[GeminiPart]:
        """Convert video input to Gemini API compatible parts."""

        base_64_string = video_to_base64_string(video_input, container_format=VideoContainer.MP4, codec=VideoCodec.H264)
        return [
            GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.video_mp4,
                    data=base_64_string,
                )
            )
        ]

    @classmethod
    def create_audio_parts(cls, audio_input: Input.Audio) -> list[GeminiPart]:
        """
        Convert audio input to Gemini API compatible parts.

        Args:
            audio_input: Audio input from ComfyUI, containing waveform tensor and sample rate.

        Returns:
            List of GeminiPart objects containing the encoded audio.
        """
        audio_parts: list[GeminiPart] = []
        for batch_index in range(audio_input["waveform"].shape[0]):
            # Recreate an IO.AUDIO object for the given batch dimension index
            audio_at_index = Input.Audio(
                waveform=audio_input["waveform"][batch_index].unsqueeze(0),
                sample_rate=audio_input["sample_rate"],
            )
            # Convert to MP3 format for compatibility with Gemini API
            audio_bytes = audio_to_base64_string(
                audio_at_index,
                container_format="mp3",
                codec_name="libmp3lame",
            )
            audio_parts.append(
                GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType=GeminiMimeType.audio_mp3,
                        data=audio_bytes,
                    )
                )
            )
        return audio_parts

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        images: torch.Tensor | None = None,
        audio: Input.Audio | None = None,
        video: Input.Video | None = None,
        files: list[GeminiPart] | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)

        # Create parts list with text prompt as the first part
        parts: list[GeminiPart] = [GeminiPart(text=prompt)]

        # Add other modal parts
        if images is not None:
            parts.extend(await create_image_parts(cls, images))
        if audio is not None:
            parts.extend(cls.create_audio_parts(audio))
        if video is not None:
            parts.extend(cls.create_video_parts(video))
        if files is not None:
            parts.extend(files)

        # Create response
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=f"{GEMINI_BASE_ENDPOINT}/{model}", method="POST"),
            data=GeminiGenerateContentRequest(
                contents=[
                    GeminiContent(
                        role=GeminiRole.user,
                        parts=parts,
                    )
                ]
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )

        output_text = get_text_from_response(response)
        return IO.NodeOutput(output_text or "Empty response from Gemini model...")


class GeminiInputFiles(IO.ComfyNode):
    """
    Loads and formats input files for use with the Gemini API.

    This node allows users to include text (.txt) and PDF (.pdf) files as input
    context for the Gemini model. Files are converted to the appropriate format
    required by the API and can be chained together to include multiple files
    in a single request.
    """

    @classmethod
    def define_schema(cls):
        """
        For details about the supported file input types, see:
        https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
        """
        input_dir = folder_paths.get_input_directory()
        input_files = [
            f
            for f in os.scandir(input_dir)
            if f.is_file()
            and (f.name.endswith(".txt") or f.name.endswith(".pdf"))
            and f.stat().st_size < GEMINI_MAX_INPUT_FILE_SIZE
        ]
        input_files = sorted(input_files, key=lambda x: x.name)
        input_files = [f.name for f in input_files]
        return IO.Schema(
            node_id="GeminiInputFiles",
            display_name="Gemini Input Files",
            category="api node/text/Gemini",
            description="Loads and prepares input files to include as inputs for Gemini LLM nodes. "
            "The files will be read by the Gemini model when generating a response. "
            "The contents of the text file count toward the token limit. "
            "🛈 TIP: Can be chained together with other Gemini Input File nodes.",
            inputs=[
                IO.Combo.Input(
                    "file",
                    options=input_files,
                    default=input_files[0] if input_files else None,
                    tooltip="Input files to include as context for the model. "
                    "Only accepts text (.txt) and PDF (.pdf) files for now.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "GEMINI_INPUT_FILES",
                    optional=True,
                    tooltip="An optional additional file(s) to batch together with the file loaded from this node. "
                    "Allows chaining of input files so that a single message can include multiple input files.",
                ),
            ],
            outputs=[
                IO.Custom("GEMINI_INPUT_FILES").Output(),
            ],
        )

    @classmethod
    def create_file_part(cls, file_path: str) -> GeminiPart:
        mime_type = GeminiMimeType.application_pdf if file_path.endswith(".pdf") else GeminiMimeType.text_plain
        # Use base64 string directly, not the data URI
        with open(file_path, "rb") as f:
            file_content = f.read()
        base64_str = base64.b64encode(file_content).decode("utf-8")

        return GeminiPart(
            inlineData=GeminiInlineData(
                mimeType=mime_type,
                data=base64_str,
            )
        )

    @classmethod
    def execute(cls, file: str, GEMINI_INPUT_FILES: list[GeminiPart] | None = None) -> IO.NodeOutput:
        """Loads and formats input files for Gemini API."""
        if GEMINI_INPUT_FILES is None:
            GEMINI_INPUT_FILES = []
        file_path = folder_paths.get_annotated_filepath(file)
        input_file_content = cls.create_file_part(file_path)
        return IO.NodeOutput([input_file_content] + GEMINI_INPUT_FILES)


class GeminiImage(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiImageNode",
            display_name="Nano Banana (Google Gemini Image)",
            category="api node/image/Gemini",
            description="Edit images synchronously via Google API.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt for generation",
                    default="",
                ),
                IO.Combo.Input(
                    "model",
                    options=GeminiImageModel,
                    default=GeminiImageModel.gemini_2_5_flash_image,
                    tooltip="The Gemini model to use for generating responses.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional image(s) to use as context for the model. "
                    "To include multiple images, you can use the Batch Images node.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="auto",
                    tooltip="Defaults to matching the output image size to that of your input image, "
                    "or otherwise generates 1:1 squares.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    tooltip="Choose 'IMAGE' for image-only output, or "
                    "'IMAGE+TEXT' to return both the generated image and a text response.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
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
        model: str,
        seed: int,
        images: torch.Tensor | None = None,
        files: list[GeminiPart] | None = None,
        aspect_ratio: str = "auto",
        response_modalities: str = "IMAGE+TEXT",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        parts: list[GeminiPart] = [GeminiPart(text=prompt)]

        if not aspect_ratio:
            aspect_ratio = "auto"  # for backward compatability with old workflows; to-do remove this in December
        image_config = GeminiImageConfig(aspectRatio=aspect_ratio)

        if images is not None:
            parts.extend(await create_image_parts(cls, images))
        if files is not None:
            parts.extend(files)

        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=f"{GEMINI_BASE_ENDPOINT}/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role=GeminiRole.user, parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                    imageConfig=None if aspect_ratio == "auto" else image_config,
                ),
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(get_image_from_response(response), get_text_from_response(response))


class GeminiImage2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiImage2Node",
            display_name="Nano Banana Pro (Google Gemini Image)",
            category="api node/image/Gemini",
            description="Generate or edit images synchronously via Google Vertex API.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt describing the image to generate or the edits to apply. "
                    "Include any constraints, styles, or details the model should follow.",
                    default="",
                ),
                IO.Combo.Input(
                    "model",
                    options=["gemini-3-pro-image-preview"],
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When the seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="auto",
                    tooltip="If set to 'auto', matches your input image's aspect ratio; "
                    "if no image is provided, a 16:9 square is usually generated.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "2K", "4K"],
                    tooltip="Target output resolution. For 2K/4K the native Gemini upscaler is used.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    tooltip="Choose 'IMAGE' for image-only output, or "
                    "'IMAGE+TEXT' to return both the generated image and a text response.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference image(s). "
                    "To include multiple images, use the Batch Images node (up to 14).",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
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
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        images: torch.Tensor | None = None,
        files: list[GeminiPart] | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)

        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            if get_number_of_images(images) > 14:
                raise ValueError("The current maximum number of supported images is 14.")
            parts.extend(await create_image_parts(cls, images))
        if files is not None:
            parts.extend(files)

        image_config = GeminiImageConfig(imageSize=resolution)
        if aspect_ratio != "auto":
            image_config.aspectRatio = aspect_ratio

        response = await sync_op(
            cls,
            ApiEndpoint(path=f"{GEMINI_BASE_ENDPOINT}/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role=GeminiRole.user, parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                    imageConfig=image_config,
                ),
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(get_image_from_response(response), get_text_from_response(response))


class GeminiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            GeminiNode,
            GeminiImage,
            GeminiImage2,
            GeminiInputFiles,
        ]


async def comfy_entrypoint() -> GeminiExtension:
    return GeminiExtension()
