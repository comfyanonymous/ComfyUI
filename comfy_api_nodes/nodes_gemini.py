"""
API Nodes for Gemini Multimodal LLM Usage via Remote API
See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from enum import Enum
from io import BytesIO
from typing import Literal, Optional

import torch
from typing_extensions import override

import folder_paths
from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api.util import VideoCodec, VideoContainer
from comfy_api_nodes.apis import (
    GeminiContent,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiInlineData,
    GeminiMimeType,
    GeminiPart,
)
from comfy_api_nodes.apis.gemini_api import (
    GeminiImageConfig,
    GeminiImageGenerateContentRequest,
    GeminiImageGenerationConfig,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_to_base64_string,
    bytesio_to_image_tensor,
    sync_op,
    tensor_to_base64_string,
    validate_string,
    video_to_base64_string,
)
from server import PromptServer

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


class GeminiImageModel(str, Enum):
    """
    Gemini Image Model Names allowed by comfy-api
    """

    gemini_2_5_flash_image_preview = "gemini-2.5-flash-image-preview"
    gemini_2_5_flash_image = "gemini-2.5-flash-image"


def create_image_parts(image_input: torch.Tensor) -> list[GeminiPart]:
    """
    Convert image tensor input to Gemini API compatible parts.

    Args:
        image_input: Batch of image tensors from ComfyUI.

    Returns:
        List of GeminiPart objects containing the encoded images.
    """
    image_parts: list[GeminiPart] = []
    for image_index in range(image_input.shape[0]):
        image_as_b64 = tensor_to_base64_string(image_input[image_index].unsqueeze(0))
        image_parts.append(
            GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.image_png,
                    data=image_as_b64,
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
        images: Optional[torch.Tensor] = None,
        audio: Optional[Input.Audio] = None,
        video: Optional[Input.Video] = None,
        files: Optional[list[GeminiPart]] = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)

        # Create parts list with text prompt as the first part
        parts: list[GeminiPart] = [GeminiPart(text=prompt)]

        # Add other modal parts
        if images is not None:
            image_parts = create_image_parts(images)
            parts.extend(image_parts)
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
                        role="user",
                        parts=parts,
                    )
                ]
            ),
            response_model=GeminiGenerateContentResponse,
        )

        # Get result output
        output_text = get_text_from_response(response)
        if output_text:
            # Not a true chat history like the OpenAI Chat node. It is emulated so the frontend can show a copy button.
            render_spec = {
                "node_id": cls.hidden.unique_id,
                "component": "ChatHistoryWidget",
                "props": {
                    "history": json.dumps(
                        [
                            {
                                "prompt": prompt,
                                "response": output_text,
                                "response_id": str(uuid.uuid4()),
                                "timestamp": time.time(),
                            }
                        ]
                    ),
                },
            }
            PromptServer.instance.send_sync(
                "display_component",
                render_spec,
            )

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
    def execute(cls, file: str, GEMINI_INPUT_FILES: Optional[list[GeminiPart]] = None) -> IO.NodeOutput:
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
            display_name="Google Gemini Image",
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
        images: Optional[torch.Tensor] = None,
        files: Optional[list[GeminiPart]] = None,
        aspect_ratio: str = "auto",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        parts: list[GeminiPart] = [GeminiPart(text=prompt)]

        if not aspect_ratio:
            aspect_ratio = "auto"  # for backward compatability with old workflows; to-do remove this in December
        image_config = GeminiImageConfig(aspectRatio=aspect_ratio)

        if images is not None:
            image_parts = create_image_parts(images)
            parts.extend(image_parts)
        if files is not None:
            parts.extend(files)

        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=f"{GEMINI_BASE_ENDPOINT}/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role="user", parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=["TEXT", "IMAGE"],
                    imageConfig=None if aspect_ratio == "auto" else image_config,
                ),
            ),
            response_model=GeminiGenerateContentResponse,
        )

        output_image = get_image_from_response(response)
        output_text = get_text_from_response(response)
        if output_text:
            # Not a true chat history like the OpenAI Chat node. It is emulated so the frontend can show a copy button.
            render_spec = {
                "node_id": cls.hidden.unique_id,
                "component": "ChatHistoryWidget",
                "props": {
                    "history": json.dumps(
                        [
                            {
                                "prompt": prompt,
                                "response": output_text,
                                "response_id": str(uuid.uuid4()),
                                "timestamp": time.time(),
                            }
                        ]
                    ),
                },
            }
            PromptServer.instance.send_sync(
                "display_component",
                render_spec,
            )

        output_text = output_text or "Empty response from Gemini model..."
        return IO.NodeOutput(output_image, output_text)


class GeminiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            GeminiNode,
            GeminiImage,
            GeminiInputFiles,
        ]


async def comfy_entrypoint() -> GeminiExtension:
    return GeminiExtension()
