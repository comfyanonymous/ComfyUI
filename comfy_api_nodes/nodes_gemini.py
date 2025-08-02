"""
API Nodes for Gemini Multimodal LLM Usage via Remote API
See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
"""
from __future__ import annotations


import os
from enum import Enum
from typing import Optional, Literal

import torch

import folder_paths
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from server import PromptServer
from comfy_api_nodes.apis import (
    GeminiContent,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiInlineData,
    GeminiPart,
    GeminiMimeType,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    validate_string,
    audio_to_base64_string,
    video_to_base64_string,
    tensor_to_base64_string,
)


GEMINI_BASE_ENDPOINT = "/proxy/vertexai/gemini"
GEMINI_MAX_INPUT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class GeminiModel(str, Enum):
    """
    Gemini Model Names allowed by comfy-api
    """

    gemini_2_5_pro_preview_05_06 = "gemini-2.5-pro-preview-05-06"
    gemini_2_5_flash_preview_04_17 = "gemini-2.5-flash-preview-04-17"


def get_gemini_endpoint(
    model: GeminiModel,
) -> ApiEndpoint[GeminiGenerateContentRequest, GeminiGenerateContentResponse]:
    """
    Get the API endpoint for a given Gemini model.

    Args:
        model: The Gemini model to use, either as enum or string value.

    Returns:
        ApiEndpoint configured for the specific Gemini model.
    """
    if isinstance(model, str):
        model = GeminiModel(model)
    return ApiEndpoint(
        path=f"{GEMINI_BASE_ENDPOINT}/{model.value}",
        method=HttpMethod.POST,
        request_model=GeminiGenerateContentRequest,
        response_model=GeminiGenerateContentResponse,
    )


class GeminiNode(ComfyNodeABC):
    """
    Node to generate text responses from a Gemini model.

    This node allows users to interact with Google's Gemini AI models, providing
    multimodal inputs (text, images, audio, video, files) to generate coherent
    text responses. The node works with the latest Gemini models, handling the
    API communication and response parsing.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text inputs to the model, used to generate a response. You can include detailed instructions, questions, or context for the model.",
                    },
                ),
                "model": (
                    IO.COMBO,
                    {
                        "tooltip": "The Gemini model to use for generating responses.",
                        "options": [model.value for model in GeminiModel],
                        "default": GeminiModel.gemini_2_5_pro_preview_05_06.value,
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "When seed is fixed to a specific value, the model makes a best effort to provide the same response for repeated requests. Deterministic output isn't guaranteed. Also, changing the model or parameter settings, such as the temperature, can cause variations in the response even when you use the same seed value. By default, a random seed value is used.",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional image(s) to use as context for the model. To include multiple images, you can use the Batch Images node.",
                    },
                ),
                "audio": (
                    IO.AUDIO,
                    {
                        "tooltip": "Optional audio to use as context for the model.",
                        "default": None,
                    },
                ),
                "video": (
                    IO.VIDEO,
                    {
                        "tooltip": "Optional video to use as context for the model.",
                        "default": None,
                    },
                ),
                "files": (
                    "GEMINI_INPUT_FILES",
                    {
                        "default": None,
                        "tooltip": "Optional file(s) to use as context for the model. Accepts inputs from the Gemini Generate Content Input Files node.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Generate text responses with Google's Gemini AI model. You can provide multiple types of inputs (text, images, audio, video) as context for generating more relevant and meaningful responses."
    RETURN_TYPES = ("STRING",)
    FUNCTION = "api_call"
    CATEGORY = "api node/text/Gemini"
    API_NODE = True

    def get_parts_from_response(
        self, response: GeminiGenerateContentResponse
    ) -> list[GeminiPart]:
        """
        Extract all parts from the Gemini API response.

        Args:
            response: The API response from Gemini.

        Returns:
            List of response parts from the first candidate.
        """
        return response.candidates[0].content.parts

    def get_parts_by_type(
        self, response: GeminiGenerateContentResponse, part_type: Literal["text"] | str
    ) -> list[GeminiPart]:
        """
        Filter response parts by their type.

        Args:
            response: The API response from Gemini.
            part_type: Type of parts to extract ("text" or a MIME type).

        Returns:
            List of response parts matching the requested type.
        """
        parts = []
        for part in self.get_parts_from_response(response):
            if part_type == "text" and hasattr(part, "text") and part.text:
                parts.append(part)
            elif (
                hasattr(part, "inlineData")
                and part.inlineData
                and part.inlineData.mimeType == part_type
            ):
                parts.append(part)
            # Skip parts that don't match the requested type
        return parts

    def get_text_from_response(self, response: GeminiGenerateContentResponse) -> str:
        """
        Extract and concatenate all text parts from the response.

        Args:
            response: The API response from Gemini.

        Returns:
            Combined text from all text parts in the response.
        """
        parts = self.get_parts_by_type(response, "text")
        return "\n".join([part.text for part in parts])

    def create_video_parts(self, video_input: IO.VIDEO, **kwargs) -> list[GeminiPart]:
        """
        Convert video input to Gemini API compatible parts.

        Args:
            video_input: Video tensor from ComfyUI.
            **kwargs: Additional arguments to pass to the conversion function.

        Returns:
            List of GeminiPart objects containing the encoded video.
        """
        from comfy_api.util import VideoContainer, VideoCodec
        base_64_string = video_to_base64_string(
            video_input,
            container_format=VideoContainer.MP4,
            codec=VideoCodec.H264
        )
        return [
            GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.video_mp4,
                    data=base_64_string,
                )
            )
        ]

    def create_audio_parts(self, audio_input: IO.AUDIO) -> list[GeminiPart]:
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
            audio_at_index = {
                "waveform": audio_input["waveform"][batch_index].unsqueeze(0),
                "sample_rate": audio_input["sample_rate"],
            }
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

    def create_image_parts(self, image_input: torch.Tensor) -> list[GeminiPart]:
        """
        Convert image tensor input to Gemini API compatible parts.

        Args:
            image_input: Batch of image tensors from ComfyUI.

        Returns:
            List of GeminiPart objects containing the encoded images.
        """
        image_parts: list[GeminiPart] = []
        for image_index in range(image_input.shape[0]):
            image_as_b64 = tensor_to_base64_string(
                image_input[image_index].unsqueeze(0)
            )
            image_parts.append(
                GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType=GeminiMimeType.image_png,
                        data=image_as_b64,
                    )
                )
            )
        return image_parts

    def create_text_part(self, text: str) -> GeminiPart:
        """
        Create a text part for the Gemini API request.

        Args:
            text: The text content to include in the request.

        Returns:
            A GeminiPart object with the text content.
        """
        return GeminiPart(text=text)

    async def api_call(
        self,
        prompt: str,
        model: GeminiModel,
        images: Optional[IO.IMAGE] = None,
        audio: Optional[IO.AUDIO] = None,
        video: Optional[IO.VIDEO] = None,
        files: Optional[list[GeminiPart]] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str]:
        # Validate inputs
        validate_string(prompt, strip_whitespace=False)

        # Create parts list with text prompt as the first part
        parts: list[GeminiPart] = [self.create_text_part(prompt)]

        # Add other modal parts
        if images is not None:
            image_parts = self.create_image_parts(images)
            parts.extend(image_parts)
        if audio is not None:
            parts.extend(self.create_audio_parts(audio))
        if video is not None:
            parts.extend(self.create_video_parts(video))
        if files is not None:
            parts.extend(files)

        # Create response
        response = await SynchronousOperation(
            endpoint=get_gemini_endpoint(model),
            request=GeminiGenerateContentRequest(
                contents=[
                    GeminiContent(
                        role="user",
                        parts=parts,
                    )
                ]
            ),
            auth_kwargs=kwargs,
        ).execute()

        # Get result output
        output_text = self.get_text_from_response(response)
        if unique_id and output_text:
            PromptServer.instance.send_progress_text(output_text, node_id=unique_id)

        return (output_text or "Empty response from Gemini model...",)


class GeminiInputFiles(ComfyNodeABC):
    """
    Loads and formats input files for use with the Gemini API.

    This node allows users to include text (.txt) and PDF (.pdf) files as input
    context for the Gemini model. Files are converted to the appropriate format
    required by the API and can be chained together to include multiple files
    in a single request.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
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
        return {
            "required": {
                "file": (
                    IO.COMBO,
                    {
                        "tooltip": "Input files to include as context for the model. Only accepts text (.txt) and PDF (.pdf) files for now.",
                        "options": input_files,
                        "default": input_files[0] if input_files else None,
                    },
                ),
            },
            "optional": {
                "GEMINI_INPUT_FILES": (
                    "GEMINI_INPUT_FILES",
                    {
                        "tooltip": "An optional additional file(s) to batch together with the file loaded from this node. Allows chaining of input files so that a single message can include multiple input files.",
                        "default": None,
                    },
                ),
            },
        }

    DESCRIPTION = "Loads and prepares input files to include as inputs for Gemini LLM nodes. The files will be read by the Gemini model when generating a response. The contents of the text file count toward the token limit. 🛈 TIP: Can be chained together with other Gemini Input File nodes."
    RETURN_TYPES = ("GEMINI_INPUT_FILES",)
    FUNCTION = "prepare_files"
    CATEGORY = "api node/text/Gemini"

    def create_file_part(self, file_path: str) -> GeminiPart:
        mime_type = (
            GeminiMimeType.application_pdf
            if file_path.endswith(".pdf")
            else GeminiMimeType.text_plain
        )
        # Use base64 string directly, not the data URI
        with open(file_path, "rb") as f:
            file_content = f.read()
        import base64
        base64_str = base64.b64encode(file_content).decode("utf-8")

        return GeminiPart(
            inlineData=GeminiInlineData(
                mimeType=mime_type,
                data=base64_str,
            )
        )

    def prepare_files(
        self, file: str, GEMINI_INPUT_FILES: list[GeminiPart] = []
    ) -> tuple[list[GeminiPart]]:
        """
        Loads and formats input files for Gemini API.
        """
        file_path = folder_paths.get_annotated_filepath(file)
        input_file_content = self.create_file_part(file_path)
        files = [input_file_content] + GEMINI_INPUT_FILES
        return (files,)


NODE_CLASS_MAPPINGS = {
    "GeminiNode": GeminiNode,
    "GeminiInputFiles": GeminiInputFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNode": "Google Gemini",
    "GeminiInputFiles": "Gemini Input Files",
}
