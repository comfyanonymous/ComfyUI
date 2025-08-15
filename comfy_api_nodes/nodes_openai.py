import io
from typing import TypedDict, Optional
import json
import os
import time
import re
import uuid
from enum import Enum
from inspect import cleandoc
import numpy as np
import torch
from PIL import Image
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from server import PromptServer
import folder_paths


from comfy_api_nodes.apis import (
    OpenAIImageGenerationRequest,
    OpenAIImageEditRequest,
    OpenAIImageGenerationResponse,
    OpenAICreateResponse,
    OpenAIResponse,
    CreateModelResponseProperties,
    Item,
    Includable,
    OutputContent,
    InputImageContent,
    Detail,
    InputTextContent,
    InputMessage,
    InputMessageContentList,
    InputContent,
    InputFileContent,
)

from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)

from comfy_api_nodes.apinode_utils import (
    downscale_image_tensor,
    validate_and_cast_response,
    validate_string,
    tensor_to_base64_string,
    text_filepath_to_data_uri,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input


RESPONSES_ENDPOINT = "/proxy/openai/v1/responses"
STARTING_POINT_ID_PATTERN = r"<starting_point_id:(.*)>"


class HistoryEntry(TypedDict):
    """Type definition for a single history entry in the chat."""

    prompt: str
    response: str
    response_id: str
    timestamp: float


class ChatHistory(TypedDict):
    """Type definition for the chat history dictionary."""

    __annotations__: dict[str, list[HistoryEntry]]


class SupportedOpenAIModel(str, Enum):
    o4_mini = "o4-mini"
    o1 = "o1"
    o3 = "o3"
    o1_pro = "o1-pro"
    gpt_4o = "gpt-4o"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1_nano = "gpt-4.1-nano"


class OpenAIDalle2(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's DALL·E 2 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for DALL·E",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["256x256", "512x512", "1024x1024"],
                        "default": "1024x1024",
                        "tooltip": "Image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    async def api_call(
        self,
        prompt,
        seed=0,
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
        unique_id=None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        model = "dall-e-2"
        path = "/proxy/openai/images/generations"
        content_type = "application/json"
        request_class = OpenAIImageGenerationRequest
        img_binary = None

        if image is not None and mask is not None:
            path = "/proxy/openai/images/edits"
            content_type = "multipart/form-data"
            request_class = OpenAIImageEditRequest

            input_tensor = image.squeeze().cpu()
            height, width, channels = input_tensor.shape
            rgba_tensor = torch.ones(height, width, 4, device="cpu")
            rgba_tensor[:, :, :channels] = input_tensor

            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            rgba_tensor[:, :, 3] = 1 - mask.squeeze().cpu()

            rgba_tensor = downscale_image_tensor(rgba_tensor.unsqueeze(0)).squeeze()

            image_np = (rgba_tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img_binary = img_byte_arr  # .getvalue()
            img_binary.name = "image.png"
        elif image is not None or mask is not None:
            raise Exception("Dall-E 2 image editing requires an image AND a mask")

        # Build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                seed=seed,
            ),
            files=(
                {
                    "image": img_binary,
                }
                if img_binary
                else None
            ),
            content_type=content_type,
            auth_kwargs=kwargs,
        )

        response = await operation.execute()

        img_tensor = await validate_and_cast_response(response, node_id=unique_id)
        return (img_tensor,)


class OpenAIDalle3(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's DALL·E 3 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for DALL·E",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["standard", "hd"],
                        "default": "standard",
                        "tooltip": "Image quality",
                    },
                ),
                "style": (
                    IO.COMBO,
                    {
                        "options": ["natural", "vivid"],
                        "default": "natural",
                        "tooltip": "Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["1024x1024", "1024x1792", "1792x1024"],
                        "default": "1024x1024",
                        "tooltip": "Image size",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    async def api_call(
        self,
        prompt,
        seed=0,
        style="natural",
        quality="standard",
        size="1024x1024",
        unique_id=None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        model = "dall-e-3"

        # build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/openai/images/generations",
                method=HttpMethod.POST,
                request_model=OpenAIImageGenerationRequest,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=OpenAIImageGenerationRequest(
                model=model,
                prompt=prompt,
                quality=quality,
                size=size,
                style=style,
                seed=seed,
            ),
            auth_kwargs=kwargs,
        )

        response = await operation.execute()

        img_tensor = await validate_and_cast_response(response, node_id=unique_id)
        return (img_tensor,)


class OpenAIGPTImage1(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's GPT Image 1 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for GPT Image 1",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["low", "medium", "high"],
                        "default": "low",
                        "tooltip": "Image quality, affects cost and generation time.",
                    },
                ),
                "background": (
                    IO.COMBO,
                    {
                        "options": ["opaque", "transparent"],
                        "default": "opaque",
                        "tooltip": "Return image with or without background",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["auto", "1024x1024", "1024x1536", "1536x1024"],
                        "default": "auto",
                        "tooltip": "Image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    async def api_call(
        self,
        prompt,
        seed=0,
        quality="low",
        background="opaque",
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
        unique_id=None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        model = "gpt-image-1"
        path = "/proxy/openai/images/generations"
        content_type = "application/json"
        request_class = OpenAIImageGenerationRequest
        files = []

        if image is not None:
            path = "/proxy/openai/images/edits"
            request_class = OpenAIImageEditRequest
            content_type = "multipart/form-data"

            batch_size = image.shape[0]

            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_image_tensor(single_image).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                if batch_size == 1:
                    files.append(("image", (f"image_{i}.png", img_byte_arr, "image/png")))
                else:
                    files.append(("image[]", (f"image_{i}.png", img_byte_arr, "image/png")))

        if mask is not None:
            if image is None:
                raise Exception("Cannot use a mask without an input image")
            if image.shape[0] != 1:
                raise Exception("Cannot use a mask with multiple image")
            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

            scaled_mask = downscale_image_tensor(rgba_mask.unsqueeze(0)).squeeze()

            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_img_byte_arr = io.BytesIO()
            mask_img.save(mask_img_byte_arr, format="PNG")
            mask_img_byte_arr.seek(0)
            files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

        # Build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                quality=quality,
                background=background,
                n=n,
                seed=seed,
                size=size,
            ),
            files=files if files else None,
            content_type=content_type,
            auth_kwargs=kwargs,
        )

        response = await operation.execute()

        img_tensor = await validate_and_cast_response(response, node_id=unique_id)
        return (img_tensor,)


class OpenAITextNode(ComfyNodeABC):
    """
    Base class for OpenAI text generation nodes.
    """

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "api_call"
    CATEGORY = "api node/text/OpenAI"
    API_NODE = True


class OpenAIChatNode(OpenAITextNode):
    """
    Node to generate text responses from an OpenAI model.
    """

    def __init__(self) -> None:
        """Initialize the chat node with a new session ID and empty history."""
        self.current_session_id: str = str(uuid.uuid4())
        self.history: dict[str, list[HistoryEntry]] = {}
        self.previous_response_id: Optional[str] = None

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text inputs to the model, used to generate a response.",
                    },
                ),
                "persist_context": (
                    IO.BOOLEAN,
                    {
                        "default": True,
                        "tooltip": "Persist chat context between calls (multi-turn conversation)",
                    },
                ),
                "model": model_field_to_node_input(
                    IO.COMBO,
                    OpenAICreateResponse,
                    "model",
                    enum_type=SupportedOpenAIModel,
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
                "files": (
                    "OPENAI_INPUT_FILES",
                    {
                        "default": None,
                        "tooltip": "Optional file(s) to use as context for the model. Accepts inputs from the OpenAI Chat Input Files node.",
                    },
                ),
                "advanced_options": (
                    "OPENAI_CHAT_CONFIG",
                    {
                        "default": None,
                        "tooltip": "Optional configuration for the model. Accepts inputs from the OpenAI Chat Advanced Options node.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Generate text responses from an OpenAI model."

    async def get_result_response(
        self,
        response_id: str,
        include: Optional[list[Includable]] = None,
        auth_kwargs: Optional[dict[str, str]] = None,
    ) -> OpenAIResponse:
        """
        Retrieve a model response with the given ID from the OpenAI API.

        Args:
            response_id (str): The ID of the response to retrieve.
            include (Optional[List[Includable]]): Additional fields to include
                in the response. See the `include` parameter for Response
                creation above for more information.

        """
        return await PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"{RESPONSES_ENDPOINT}/{response_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=OpenAIResponse,
                query_params={"include": include},
            ),
            completed_statuses=["completed"],
            failed_statuses=["failed"],
            status_extractor=lambda response: response.status,
            auth_kwargs=auth_kwargs,
        ).execute()

    def get_message_content_from_response(
        self, response: OpenAIResponse
    ) -> list[OutputContent]:
        """Extract message content from the API response."""
        for output in response.output:
            if output.root.type == "message":
                return output.root.content
        raise TypeError("No output message found in response")

    def get_text_from_message_content(
        self, message_content: list[OutputContent]
    ) -> str:
        """Extract text content from message content."""
        for content_item in message_content:
            if content_item.root.type == "output_text":
                return str(content_item.root.text)
        return "No text output found in response"

    def get_history_text(self, session_id: str) -> str:
        """Convert the entire history for a given session to JSON string."""
        return json.dumps(self.history[session_id])

    def display_history_on_node(self, session_id: str, node_id: str) -> None:
        """Display formatted chat history on the node UI."""
        render_spec = {
            "node_id": node_id,
            "component": "ChatHistoryWidget",
            "props": {
                "history": self.get_history_text(session_id),
            },
        }
        PromptServer.instance.send_sync(
            "display_component",
            render_spec,
        )

    def add_to_history(
        self, session_id: str, prompt: str, output_text: str, response_id: str
    ) -> None:
        """Add a new entry to the chat history."""
        if session_id not in self.history:
            self.history[session_id] = []
        self.history[session_id].append(
            {
                "prompt": prompt,
                "response": output_text,
                "response_id": response_id,
                "timestamp": time.time(),
            }
        )

    def parse_output_text_from_response(self, response: OpenAIResponse) -> str:
        """Extract text output from the API response."""
        message_contents = self.get_message_content_from_response(response)
        return self.get_text_from_message_content(message_contents)

    def generate_new_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())

    def get_session_id(self, persist_context: bool) -> str:
        """Get the current or generate a new session ID based on context persistence."""
        return (
            self.current_session_id
            if persist_context
            else self.generate_new_session_id()
        )

    def tensor_to_input_image_content(
        self, image: torch.Tensor, detail_level: Detail = "auto"
    ) -> InputImageContent:
        """Convert a tensor to an input image content object."""
        return InputImageContent(
            detail=detail_level,
            image_url=f"data:image/png;base64,{tensor_to_base64_string(image)}",
            type="input_image",
        )

    def create_input_message_contents(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        files: Optional[list[InputFileContent]] = None,
    ) -> InputMessageContentList:
        """Create a list of input message contents from prompt and optional image."""
        content_list: list[InputContent] = [
            InputTextContent(text=prompt, type="input_text"),
        ]
        if image is not None:
            for i in range(image.shape[0]):
                content_list.append(
                    self.tensor_to_input_image_content(image[i].unsqueeze(0))
                )
        if files is not None:
            content_list.extend(files)

        return InputMessageContentList(
            root=content_list,
        )

    def parse_response_id_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract response ID from prompt if it exists."""
        parsed_id = re.search(STARTING_POINT_ID_PATTERN, prompt)
        return parsed_id.group(1) if parsed_id else None

    def strip_response_tag_from_prompt(self, prompt: str) -> str:
        """Remove the response ID tag from the prompt."""
        return re.sub(STARTING_POINT_ID_PATTERN, "", prompt.strip())

    def delete_history_after_response_id(
        self, new_start_id: str, session_id: str
    ) -> None:
        """Delete history entries after a specific response ID."""
        if session_id not in self.history:
            return

        new_history = []
        i = 0
        while (
            i < len(self.history[session_id])
            and self.history[session_id][i]["response_id"] != new_start_id
        ):
            new_history.append(self.history[session_id][i])
            i += 1

        # Since it's the new starting point (not the response being edited), we include it as well
        if i < len(self.history[session_id]):
            new_history.append(self.history[session_id][i])

        self.history[session_id] = new_history

    async def api_call(
        self,
        prompt: str,
        persist_context: bool,
        model: SupportedOpenAIModel,
        unique_id: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
        files: Optional[list[InputFileContent]] = None,
        advanced_options: Optional[CreateModelResponseProperties] = None,
        **kwargs,
    ) -> tuple[str]:
        # Validate inputs
        validate_string(prompt, strip_whitespace=False)

        session_id = self.get_session_id(persist_context)
        response_id_override = self.parse_response_id_from_prompt(prompt)
        if response_id_override:
            is_starting_from_beginning = response_id_override == "start"
            if is_starting_from_beginning:
                self.history[session_id] = []
                previous_response_id = None
            else:
                previous_response_id = response_id_override
                self.delete_history_after_response_id(response_id_override, session_id)
            prompt = self.strip_response_tag_from_prompt(prompt)
        elif persist_context:
            previous_response_id = self.previous_response_id
        else:
            previous_response_id = None

        # Create response
        create_response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path=RESPONSES_ENDPOINT,
                method=HttpMethod.POST,
                request_model=OpenAICreateResponse,
                response_model=OpenAIResponse,
            ),
            request=OpenAICreateResponse(
                input=[
                    Item(
                        root=InputMessage(
                            content=self.create_input_message_contents(
                                prompt, images, files
                            ),
                            role="user",
                        )
                    ),
                ],
                store=True,
                stream=False,
                model=model,
                previous_response_id=previous_response_id,
                **(
                    advanced_options.model_dump(exclude_none=True)
                    if advanced_options
                    else {}
                ),
            ),
            auth_kwargs=kwargs,
        ).execute()
        response_id = create_response.id

        # Get result output
        result_response = await self.get_result_response(response_id, auth_kwargs=kwargs)
        output_text = self.parse_output_text_from_response(result_response)

        # Update history
        self.add_to_history(session_id, prompt, output_text, response_id)
        self.display_history_on_node(session_id, unique_id)
        self.previous_response_id = response_id

        return (output_text,)


class OpenAIInputFiles(ComfyNodeABC):
    """
    Loads and formats input files for OpenAI API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """
        For details about the supported file input types, see:
        https://platform.openai.com/docs/guides/pdf-files?api-mode=responses
        """
        input_dir = folder_paths.get_input_directory()
        input_files = [
            f
            for f in os.scandir(input_dir)
            if f.is_file()
            and (f.name.endswith(".txt") or f.name.endswith(".pdf"))
            and f.stat().st_size < 32 * 1024 * 1024
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
                "OPENAI_INPUT_FILES": (
                    "OPENAI_INPUT_FILES",
                    {
                        "tooltip": "An optional additional file(s) to batch together with the file loaded from this node. Allows chaining of input files so that a single message can include multiple input files.",
                        "default": None,
                    },
                ),
            },
        }

    DESCRIPTION = "Loads and prepares input files (text, pdf, etc.) to include as inputs for the OpenAI Chat Node. The files will be read by the OpenAI model when generating a response. 🛈 TIP: Can be chained together with other OpenAI Input File nodes."
    RETURN_TYPES = ("OPENAI_INPUT_FILES",)
    FUNCTION = "prepare_files"
    CATEGORY = "api node/text/OpenAI"

    def create_input_file_content(self, file_path: str) -> InputFileContent:
        return InputFileContent(
            file_data=text_filepath_to_data_uri(file_path),
            filename=os.path.basename(file_path),
            type="input_file",
        )

    def prepare_files(
        self, file: str, OPENAI_INPUT_FILES: list[InputFileContent] = []
    ) -> tuple[list[InputFileContent]]:
        """
        Loads and formats input files for OpenAI API.
        """
        file_path = folder_paths.get_annotated_filepath(file)
        input_file_content = self.create_input_file_content(file_path)
        files = [input_file_content] + OPENAI_INPUT_FILES
        return (files,)


class OpenAIChatConfig(ComfyNodeABC):
    """Allows setting additional configuration for the OpenAI Chat Node."""

    RETURN_TYPES = ("OPENAI_CHAT_CONFIG",)
    FUNCTION = "configure"
    DESCRIPTION = (
        "Allows specifying advanced configuration options for the OpenAI Chat Nodes."
    )
    CATEGORY = "api node/text/OpenAI"

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "truncation": (
                    IO.COMBO,
                    {
                        "options": ["auto", "disabled"],
                        "default": "auto",
                        "tooltip": "The truncation strategy to use for the model response. auto: If the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.disabled: If a model response will exceed the context window size for a model, the request will fail with a 400 error",
                    },
                ),
            },
            "optional": {
                "max_output_tokens": model_field_to_node_input(
                    IO.INT,
                    OpenAICreateResponse,
                    "max_output_tokens",
                    min=16,
                    default=4096,
                    max=16384,
                    tooltip="An upper bound for the number of tokens that can be generated for a response, including visible output tokens",
                ),
                "instructions": model_field_to_node_input(
                    IO.STRING, OpenAICreateResponse, "instructions", multiline=True
                ),
            },
        }

    def configure(
        self,
        truncation: bool,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> tuple[CreateModelResponseProperties]:
        """
        Configure advanced options for the OpenAI Chat Node.

        Note:
            While `top_p` and `temperature` are listed as properties in the
            spec, they are not supported for all models (e.g., o4-mini).
            They are not exposed as inputs at all to avoid having to manually
            remove depending on model choice.
        """
        return (
            CreateModelResponseProperties(
                instructions=instructions,
                truncation=truncation,
                max_output_tokens=max_output_tokens,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "OpenAIDalle2": OpenAIDalle2,
    "OpenAIDalle3": OpenAIDalle3,
    "OpenAIGPTImage1": OpenAIGPTImage1,
    "OpenAIChatNode": OpenAIChatNode,
    "OpenAIInputFiles": OpenAIInputFiles,
    "OpenAIChatConfig": OpenAIChatConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIDalle2": "OpenAI DALL·E 2",
    "OpenAIDalle3": "OpenAI DALL·E 3",
    "OpenAIGPTImage1": "OpenAI GPT Image 1",
    "OpenAIChatNode": "OpenAI Chat",
    "OpenAIInputFiles": "OpenAI Chat Input Files",
    "OpenAIChatConfig": "OpenAI Chat Advanced Options",
}
