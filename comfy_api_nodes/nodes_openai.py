from io import BytesIO
from typing import Optional, Union
import json
import os
import time
import uuid
from enum import Enum
from inspect import cleandoc
import numpy as np
import torch
from PIL import Image
from server import PromptServer
import folder_paths
import base64
from comfy_api.latest import IO, ComfyExtension
from typing_extensions import override


from comfy_api_nodes.apis import (
    OpenAIImageGenerationRequest,
    OpenAIImageEditRequest,
    OpenAIImageGenerationResponse,
    OpenAICreateResponse,
    OpenAIResponse,
    CreateModelResponseProperties,
    Item,
    OutputContent,
    InputImageContent,
    Detail,
    InputTextContent,
    InputMessage,
    InputMessageContentList,
    InputContent,
    InputFileContent,
)

from comfy_api_nodes.util import (
    downscale_image_tensor,
    download_url_to_bytesio,
    validate_string,
    tensor_to_base64_string,
    ApiEndpoint,
    sync_op,
    poll_op,
    text_filepath_to_data_uri,
)


RESPONSES_ENDPOINT = "/proxy/openai/v1/responses"
STARTING_POINT_ID_PATTERN = r"<starting_point_id:(.*)>"


class SupportedOpenAIModel(str, Enum):
    o4_mini = "o4-mini"
    o1 = "o1"
    o3 = "o3"
    o1_pro = "o1-pro"
    gpt_4o = "gpt-4o"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1_nano = "gpt-4.1-nano"
    gpt_5 = "gpt-5"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"


async def validate_and_cast_response(response, timeout: int = None) -> torch.Tensor:
    """Validates and casts a response to a torch.Tensor.

    Args:
        response: The response to validate and cast.
        timeout: Request timeout in seconds. Defaults to None (no timeout).

    Returns:
        A torch.Tensor representing the image (1, H, W, C).

    Raises:
        ValueError: If the response is not valid.
    """
    # validate raw JSON response
    data = response.data
    if not data or len(data) == 0:
        raise ValueError("No images returned from API endpoint")

    # Initialize list to store image tensors
    image_tensors: list[torch.Tensor] = []

    # Process each image in the data array
    for img_data in data:
        if img_data.b64_json:
            img_io = BytesIO(base64.b64decode(img_data.b64_json))
        elif img_data.url:
            img_io = BytesIO()
            await download_url_to_bytesio(img_data.url, img_io, timeout=timeout)
        else:
            raise ValueError("Invalid image payload – neither URL nor base64 data present.")

        pil_img = Image.open(img_io).convert("RGBA")
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        image_tensors.append(torch.from_numpy(arr))

    return torch.stack(image_tensors, dim=0)


class OpenAIDalle2(IO.ComfyNode):
    """
    Generates images synchronously via OpenAI's DALL·E 2 endpoint.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIDalle2",
            display_name="OpenAI DALL·E 2",
            category="api node/image/OpenAI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for DALL·E",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="not implemented yet in backend",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size",
                    default="1024x1024",
                    options=["256x256", "512x512", "1024x1024"],
                    tooltip="Image size",
                    optional=True,
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="How many images to generate",
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image for image editing.",
                    optional=True,
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="Optional mask for inpainting (white areas will be replaced)",
                    optional=True,
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
        prompt,
        seed=0,
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
    ) -> IO.NodeOutput:
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
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img_binary = img_byte_arr  # .getvalue()
            img_binary.name = "image.png"
        elif image is not None or mask is not None:
            raise Exception("Dall-E 2 image editing requires an image AND a mask")

        response = await sync_op(
            cls,
            ApiEndpoint(path=path, method="POST"),
            response_model=OpenAIImageGenerationResponse,
            data=request_class(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                seed=seed,
            ),
            files=(
                {
                    "image": ("image.png", img_binary, "image/png"),
                }
                if img_binary
                else None
            ),
            content_type=content_type,
        )

        return IO.NodeOutput(await validate_and_cast_response(response))


class OpenAIDalle3(IO.ComfyNode):
    """
    Generates images synchronously via OpenAI's DALL·E 3 endpoint.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIDalle3",
            display_name="OpenAI DALL·E 3",
            category="api node/image/OpenAI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for DALL·E",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2 ** 31 - 1,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="not implemented yet in backend",
                    optional=True,
                ),
                IO.Combo.Input(
                    "quality",
                    default="standard",
                    options=["standard", "hd"],
                    tooltip="Image quality",
                    optional=True,
                ),
                IO.Combo.Input(
                    "style",
                    default="natural",
                    options=["natural", "vivid"],
                    tooltip="Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size",
                    default="1024x1024",
                    options=["1024x1024", "1024x1792", "1792x1024"],
                    tooltip="Image size",
                    optional=True,
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
        prompt,
        seed=0,
        style="natural",
        quality="standard",
        size="1024x1024",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        model = "dall-e-3"

        # build the operation
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
            response_model=OpenAIImageGenerationResponse,
            data=OpenAIImageGenerationRequest(
                model=model,
                prompt=prompt,
                quality=quality,
                size=size,
                style=style,
                seed=seed,
            ),
        )

        return IO.NodeOutput(await validate_and_cast_response(response))


class OpenAIGPTImage1(IO.ComfyNode):
    """
    Generates images synchronously via OpenAI's GPT Image 1 endpoint.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImage1",
            display_name="OpenAI GPT Image 1",
            category="api node/image/OpenAI",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for GPT Image 1",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2 ** 31 - 1,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="not implemented yet in backend",
                    optional=True,
                ),
                IO.Combo.Input(
                    "quality",
                    default="low",
                    options=["low", "medium", "high"],
                    tooltip="Image quality, affects cost and generation time.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "background",
                    default="opaque",
                    options=["opaque", "transparent"],
                    tooltip="Return image with or without background",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size",
                    default="auto",
                    options=["auto", "1024x1024", "1024x1536", "1536x1024"],
                    tooltip="Image size",
                    optional=True,
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="How many images to generate",
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image for image editing.",
                    optional=True,
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="Optional mask for inpainting (white areas will be replaced)",
                    optional=True,
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
        prompt,
        seed=0,
        quality="low",
        background="opaque",
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
    ) -> IO.NodeOutput:
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
                img_byte_arr = BytesIO()
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
            mask_img_byte_arr = BytesIO()
            mask_img.save(mask_img_byte_arr, format="PNG")
            mask_img_byte_arr.seek(0)
            files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

        # Build the operation
        response = await sync_op(
            cls,
            ApiEndpoint(path=path, method="POST"),
            response_model=OpenAIImageGenerationResponse,
            data=request_class(
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
        )

        return IO.NodeOutput(await validate_and_cast_response(response))


class OpenAIChatNode(IO.ComfyNode):
    """
    Node to generate text responses from an OpenAI model.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIChatNode",
            display_name="OpenAI ChatGPT",
            category="api node/text/OpenAI",
            description="Generate text responses from an OpenAI model.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text inputs to the model, used to generate a response.",
                ),
                IO.Boolean.Input(
                    "persist_context",
                    default=False,
                    tooltip="This parameter is deprecated and has no effect.",
                ),
                IO.Combo.Input(
                    "model",
                    options=SupportedOpenAIModel,
                    tooltip="The model used to generate the response",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="Optional image(s) to use as context for the model. To include multiple images, you can use the Batch Images node.",
                    optional=True,
                ),
                IO.Custom("OPENAI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. Accepts inputs from the OpenAI Chat Input Files node.",
                ),
                IO.Custom("OPENAI_CHAT_CONFIG").Input(
                    "advanced_options",
                    optional=True,
                    tooltip="Optional configuration for the model. Accepts inputs from the OpenAI Chat Advanced Options node.",
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
    def get_message_content_from_response(
        cls, response: OpenAIResponse
    ) -> list[OutputContent]:
        """Extract message content from the API response."""
        for output in response.output:
            if output.root.type == "message":
                return output.root.content
        raise TypeError("No output message found in response")

    @classmethod
    def get_text_from_message_content(
        cls, message_content: list[OutputContent]
    ) -> str:
        """Extract text content from message content."""
        for content_item in message_content:
            if content_item.root.type == "output_text":
                return str(content_item.root.text)
        return "No text output found in response"

    @classmethod
    def tensor_to_input_image_content(
        cls, image: torch.Tensor, detail_level: Detail = "auto"
    ) -> InputImageContent:
        """Convert a tensor to an input image content object."""
        return InputImageContent(
            detail=detail_level,
            image_url=f"data:image/png;base64,{tensor_to_base64_string(image)}",
            type="input_image",
        )

    @classmethod
    def create_input_message_contents(
        cls,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        files: Optional[list[InputFileContent]] = None,
    ) -> InputMessageContentList:
        """Create a list of input message contents from prompt and optional image."""
        content_list: list[Union[InputContent, InputTextContent, InputImageContent, InputFileContent]] = [
            InputTextContent(text=prompt, type="input_text"),
        ]
        if image is not None:
            for i in range(image.shape[0]):
                content_list.append(
                    InputImageContent(
                        detail="auto",
                        image_url=f"data:image/png;base64,{tensor_to_base64_string(image[i].unsqueeze(0))}",
                        type="input_image",
                    )
                )

        if files is not None:
            content_list.extend(files)

        return InputMessageContentList(
            root=content_list,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        persist_context: bool = False,
        model: SupportedOpenAIModel = SupportedOpenAIModel.gpt_5.value,
        images: Optional[torch.Tensor] = None,
        files: Optional[list[InputFileContent]] = None,
        advanced_options: Optional[CreateModelResponseProperties] = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)

        # Create response
        create_response = await sync_op(
            cls,
            ApiEndpoint(path=RESPONSES_ENDPOINT, method="POST"),
            response_model=OpenAIResponse,
            data=OpenAICreateResponse(
                input=[
                    Item(
                        root=InputMessage(
                            content=cls.create_input_message_contents(
                                prompt, images, files
                            ),
                            role="user",
                        )
                    ),
                ],
                store=True,
                stream=False,
                model=model,
                previous_response_id=None,
                **(
                    advanced_options.model_dump(exclude_none=True)
                    if advanced_options
                    else {}
                ),
            ),
        )
        response_id = create_response.id

        # Get result output
        result_response = await poll_op(
                cls,
                ApiEndpoint(path=f"{RESPONSES_ENDPOINT}/{response_id}"),
                response_model=OpenAIResponse,
                status_extractor=lambda response: response.status,
                completed_statuses=["incomplete", "completed"]
            )
        output_text = cls.get_text_from_message_content(cls.get_message_content_from_response(result_response))

        # Update history
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
        return IO.NodeOutput(output_text)


class OpenAIInputFiles(IO.ComfyNode):
    """
    Loads and formats input files for OpenAI API.
    """

    @classmethod
    def define_schema(cls):
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
        return IO.Schema(
            node_id="OpenAIInputFiles",
            display_name="OpenAI ChatGPT Input Files",
            category="api node/text/OpenAI",
            description="Loads and prepares input files (text, pdf, etc.) to include as inputs for the OpenAI Chat Node. The files will be read by the OpenAI model when generating a response. 🛈 TIP: Can be chained together with other OpenAI Input File nodes.",
            inputs=[
                IO.Combo.Input(
                    "file",
                    options=input_files,
                    default=input_files[0] if input_files else None,
                    tooltip="Input files to include as context for the model. Only accepts text (.txt) and PDF (.pdf) files for now.",
                ),
                IO.Custom("OPENAI_INPUT_FILES").Input(
                    "OPENAI_INPUT_FILES",
                    tooltip="An optional additional file(s) to batch together with the file loaded from this node. Allows chaining of input files so that a single message can include multiple input files.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Custom("OPENAI_INPUT_FILES").Output(),
            ],
        )

    @classmethod
    def create_input_file_content(cls, file_path: str) -> InputFileContent:
        return InputFileContent(
            file_data=text_filepath_to_data_uri(file_path),
            filename=os.path.basename(file_path),
            type="input_file",
        )

    @classmethod
    def execute(cls, file: str, OPENAI_INPUT_FILES: list[InputFileContent] = []) -> IO.NodeOutput:
        """
        Loads and formats input files for OpenAI API.
        """
        file_path = folder_paths.get_annotated_filepath(file)
        input_file_content = cls.create_input_file_content(file_path)
        files = [input_file_content] + OPENAI_INPUT_FILES
        return IO.NodeOutput(files)


class OpenAIChatConfig(IO.ComfyNode):
    """Allows setting additional configuration for the OpenAI Chat Node."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIChatConfig",
            display_name="OpenAI ChatGPT Advanced Options",
            category="api node/text/OpenAI",
            description="Allows specifying advanced configuration options for the OpenAI Chat Nodes.",
            inputs=[
                IO.Combo.Input(
                    "truncation",
                    options=["auto", "disabled"],
                    default="auto",
                    tooltip="The truncation strategy to use for the model response. auto: If the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.disabled: If a model response will exceed the context window size for a model, the request will fail with a 400 error",
                ),
                IO.Int.Input(
                    "max_output_tokens",
                    min=16,
                    default=4096,
                    max=16384,
                    tooltip="An upper bound for the number of tokens that can be generated for a response, including visible output tokens",
                    optional=True,
                ),
                IO.String.Input(
                    "instructions",
                    multiline=True,
                    optional=True,
                    tooltip="Instructions for the model on how to generate the response",
                ),
            ],
            outputs=[
                IO.Custom("OPENAI_CHAT_CONFIG").Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        truncation: bool,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> IO.NodeOutput:
        """
        Configure advanced options for the OpenAI Chat Node.

        Note:
            While `top_p` and `temperature` are listed as properties in the
            spec, they are not supported for all models (e.g., o4-mini).
            They are not exposed as inputs at all to avoid having to manually
            remove depending on model choice.
        """
        return IO.NodeOutput(
            CreateModelResponseProperties(
                instructions=instructions,
                truncation=truncation,
                max_output_tokens=max_output_tokens,
            )
        )


class OpenAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            OpenAIDalle2,
            OpenAIDalle3,
            OpenAIGPTImage1,
            OpenAIChatNode,
            OpenAIInputFiles,
            OpenAIChatConfig,
        ]


async def comfy_entrypoint() -> OpenAIExtension:
    return OpenAIExtension()
