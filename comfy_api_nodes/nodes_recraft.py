from __future__ import annotations
from inspect import cleandoc
from typing import Optional
from comfy.utils import ProgressBar
from comfy_extras.nodes_images import SVG # Added
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apis.recraft_api import (
    RecraftImageGenerationRequest,
    RecraftImageGenerationResponse,
    RecraftImageSize,
    RecraftModel,
    RecraftStyle,
    RecraftStyleV3,
    RecraftColor,
    RecraftColorChain,
    RecraftControls,
    RecraftIO,
    get_v3_substyles,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    bytesio_to_image_tensor,
    download_url_to_bytesio,
    tensor_to_bytesio,
    resize_mask_to_image,
    validate_string,
)
from server import PromptServer

import torch
from io import BytesIO
from PIL import UnidentifiedImageError


async def handle_recraft_file_request(
        image: torch.Tensor,
        path: str,
        mask: torch.Tensor=None,
        total_pixels=4096*4096,
        timeout=1024,
        request=None,
        auth_kwargs: dict[str,str] = None,
    ) -> list[BytesIO]:
        """
        Handle sending common Recraft file-only request to get back file bytes.
        """
        if request is None:
            request = EmptyRequest()

        files = {
            'image': tensor_to_bytesio(image, total_pixels=total_pixels).read()
        }
        if mask is not None:
            files['mask'] = tensor_to_bytesio(mask, total_pixels=total_pixels).read()

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=type(request),
                response_model=RecraftImageGenerationResponse,
            ),
            request=request,
            files=files,
            content_type="multipart/form-data",
            auth_kwargs=auth_kwargs,
            multipart_parser=recraft_multipart_parser,
        )
        response: RecraftImageGenerationResponse = await operation.execute()
        all_bytesio = []
        if response.image is not None:
            all_bytesio.append(await download_url_to_bytesio(response.image.url, timeout=timeout))
        else:
            for data in response.data:
                all_bytesio.append(await download_url_to_bytesio(data.url, timeout=timeout))

        return all_bytesio


def recraft_multipart_parser(data, parent_key=None, formatter: callable=None, converted_to_check: list[list]=None, is_list=False) -> dict:
    """
    Formats data such that multipart/form-data will work with requests library
    when both files and data are present.

    The OpenAI client that Recraft uses has a bizarre way of serializing lists:

    It does NOT keep track of indeces of each list, so for background_color, that must be serialized as:
        'background_color[rgb][]' = [0, 0, 255]
    where the array is assigned to a key that has '[]' at the end, to signal it's an array.

    This has the consequence of nested lists having the exact same key, forcing arrays to merge; all colors inputs fall under the same key:
        if 1 color  -> 'controls[colors][][rgb][]' = [0, 0, 255]
        if 2 colors -> 'controls[colors][][rgb][]' = [0, 0, 255, 255, 0, 0]
        if 3 colors -> 'controls[colors][][rgb][]' = [0, 0, 255, 255, 0, 0, 0, 255, 0]
        etc.
    Whoever made this serialization up at OpenAI added the constraint that lists must be of uniform length on objects of same 'type'.
    """
    # Modification of a function that handled a different type of multipart parsing, big ups:
    # https://gist.github.com/kazqvaizer/4cebebe5db654a414132809f9f88067b

    def handle_converted_lists(data, parent_key, lists_to_check=tuple[list]):
        # if list already exists exists, just extend list with data
        for check_list in lists_to_check:
            for conv_tuple in check_list:
                if conv_tuple[0] == parent_key and type(conv_tuple[1]) is list:
                    conv_tuple[1].append(formatter(data))
                    return True
        return False

    if converted_to_check is None:
        converted_to_check = []


    if formatter is None:
        formatter = lambda v: v  # Multipart representation of value

    if type(data) is not dict:
        # if list already exists exists, just extend list with data
        added = handle_converted_lists(data, parent_key, converted_to_check)
        if added:
            return {}
        # otherwise if is_list, create new list with data
        if is_list:
            return {parent_key: [formatter(data)]}
        # return new key with data
        return {parent_key: formatter(data)}

    converted = []
    next_check = [converted]
    next_check.extend(converted_to_check)

    for key, value in data.items():
        current_key = key if parent_key is None else f"{parent_key}[{key}]"
        if type(value) is dict:
            converted.extend(recraft_multipart_parser(value, current_key, formatter, next_check).items())
        elif type(value) is list:
            for ind, list_value in enumerate(value):
                iter_key = f"{current_key}[]"
                converted.extend(recraft_multipart_parser(list_value, iter_key, formatter, next_check, is_list=True).items())
        else:
            converted.append((current_key, formatter(value)))

    return dict(converted)


class handle_recraft_image_output:
    """
    Catch an exception related to receiving SVG data instead of image, when Infinite Style Library style_id is in use.
    """
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and exc_type is UnidentifiedImageError:
            raise Exception("Received output data was not an image; likely an SVG. If you used style_id, make sure it is not a Vector art style.")


class RecraftColorRGBNode:
    """
    Create Recraft Color by choosing specific RGB values.
    """

    RETURN_TYPES = (RecraftIO.COLOR,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    RETURN_NAMES = ("recraft_color",)
    FUNCTION = "create_color"
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "r": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Red value of color."
                }),
                "g": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Green value of color."
                }),
                "b": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Blue value of color."
                }),
            },
            "optional": {
                "recraft_color": (RecraftIO.COLOR,),
            }
        }

    def create_color(self, r: int, g: int, b: int, recraft_color: RecraftColorChain=None):
        recraft_color = recraft_color.clone() if recraft_color else RecraftColorChain()
        recraft_color.add(RecraftColor(r, g, b))
        return (recraft_color, )


class RecraftControlsNode:
    """
    Create Recraft Controls for customizing Recraft generation.
    """

    RETURN_TYPES = (RecraftIO.CONTROLS,)
    RETURN_NAMES = ("recraft_controls",)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "create_controls"
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "colors": (RecraftIO.COLOR,),
                "background_color": (RecraftIO.COLOR,),
            }
        }

    def create_controls(self, colors: RecraftColorChain=None, background_color: RecraftColorChain=None):
        return (RecraftControls(colors=colors, background_color=background_color), )


class RecraftStyleV3RealisticImageNode:
    """
    Select realistic_image style and optional substyle.
    """

    RETURN_TYPES = (RecraftIO.STYLEV3,)
    RETURN_NAMES = ("recraft_style",)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "create_style"
    CATEGORY = "api node/image/Recraft"

    RECRAFT_STYLE = RecraftStyleV3.realistic_image

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "substyle": (get_v3_substyles(s.RECRAFT_STYLE),),
            }
        }

    def create_style(self, substyle: str):
        if substyle == "None":
            substyle = None
        return (RecraftStyle(self.RECRAFT_STYLE, substyle),)


class RecraftStyleV3DigitalIllustrationNode(RecraftStyleV3RealisticImageNode):
    """
    Select digital_illustration style and optional substyle.
    """

    RECRAFT_STYLE = RecraftStyleV3.digital_illustration


class RecraftStyleV3VectorIllustrationNode(RecraftStyleV3RealisticImageNode):
    """
    Select vector_illustration style and optional substyle.
    """

    RECRAFT_STYLE = RecraftStyleV3.vector_illustration


class RecraftStyleV3LogoRasterNode(RecraftStyleV3RealisticImageNode):
    """
    Select vector_illustration style and optional substyle.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "substyle": (get_v3_substyles(s.RECRAFT_STYLE, include_none=False),),
            }
        }

    RECRAFT_STYLE = RecraftStyleV3.logo_raster


class RecraftStyleInfiniteStyleLibrary:
    """
    Select style based on preexisting UUID from Recraft's Infinite Style Library.
    """

    RETURN_TYPES = (RecraftIO.STYLEV3,)
    RETURN_NAMES = ("recraft_style",)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "create_style"
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_id": (IO.STRING, {
                    "default": "",
                    "tooltip": "UUID of style from Infinite Style Library.",
                })
            }
        }

    def create_style(self, style_id: str):
        if not style_id:
            raise Exception("The style_id input cannot be empty.")
        return (RecraftStyle(style_id=style_id),)


class RecraftTextToImageNode:
    """
    Generates images synchronously based on prompt and resolution.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation.",
                    },
                ),
                "size": (
                    [res.value for res in RecraftImageSize],
                    {
                        "default": RecraftImageSize.res_1024x1024,
                        "tooltip": "The size of the generated image.",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "The number of images to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "recraft_style": (RecraftIO.STYLEV3,),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
                "recraft_controls": (
                    RecraftIO.CONTROLS,
                    {
                        "tooltip": "Optional additional controls over the generation via the Recraft Controls node."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        size: str,
        n: int,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
        recraft_controls: RecraftControls = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False, max_length=1000)
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        controls_api = None
        if recraft_controls:
            controls_api = recraft_controls.create_api_model()

        if not negative_prompt:
            negative_prompt = None

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/recraft/image_generation",
                method=HttpMethod.POST,
                request_model=RecraftImageGenerationRequest,
                response_model=RecraftImageGenerationResponse,
            ),
            request=RecraftImageGenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=RecraftModel.recraftv3,
                size=size,
                n=n,
                style=recraft_style.style,
                substyle=recraft_style.substyle,
                style_id=recraft_style.style_id,
                controls=controls_api,
            ),
            auth_kwargs=kwargs,
        )
        response: RecraftImageGenerationResponse = await operation.execute()
        images = []
        urls = []
        for data in response.data:
            with handle_recraft_image_output():
                if unique_id and data.url:
                    urls.append(data.url)
                    urls_string = '\n'.join(urls)
                    PromptServer.instance.send_progress_text(
                        f"Result URL: {urls_string}", unique_id
                    )
                image = bytesio_to_image_tensor(
                    await download_url_to_bytesio(data.url, timeout=1024)
                )
            if len(image.shape) < 4:
                image = image.unsqueeze(0)
            images.append(image)
        output_image = torch.cat(images, dim=0)

        return (output_image,)


class RecraftImageToImageNode:
    """
    Modify image based on prompt and strength.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation.",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "The number of images to generate.",
                    },
                ),
                "strength": (
                    IO.FLOAT,
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Defines the difference with the original image, should lie in [0, 1], where 0 means almost identical, and 1 means miserable similarity."
                    }
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "recraft_style": (RecraftIO.STYLEV3,),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
                "recraft_controls": (
                    RecraftIO.CONTROLS,
                    {
                        "tooltip": "Optional additional controls over the generation via the Recraft Controls node."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
        prompt: str,
        n: int,
        strength: float,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
        recraft_controls: RecraftControls = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False, max_length=1000)
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        controls_api = None
        if recraft_controls:
            controls_api = recraft_controls.create_api_model()

        if not negative_prompt:
            negative_prompt = None

        request = RecraftImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=RecraftModel.recraftv3,
            n=n,
            strength=round(strength, 2),
            style=recraft_style.style,
            substyle=recraft_style.substyle,
            style_id=recraft_style.style_id,
            controls=controls_api,
        )

        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                image=image[i],
                path="/proxy/recraft/images/imageToImage",
                request=request,
                auth_kwargs=kwargs,
            )
            with handle_recraft_image_output():
                images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        images_tensor = torch.cat(images, dim=0)
        return (images_tensor, )


class RecraftImageInpaintingNode:
    """
    Modify image based on prompt and mask.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, ),
                "mask": (IO.MASK, ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation.",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "The number of images to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "recraft_style": (RecraftIO.STYLEV3,),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        n: int,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False, max_length=1000)
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        if not negative_prompt:
            negative_prompt = None

        request = RecraftImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=RecraftModel.recraftv3,
            n=n,
            style=recraft_style.style,
            substyle=recraft_style.substyle,
            style_id=recraft_style.style_id,
        )

        # prepare mask tensor
        mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)

        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                image=image[i],
                mask=mask[i:i+1],
                path="/proxy/recraft/images/inpaint",
                request=request,
                auth_kwargs=kwargs,
            )
            with handle_recraft_image_output():
                images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        images_tensor = torch.cat(images, dim=0)
        return (images_tensor, )


class RecraftTextToVectorNode:
    """
    Generates SVG synchronously based on prompt and resolution.
    """

    RETURN_TYPES = ("SVG",) # Changed
    DESCRIPTION = cleandoc(__doc__ or "") if 'cleandoc' in globals() else __doc__ # Keep cleandoc if other nodes use it
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation.",
                    },
                ),
                "substyle": (get_v3_substyles(RecraftStyleV3.vector_illustration),),
                "size": (
                    [res.value for res in RecraftImageSize],
                    {
                        "default": RecraftImageSize.res_1024x1024,
                        "tooltip": "The size of the generated image.",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "The number of images to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
                "recraft_controls": (
                    RecraftIO.CONTROLS,
                    {
                        "tooltip": "Optional additional controls over the generation via the Recraft Controls node."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        substyle: str,
        size: str,
        n: int,
        seed,
        negative_prompt: str = None,
        recraft_controls: RecraftControls = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False, max_length=1000)
        # create RecraftStyle so strings will be formatted properly (i.e. "None" will become None)
        recraft_style = RecraftStyle(RecraftStyleV3.vector_illustration, substyle=substyle)

        controls_api = None
        if recraft_controls:
            controls_api = recraft_controls.create_api_model()

        if not negative_prompt:
            negative_prompt = None

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/recraft/image_generation",
                method=HttpMethod.POST,
                request_model=RecraftImageGenerationRequest,
                response_model=RecraftImageGenerationResponse,
            ),
            request=RecraftImageGenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=RecraftModel.recraftv3,
                size=size,
                n=n,
                style=recraft_style.style,
                substyle=recraft_style.substyle,
                controls=controls_api,
            ),
            auth_kwargs=kwargs,
        )
        response: RecraftImageGenerationResponse = await operation.execute()
        svg_data = []
        urls = []
        for data in response.data:
            if unique_id and data.url:
                urls.append(data.url)
                # Print result on each iteration in case of error
                PromptServer.instance.send_progress_text(
                    f"Result URL: {' '.join(urls)}", unique_id
                )
            svg_data.append(await download_url_to_bytesio(data.url, timeout=1024))

        return (SVG(svg_data),)


class RecraftVectorizeImageNode:
    """
    Generates SVG synchronously from an input image.
    """

    RETURN_TYPES = ("SVG",) # Changed
    DESCRIPTION = cleandoc(__doc__ or "") if 'cleandoc' in globals() else __doc__ # Keep cleandoc if other nodes use it
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, ),
            },
            "optional": {
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
        **kwargs,
    ):
        svgs = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                image=image[i],
                path="/proxy/recraft/images/vectorize",
                auth_kwargs=kwargs,
            )
            svgs.append(SVG(sub_bytes))
            pbar.update(1)

        return (SVG.combine_all(svgs), )


class RecraftReplaceBackgroundNode:
    """
    Replace background on image, based on provided prompt.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation.",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "The number of images to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "recraft_style": (RecraftIO.STYLEV3,),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "An optional text description of undesired elements on an image.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
        prompt: str,
        n: int,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
        **kwargs,
    ):
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        if not negative_prompt:
            negative_prompt = None

        request = RecraftImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=RecraftModel.recraftv3,
            n=n,
            style=recraft_style.style,
            substyle=recraft_style.substyle,
            style_id=recraft_style.style_id,
        )

        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                image=image[i],
                path="/proxy/recraft/images/replaceBackground",
                request=request,
                auth_kwargs=kwargs,
            )
            images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        images_tensor = torch.cat(images, dim=0)
        return (images_tensor, )


class RecraftRemoveBackgroundNode:
    """
    Remove background from image, and return processed image and mask.
    """

    RETURN_TYPES = (IO.IMAGE, IO.MASK)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, ),
            },
            "optional": {
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
        **kwargs,
    ):
        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                image=image[i],
                path="/proxy/recraft/images/removeBackground",
                auth_kwargs=kwargs,
            )
            images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        images_tensor = torch.cat(images, dim=0)
        # use alpha channel as masks, in B,H,W format
        masks_tensor = images_tensor[:,:,:,-1:].squeeze(-1)
        return (images_tensor, masks_tensor)


class RecraftCrispUpscaleNode:
    """
    Upscale image synchronously.
    Enhances a given raster image using ‘crisp upscale’ tool, increasing image resolution, making the image sharper and cleaner.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    RECRAFT_PATH = "/proxy/recraft/images/crispUpscale"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, ),
            },
            "optional": {
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
        **kwargs,
    ):
        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                image=image[i],
                path=self.RECRAFT_PATH,
                auth_kwargs=kwargs,
            )
            images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        images_tensor = torch.cat(images, dim=0)
        return (images_tensor,)


class RecraftCreativeUpscaleNode(RecraftCrispUpscaleNode):
    """
    Upscale image synchronously.
    Enhances a given raster image using ‘creative upscale’ tool, boosting resolution with a focus on refining small details and faces.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Recraft"

    RECRAFT_PATH = "/proxy/recraft/images/creativeUpscale"


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "RecraftTextToImageNode": RecraftTextToImageNode,
    "RecraftImageToImageNode": RecraftImageToImageNode,
    "RecraftImageInpaintingNode": RecraftImageInpaintingNode,
    "RecraftTextToVectorNode": RecraftTextToVectorNode,
    "RecraftVectorizeImageNode": RecraftVectorizeImageNode,
    "RecraftRemoveBackgroundNode": RecraftRemoveBackgroundNode,
    "RecraftReplaceBackgroundNode": RecraftReplaceBackgroundNode,
    "RecraftCrispUpscaleNode": RecraftCrispUpscaleNode,
    "RecraftCreativeUpscaleNode": RecraftCreativeUpscaleNode,
    "RecraftStyleV3RealisticImage": RecraftStyleV3RealisticImageNode,
    "RecraftStyleV3DigitalIllustration": RecraftStyleV3DigitalIllustrationNode,
    "RecraftStyleV3LogoRaster": RecraftStyleV3LogoRasterNode,
    "RecraftStyleV3InfiniteStyleLibrary": RecraftStyleInfiniteStyleLibrary,
    "RecraftColorRGB": RecraftColorRGBNode,
    "RecraftControls": RecraftControlsNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RecraftTextToImageNode": "Recraft Text to Image",
    "RecraftImageToImageNode": "Recraft Image to Image",
    "RecraftImageInpaintingNode": "Recraft Image Inpainting",
    "RecraftTextToVectorNode": "Recraft Text to Vector",
    "RecraftVectorizeImageNode": "Recraft Vectorize Image",
    "RecraftRemoveBackgroundNode": "Recraft Remove Background",
    "RecraftReplaceBackgroundNode": "Recraft Replace Background",
    "RecraftCrispUpscaleNode": "Recraft Crisp Upscale Image",
    "RecraftCreativeUpscaleNode": "Recraft Creative Upscale Image",
    "RecraftStyleV3RealisticImage": "Recraft Style - Realistic Image",
    "RecraftStyleV3DigitalIllustration": "Recraft Style - Digital Illustration",
    "RecraftStyleV3LogoRaster": "Recraft Style - Logo Raster",
    "RecraftStyleV3InfiniteStyleLibrary": "Recraft Style - Infinite Style Library",
    "RecraftColorRGB": "Recraft Color RGB",
    "RecraftControls": "Recraft Controls",
}
