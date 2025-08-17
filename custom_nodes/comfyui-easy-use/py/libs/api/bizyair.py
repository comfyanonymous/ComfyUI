import yaml
import pathlib
import base64
import io
import json
import os
import pickle
import zlib
import urllib.parse
import urllib.request
import urllib.error
from enum import Enum
from functools import singledispatch
from typing import Any, List, Union

import numpy as np
import torch
from PIL import Image

root_path = pathlib.Path(__file__).parent.parent.parent.parent
config_path = os.path.join(root_path, 'config.yaml')

class BizyAIRAPI:
    def __init__(self):
        self.base_url = 'https://bizyair-api.siliconflow.cn/x/v1'
        self.api_key = None


    def getAPIKey(self):
        if self.api_key is None:
            if os.path.isfile(config_path):
                with open(config_path, 'r') as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                    if 'BIZYAIR_API_KEY' not in data:
                        raise Exception("Please add BIZYAIR_API_KEY to config.yaml")
                    self.api_key = data['BIZYAIR_API_KEY']
            else:
                raise Exception("Please add config.yaml to root path")
        return self.api_key

    def send_post_request(self, url, payload, headers):
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req) as response:
                response_data = response.read().decode("utf-8")
            return response_data
        except urllib.error.URLError as e:
            if "Unauthorized" in str(e):
                raise Exception(
                    "Key is invalid, please refer to https://cloud.siliconflow.cn to get the API key.\n"
                    "If you have the key, please click the 'BizyAir Key' button at the bottom right to set the key."
                )
            else:
                raise Exception(
                    f"Failed to connect to the server: {e}, if you have no key, "
                )

    # joycaption
    def joyCaption(self, payload, image, apikey_override=None, API_URL='/supernode/joycaption2'):
        if apikey_override is not None:
            api_key = apikey_override
        else:
            api_key = self.getAPIKey()
        url = f"{self.base_url}{API_URL}"
        print('Sending request to:', url)
        auth = f"Bearer {api_key}"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": auth,
        }
        input_image = encode_data(image, disable_image_marker=True)
        payload["image"] = input_image

        ret: str = self.send_post_request(url=url, payload=payload, headers=headers)
        ret = json.loads(ret)

        try:
            if "result" in ret:
                ret = json.loads(ret["result"])
        except Exception as e:
            raise Exception(f"Unexpected response: {ret} {e=}")

        if ret["type"] == "error":
            raise Exception(ret["message"])

        msg = ret["data"]
        if msg["type"] not in ("comfyair", "bizyair",):
            raise Exception(f"Unexpected response type: {msg}")

        caption = msg["data"]

        return caption

bizyairAPI = BizyAIRAPI()



BIZYAIR_DEBUG = True
# Marker to identify base64-encoded tensors
TENSOR_MARKER = "TENSOR:"
IMAGE_MARKER = "IMAGE:"


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


def convert_image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def encode_image_to_base64(
    image: Image.Image, format: str = "png", quality: int = 100, lossless=False
) -> str:
    image = convert_image_to_rgb(image)
    with io.BytesIO() as output:
        image.save(output, format=format, quality=quality, lossless=lossless)
        output.seek(0)
        img_bytes = output.getvalue()
        if BIZYAIR_DEBUG:
            print(f"encode_image_to_base64: {format_bytes(len(img_bytes))}")
    return base64.b64encode(img_bytes).decode("utf-8")


def decode_base64_to_np(img_data: str, format: str = "png") -> np.ndarray:
    img_bytes = base64.b64decode(img_data)
    if BIZYAIR_DEBUG:
        print(f"decode_base64_to_np: {format_bytes(len(img_bytes))}")
    with io.BytesIO(img_bytes) as input_buffer:
        img = Image.open(input_buffer)
        # https://github.com/comfyanonymous/ComfyUI/blob/a178e25912b01abf436eba1cfaab316ba02d272d/nodes.py#L1511
        img = img.convert("RGB")
        return np.array(img)


def decode_base64_to_image(img_data: str) -> Image.Image:
    img_bytes = base64.b64decode(img_data)
    with io.BytesIO(img_bytes) as input_buffer:
        img = Image.open(input_buffer)
        if BIZYAIR_DEBUG:
            format_info = img.format.upper() if img.format else "Unknown"
            print(f"decode image format: {format_info}")
        return img


def format_bytes(num_bytes: int) -> str:
    """
    Converts a number of bytes to a human-readable string with units (B, KB, or MB).

    :param num_bytes: The number of bytes to convert.
    :return: A string representing the number of bytes in a human-readable format.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


def _legacy_encode_comfy_image(image: torch.Tensor, image_format="png") -> str:
    input_image = image.cpu().detach().numpy()
    i = 255.0 * input_image[0]
    input_image = np.clip(i, 0, 255).astype(np.uint8)
    base64ed_image = encode_image_to_base64(
        Image.fromarray(input_image), format=image_format
    )
    return base64ed_image


def _legacy_decode_comfy_image(
    img_data: Union[List, str], image_format="png"
) -> torch.tensor:
    if isinstance(img_data, List):
        decoded_imgs = [decode_comfy_image(x, old_version=True) for x in img_data]

        combined_imgs = torch.cat(decoded_imgs, dim=0)
        return combined_imgs

    out = decode_base64_to_np(img_data, format=image_format)
    out = np.array(out).astype(np.float32) / 255.0
    output = torch.from_numpy(out)[None,]
    return output


def _new_encode_comfy_image(images: torch.Tensor, image_format="WEBP", **kwargs) -> str:
    """https://docs.comfy.org/essentials/custom_node_snippets#save-an-image-batch
    Encode a batch of images to base64 strings.

    Args:
        images (torch.Tensor): A batch of images.
        image_format (str, optional): The format of the images. Defaults to "WEBP".

    Returns:
        str: A JSON string containing the base64-encoded images.
    """
    results = {}
    for batch_number, image in enumerate(images):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        base64ed_image = encode_image_to_base64(img, format=image_format, **kwargs)
        results[batch_number] = base64ed_image

    return json.dumps(results)


def _new_decode_comfy_image(img_datas: str, image_format="WEBP") -> torch.tensor:
    """
    Decode a batch of base64-encoded images.

    Args:
        img_datas (str): A JSON string containing the base64-encoded images.
        image_format (str, optional): The format of the images. Defaults to "WEBP".

    Returns:
        torch.Tensor: A tensor containing the decoded images.
    """
    img_datas = json.loads(img_datas)

    decoded_imgs = []
    for img_data in img_datas.values():
        decoded_image = decode_base64_to_np(img_data, format=image_format)
        decoded_image = np.array(decoded_image).astype(np.float32) / 255.0
        decoded_imgs.append(torch.from_numpy(decoded_image)[None,])

    return torch.cat(decoded_imgs, dim=0)


def encode_comfy_image(
    image: torch.Tensor, image_format="WEBP", old_version=False, lossless=False
) -> str:
    if old_version:
        return _legacy_encode_comfy_image(image, image_format)
    return _new_encode_comfy_image(image, image_format, lossless=lossless)


def decode_comfy_image(
    img_data: Union[List, str], image_format="WEBP", old_version=False
) -> torch.tensor:
    if old_version:
        return _legacy_decode_comfy_image(img_data, image_format)
    return _new_decode_comfy_image(img_data, image_format)


def tensor_to_base64(tensor: torch.Tensor, compress=True) -> str:
    tensor_np = tensor.cpu().detach().numpy()

    tensor_bytes = pickle.dumps(tensor_np)
    if compress:
        tensor_bytes = zlib.compress(tensor_bytes)

    tensor_b64 = base64.b64encode(tensor_bytes).decode("utf-8")
    return tensor_b64


def base64_to_tensor(tensor_b64: str, compress=True) -> torch.Tensor:
    tensor_bytes = base64.b64decode(tensor_b64)

    if compress:
        tensor_bytes = zlib.decompress(tensor_bytes)

    tensor_np = pickle.loads(tensor_bytes)

    tensor = torch.from_numpy(tensor_np)
    return tensor


@singledispatch
def decode_data(input, old_version=False):
    raise NotImplementedError(f"Unsupported type: {type(input)}")


@decode_data.register(int)
@decode_data.register(float)
@decode_data.register(bool)
@decode_data.register(type(None))
def _(input, **kwargs):
    return input


@decode_data.register(dict)
def _(input, **kwargs):
    return {k: decode_data(v, **kwargs) for k, v in input.items()}


@decode_data.register(list)
def _(input, **kwargs):
    return [decode_data(x, **kwargs) for x in input]


@decode_data.register(str)
def _(input: str, **kwargs):
    if input.startswith(TENSOR_MARKER):
        tensor_b64 = input[len(TENSOR_MARKER) :]
        return base64_to_tensor(tensor_b64)
    elif input.startswith(IMAGE_MARKER):
        tensor_b64 = input[len(IMAGE_MARKER) :]
        old_version = kwargs.get("old_version", False)
        return decode_comfy_image(tensor_b64, old_version=old_version)
    return input


@singledispatch
def encode_data(output, disable_image_marker=False, old_version=False):
    raise NotImplementedError(f"Unsupported type: {type(output)}")


@encode_data.register(dict)
def _(output, **kwargs):
    return {k: encode_data(v, **kwargs) for k, v in output.items()}


@encode_data.register(list)
def _(output, **kwargs):
    return [encode_data(x, **kwargs) for x in output]


def is_image_tensor(tensor) -> bool:
    """https://docs.comfy.org/essentials/custom_node_datatypes#image

    Check if the given tensor is in the format of an IMAGE (shape [B, H, W, C] where C=3).

    `Args`:
        tensor (torch.Tensor): The tensor to check.

    `Returns`:
        bool: True if the tensor is in the IMAGE format, False otherwise.
    """
    try:
        if not isinstance(tensor, torch.Tensor):
            return False

        if len(tensor.shape) != 4:
            return False

        B, H, W, C = tensor.shape
        if C != 3:
            return False

        return True
    except:
        return False


@encode_data.register(torch.Tensor)
def _(output, **kwargs):
    if is_image_tensor(output) and not kwargs.get("disable_image_marker", False):
        old_version = kwargs.get("old_version", False)
        lossless = kwargs.get("lossless", True)
        return IMAGE_MARKER + encode_comfy_image(
            output, image_format="WEBP", old_version=old_version, lossless=lossless
        )
    return TENSOR_MARKER + tensor_to_base64(output)


@encode_data.register(int)
@encode_data.register(float)
@encode_data.register(bool)
@encode_data.register(type(None))
def _(output, **kwargs):
    return output


@encode_data.register(str)
def _(output, **kwargs):
    return output
