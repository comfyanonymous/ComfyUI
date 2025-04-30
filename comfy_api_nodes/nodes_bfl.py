import io
from inspect import cleandoc
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy_api_nodes.apis.bfl_api import (
    BFLStatus,
    BFLFluxProGenerateRequest,
    BFLFluxProGenerateResponse,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    downscale_image_tensor,
    validate_aspect_ratio,
    process_image_response,
)

import numpy as np
from PIL import Image
import requests
import torch
import base64
import time


class FluxProUltraImageNode(ComfyNodeABC):
    """
    Generates images synchronously based on prompt and resolution.
    """

    MINIMUM_RATIO = 1 / 4
    MAXIMUM_RATIO = 4 / 1
    MINIMUM_RATIO_STR = "1:4"
    MAXIMUM_RATIO_STR = "4:1"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation",
                    },
                ),
                "prompt_upsampling": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "aspect_ratio": (
                    IO.STRING,
                    {
                        "default": "16:9",
                        "tooltip": "Aspect ratio of image; must be between 1:4 and 4:1.",
                    },
                ),
                "raw": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "When True, generate less processed, more natural-looking images.",
                    },
                ),
            },
            "optional": {
                "image_prompt": (IO.IMAGE,),
                "image_prompt_strength": (
                    IO.FLOAT,
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend between the prompt and the image prompt.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, aspect_ratio: str):
        try:
            validate_aspect_ratio(
                aspect_ratio,
                minimum_ratio=cls.MINIMUM_RATIO,
                maximum_ratio=cls.MAXIMUM_RATIO,
                minimum_ratio_str=cls.MINIMUM_RATIO_STR,
                maximum_ratio_str=cls.MAXIMUM_RATIO_STR,
            )
        except Exception as e:
            return str(e)
        return True

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/bfl"

    def api_call(
        self,
        prompt: str,
        aspect_ratio: str,
        prompt_upsampling=False,
        raw=False,
        seed=0,
        image_prompt=None,
        image_prompt_strength=0.1,
        auth_token=None,
        **kwargs,
    ):
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.1-ultra/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxProGenerateRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                seed=seed,
                aspect_ratio=validate_aspect_ratio(
                    aspect_ratio,
                    minimum_ratio=self.MINIMUM_RATIO,
                    maximum_ratio=self.MAXIMUM_RATIO,
                    minimum_ratio_str=self.MINIMUM_RATIO_STR,
                    maximum_ratio_str=self.MAXIMUM_RATIO_STR,
                ),
                raw=raw,
                image_prompt=(
                    image_prompt
                    if image_prompt is None
                    else self._convert_image_to_base64(image_prompt)
                ),
                image_prompt_strength=(
                    None if image_prompt is None else round(image_prompt_strength, 2)
                ),
            ),
            auth_token=auth_token,
        )
        output_image = self._handle_bfl_synchronous_operation(operation)
        return (output_image,)

    def _handle_bfl_synchronous_operation(
        self, operation: SynchronousOperation, timeout_bfl_calls=360
    ):
        response_api: BFLFluxProGenerateResponse = operation.execute()
        return self._poll_until_generated(
            response_api.polling_url, timeout=timeout_bfl_calls
        )

    def _poll_until_generated(self, polling_url: str, timeout=360):
        # used bfl-comfy-nodes to verify code implementation:
        # https://github.com/black-forest-labs/bfl-comfy-nodes/tree/main
        start_time = time.time()
        retries_404 = 0
        max_retries_404 = 5
        retry_404_seconds = 2
        retry_202_seconds = 2
        retry_pending_seconds = 1
        request = requests.Request(method=HttpMethod.GET, url=polling_url)
        # NOTE: should True loop be replaced with checking if workflow has been interrupted?
        while True:
            response = requests.Session().send(request.prepare())
            if response.status_code == 200:
                result = response.json()
                if result["status"] == BFLStatus.ready:
                    img_url = result["result"]["sample"]
                    img_response = requests.get(img_url)
                    return process_image_response(img_response)
                elif result["status"] in [
                    BFLStatus.request_moderated,
                    BFLStatus.content_moderated,
                ]:
                    status = result["status"]
                    raise Exception(
                        f"BFL API did not return an image due to: {status}."
                    )
                elif result["status"] == BFLStatus.error:
                    raise Exception(f"BFL API encountered an error: {result}.")
                elif result["status"] == BFLStatus.pending:
                    time.sleep(retry_pending_seconds)
                    continue
            elif response.status_code == 404:
                if retries_404 < max_retries_404:
                    retries_404 += 1
                    time.sleep(retry_404_seconds)
                    continue
                raise Exception(
                    f"BFL API could not find task after {max_retries_404} tries."
                )
            elif response.status_code == 202:
                time.sleep(retry_202_seconds)
            elif time.time() - start_time > timeout:
                raise Exception(
                    f"BFL API experienced a timeout; could not return request under {timeout} seconds."
                )
            else:
                raise Exception(f"BFL API encountered an error: {response.json()}")

    def _convert_image_to_base64(self, image: torch.Tensor):
        scaled_image = downscale_image_tensor(image, total_pixels=2048 * 2048)
        # remove batch dimension if present
        if len(scaled_image.shape) > 3:
            scaled_image = scaled_image[0]
        image_np = (scaled_image.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        return base64.b64encode(img_byte_arr.getvalue()).decode()


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FluxProUltraImageNode": FluxProUltraImageNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxProUltraImageNode": "Flux 1.1 [pro] Ultra Image",
}
