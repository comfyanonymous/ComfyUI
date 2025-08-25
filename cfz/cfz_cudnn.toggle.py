import torch

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False

anyType = AnyType("*")

class CUDNNToggleAutoPassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "audio": ("AUDIO",),
                "image": ("IMAGE",),
                "wan_model": ("WANVIDEOMODEL",),
                "any_input": (anyType, {}),
            },
            "required": {
                "enable_cudnn": ("BOOLEAN", {"default": True}),
                "cudnn_benchmark": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "AUDIO", "IMAGE", "WANVIDEOMODEL", anyType, "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("model", "conditioning", "latent", "audio", "image", "wan_model", "any_output", "prev_cudnn", "prev_benchmark")
    FUNCTION = "toggle"
    CATEGORY = "utils"

    def toggle(self, enable_cudnn, cudnn_benchmark, any_input=None, wan_model=None, model=None, conditioning=None, latent=None, audio=None, image=None):
        prev_cudnn = torch.backends.cudnn.enabled
        prev_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.enabled = enable_cudnn
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if enable_cudnn != prev_cudnn:
            print(f"[CUDNN_TOGGLE] torch.backends.cudnn.enabled set to {enable_cudnn} (was {prev_cudnn})")
        else:
            print(f"[CUDNN_TOGGLE] torch.backends.cudnn.enabled still set to {enable_cudnn}")

        if cudnn_benchmark != prev_benchmark:
            print(f"[CUDNN_TOGGLE] torch.backends.cudnn.benchmark set to {cudnn_benchmark} (was {prev_benchmark})")
        else:
            print(f"[CUDNN_TOGGLE] torch.backends.cudnn.benchmark still set to {cudnn_benchmark}")

        return_tuple = (model, conditioning, latent, audio, image, wan_model, any_input, prev_cudnn, prev_benchmark)
        return return_tuple

NODE_CLASS_MAPPINGS = {
    "CUDNNToggleAutoPassthrough": CUDNNToggleAutoPassthrough
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CUDNNToggleAutoPassthrough": "CFZ CUDNN Toggle"
}
