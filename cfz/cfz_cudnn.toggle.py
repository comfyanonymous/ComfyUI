import torch

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
            },
            "required": {
                "enable_cudnn": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "AUDIO", "IMAGE")
    RETURN_NAMES = ("model", "conditioning", "latent", "audio", "image")
    FUNCTION = "toggle"
    CATEGORY = "utils"

    def toggle(self, enable_cudnn, model=None, conditioning=None, latent=None, audio=None, image=None):
        torch.backends.cudnn.enabled = enable_cudnn
        print(f"[CUDNN_TOGGLE] torch.backends.cudnn.enabled set to {enable_cudnn}")

        return_tuple = (None, None, None, None, None)
        if model is not None:
            return_tuple = (model, None, None, None, None)
        elif conditioning is not None:
            return_tuple = (None, conditioning, None, None, None)
        elif latent is not None:
            return_tuple = (None, None, latent, None, None)
        elif audio is not None:
            return_tuple = (None, None, None, audio, None)
        elif image is not None:
            return_tuple = (None, None, None, None, image)

        return return_tuple

NODE_CLASS_MAPPINGS = {
    "CUDNNToggleAutoPassthrough": CUDNNToggleAutoPassthrough
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CUDNNToggleAutoPassthrough": "CFZ CUDNN Toggle"
}
