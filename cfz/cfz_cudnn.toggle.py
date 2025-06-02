import torch

class AutoCUDNNToggle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_cudnn": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "latent": ("LATENT",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("latent_out", "audio_out")
    FUNCTION = "toggle"
    CATEGORY = "advanced/utils"

    def toggle(self, enable_cudnn, latent=None, audio=None):
        # Set CuDNN state
        torch.backends.cudnn.enabled = enable_cudnn
        
        # Auto-detect active path
        if latent is not None:
            print(f"[CuDNN] Latent mode | Enabled: {enable_cudnn}")
            return (latent, None)
        elif audio is not None:
            print(f"[CuDNN] Audio mode | Enabled: {enable_cudnn}")
            return (None, audio)
        else:
            raise ValueError("No valid input connected - must connect either latent OR audio")

NODE_CLASS_MAPPINGS = {"CFZ-CUDNNToggle": AutoCUDNNToggle}
NODE_DISPLAY_NAME_MAPPINGS = {"CFZ-CUDNNToggle": "CFZ CuDNN Toggle"}
