import torch
import comfy.model_management

from kornia.morphology import dilation, erosion, opening, closing, gradient, top_hat, bottom_hat


class Morphology:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                                "operation": (["erode",  "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],),
                                "kernel_size": ("INT", {"default": 3, "min": 3, "max": 999, "step": 1}),
                                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "image/postprocessing"

    def process(self, image, operation, kernel_size):
        device = comfy.model_management.get_torch_device()
        kernel = torch.ones(kernel_size, kernel_size, device=device)
        image_k = image.to(device).movedim(-1, 1)
        if operation == "erode":
            output = erosion(image_k, kernel)
        elif operation == "dilate":
            output = dilation(image_k, kernel)
        elif operation == "open":
            output = opening(image_k, kernel)
        elif operation == "close":
            output = closing(image_k, kernel)
        elif operation == "gradient":
            output = gradient(image_k, kernel)
        elif operation == "top_hat":
            output = top_hat(image_k, kernel)
        elif operation == "bottom_hat":
            output = bottom_hat(image_k, kernel)
        else:
            raise ValueError(f"Invalid operation {operation} for morphology. Must be one of 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'bottomhat'")
        img_out = output.to(comfy.model_management.intermediate_device()).movedim(1, -1)
        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "Morphology": Morphology,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Morphology": "ImageMorphology",
}