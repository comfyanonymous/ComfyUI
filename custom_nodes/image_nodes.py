import torch.nn.functional as F
class ResizeImage:
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "max_dimension_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64}),
                     }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"

    def resize_image(self, image, max_dimension_size):
        _, height, width, _ = image.shape

        # Calculate the new dimensions while maintaining the aspect ratio
        if height > width:
            new_height = max_dimension_size
            new_width = int(width * (max_dimension_size / height))
        else:
            new_width = max_dimension_size
            new_height = int(height * (max_dimension_size / width))

        # Rearrange the image tensor to (1, 3, height, width) format
        image = image.permute(0, 3, 1, 2)

        # Resize the image using F.interpolate
        resized_image = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Rearrange the resized image tensor back to (1, height, width, 3) format
        resized_image = resized_image.permute(0, 2, 3, 1)

        return (resized_image,)


NODE_CLASS_MAPPINGS = {
    "ResizeImage": ResizeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResizeImage": "Resize Image",
}