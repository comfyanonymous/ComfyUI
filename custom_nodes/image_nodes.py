import torch.nn.functional as F
class ResizeImage:
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "min_dimension_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64}),
                     }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"

    def resize_image(self, image, min_dimension_size):
        _, height, width, _ = image.shape

        if height < width:
            new_height = min_dimension_size
            new_width = int(width * (min_dimension_size / height))
        else:
            new_width = min_dimension_size
            new_height = int(height * (min_dimension_size / width))

        image = image.permute(0, 3, 1, 2)
        resized_image = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)
        resized_image = resized_image.permute(0, 2, 3, 1)

        return (resized_image,)


NODE_CLASS_MAPPINGS = {
    "ResizeImage": ResizeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResizeImage": "Resize Image",
}