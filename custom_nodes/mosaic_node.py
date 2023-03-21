import torch

class Mosaic:
    interpolation_methods = ['nearest', 'bilinear', 'bicubic', 'area', 'nearest-exact']

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {
                    "default": 4, 
                    "min": 1,
                    "max": 512,
                    "step": 1
                }),
                "interpolation_method": (s.interpolation_methods, {"default": s.interpolation_methods[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mosaic"

    CATEGORY = "image"

    def mosaic(self, image, interpolation_method, pixel_size):
        samples = image.movedim(-1,1)

        starting_width = samples.shape[3]
        starting_height = samples.shape[2]

        #downsample dimensions
        dowsample_width = int(starting_width / pixel_size) | 1
        dowsample_height = int(starting_height / pixel_size) | 1

        downsampled_image = torch.nn.functional.interpolate(samples, size=(dowsample_height, dowsample_width), mode=interpolation_method)
        output_image = torch.nn.functional.interpolate(downsampled_image, size=(starting_height, starting_width), mode='nearest')

        output = output_image.movedim(1,-1)

        return (output,)

NODE_CLASS_MAPPINGS = {
    "Mosaic": Mosaic
}