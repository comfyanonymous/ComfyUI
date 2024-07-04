import torch
from skimage.morphology import skeletonize, thin

import comfy.model_management


class SkeletonizeThin:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "binary_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
            "approach": (["skeletonize", "thinning"], {}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "image/preprocessors"

    def process_image(self, image, binary_threshold, approach):
        use_skeletonize = approach == "skeletonize"
        use_thinning = approach == "thinning"
        device = comfy.model_management.intermediate_device()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        batch_size, height, width, channels = image.shape
        if channels == 3:
            image = torch.mean(image, dim=-1, keepdim=True)
        binary = (image > binary_threshold).float()

        results = []
        for img in binary:
            img_np = img.squeeze().cpu().numpy()

            if use_skeletonize:
                result = skeletonize(img_np)
            elif use_thinning:
                result = thin(img_np)
            else:
                result = img_np

            result = torch.from_numpy(result).float().to(device)
            result = result.unsqueeze(-1).repeat(1, 1, 3)
            results.append(result)
        final_result = torch.stack(results).to(comfy.model_management.intermediate_device())
        return (final_result,)


NODE_CLASS_MAPPINGS = {
    "SkeletonizeThin": SkeletonizeThin,
}
