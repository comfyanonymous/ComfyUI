import torch

class MaskedSection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "image": ("IMAGE",),
                "minimum": ("INT", {"default":512, "min":16, "max":4096})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    CATEGORY = "image_filter/helpers"
    
    def func(self, mask:torch.Tensor, image, minimum=512):
        mbb = mask.squeeze()
        H,W = mbb.shape
        masked = mbb > 0.5

        non_zero_positions = torch.nonzero(masked)
        if len(non_zero_positions) < 2: return (image,)

        min_x = int(torch.min(non_zero_positions[:, 1]))
        max_x = int(torch.max(non_zero_positions[:, 1]))
        min_y = int(torch.min(non_zero_positions[:, 0]))
        max_y = int(torch.max(non_zero_positions[:, 0]))

        if (x:=(minimum-(max_x-min_x))//2)>0:
            min_x = max(min_x-x, 0)
            max_x = min(max_x+x, W)
        if (y:=(minimum-(max_y-min_y))//2)>0:
            min_y = max(min_y-y, 0)
            max_y = min(max_y+y, H)       

        return (image[:,min_y:max_y,min_x:max_x,:],)

