import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F

def to_tensor(image_pt):

    image_pt = image_pt / 255 * 2 - 1
    if image_pt.dim() == 4:
        image_pt = image_pt.permute(0, 3, 1, 2)
    else: 
        image_pt = image_pt.permute(2, 1, 0)

    return image_pt

def resize_bilinear(img: torch.Tensor, size: int) -> torch.Tensor:
    # pytorch implementation of cv2.INTER_LINEAR

    batched = (img.ndim == 4)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    B, _, H, W = img.shape
    H_out = W_out = size

    xs = torch.linspace(0, H_out  - 1, H_out, device = img.device)
    ys = torch.linspace(0, W_out  - 1, W_out, device = img.device)

    xs = (xs + 0.5) * (H / H_out) - 0.5   
    ys = (ys + 0.5) * (W / W_out) - 0.5  

    # normalize
    xs = 2 * xs / (H - 1) - 1             
    ys = 2 * ys / (W - 1) - 1      

    # meshgrid in “ij” order: first rows (xs), then cols (ys)
    grid_i, grid_j = torch.meshgrid(xs, ys, indexing='ij') 

    # stack into (x,y) where x=columns, y=rows
    grid = torch.stack((grid_j, grid_i), dim=-1)            
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)     

    out = F.grid_sample(img, grid, mode = 'bilinear',
                        padding_mode = 'zeros', align_corners = True)
    
    return out if batched else out.squeeze(0)

def resize_bicubic(img: torch.Tensor, size: int) -> torch.Tensor:
    # pytorch implementation of INTER_CUBIC

    was_batched = img.ndim == 4
    if img.ndim == 3:
        img = img.unsqueeze(0)  

    out = F.interpolate(
        img.permute(0, 3, 2, 1),
        size = (size, size),
        mode = "bicubic",
        align_corners = True
    )
    
    return out if was_batched else out.squeeze(0)

def resize_area(img: torch.Tensor, size: tuple) -> torch.Tensor:
    # pytorch implementation of INTER_AREA

    was_batched = img.ndim == 4
    if img.ndim == 3:
        img = img.unsqueeze(0)

    image = F.interpolate(img.permute(0,3,1,2).float(), (size[1], size[0]), mode = "area")

    if was_batched:
        image = image.permute(0, 2, 3, 1) # return to channel last
    else:
        image = image.squeeze(0).permute(1, 2, 0)

    return image if was_batched else image.squeeze(0)

class ImageProcessorV2(nn.Module):
    def __init__(self, size: int = 512, border_ratio: float = None):
        self.size = size
        self.border_ratio = border_ratio

    def load_image(self, pic, border_ratio: float = 0.15) -> torch.Tensor:

        if isinstance(pic, str):
            img = Image.open(pic)
            img = np.array(img)
            
        elif isinstance(pic, Image.Image):
            img = np.array(pic) 

        if img.ndim == 2:  # grayscale
            img = img[:, :, None]

        img = torch.from_numpy(img)
        img, mask = self.recenter(img, border_ratio = border_ratio)

        img = resize_bicubic(img, size = self.size)
        mask = resize_bilinear(mask.float(), size = self.size)
        mask = mask[..., torch.newaxis]

        img = to_tensor(img)
        mask = to_tensor(mask)

        return img, mask
    
    @staticmethod
    def recenter(image, border_ratio: float = 0.2):

        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask = torch.ones_like(image[..., 0:1]) * 255
            image = torch.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]

        H, W, C = image.shape

        size = max(H, W)
        result = torch.zeros((size, size, C), dtype = torch.uint8)

        # as_tuple to match numpy behaviour
        x_coords, y_coords = torch.nonzero(mask, as_tuple=True)

        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        h = x_max - x_min
        w = y_max - y_min
        
        if h == 0 or w == 0:
            raise ValueError('input image is empty')
        
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)

        h2 = int(h * scale)
        w2 = int(w * scale)

        x2_min = (size - h2) // 2
        x2_max = x2_min + h2

        y2_min = (size - w2) // 2
        y2_max = y2_min + w2

        # note: opencv takes columns first (opposite to pytorch and numpy that take the row first)
        result[x2_min:x2_max, y2_min:y2_max] = resize_area(image[x_min:x_max, y_min:y_max], (w2, h2))

        bg = torch.ones((result.shape[0], result.shape[1], 3), dtype = torch.uint8) * 255

        mask = result[..., 3:].to(torch.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)

        mask = mask * 255
        result = result.clip(0, 255).to(torch.uint8)
        mask = mask.clip(0, 255).to(torch.uint8)
        
        return result, mask
    
    def __call__(self, image, border_ratio = 0.15, **kwargs):

        if self.border_ratio is not None:
            border_ratio = self.border_ratio

        image, mask = self.load_image(image, border_ratio = border_ratio)

        outputs = {
            'image': image,
            'mask': mask
        }

        return outputs
    
def test_image_processor():
    import matplotlib.pyplot as plt

    image_processor = ImageProcessorV2(size = 224)
    import time
    start = time.time()
    outputs = image_processor(image = r"C:\Users\yrafa\Work\Hunyuan 3D\cat.jpg")
    print(time.time() - start)
    image = outputs["image"]
    print(image.shape)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
if __name__ == "__main__":
    test_image_processor()