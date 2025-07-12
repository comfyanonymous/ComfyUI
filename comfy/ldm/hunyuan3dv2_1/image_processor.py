import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F

def to_tensor(image_pt):
    image_pt = image_pt / 255 * 2 - 1
    return image_pt

def resize_nearest(img: torch.Tensor, size: int) -> torch.Tensor:

    batched = (img.ndim == 4)
    if img.ndim == 3:       
        img = img.unsqueeze(0)

    img = img.permute(0, 3, 1, 2)

    out =  F.interpolate(img, size=size, mode='nearest')

    if not batched:
        out = out.squeeze(0) 

    return out

def cubic_kernel(x, a: float = -0.75):
    absx = x.abs()
    absx2 = absx ** 2
    absx3 = absx ** 3

    w = (a + 2) * absx3 - (a + 3) * absx2 + 1
    w2 = a * absx3 - 5*a * absx2 + 8*a * absx - 4*a

    return torch.where(absx <= 1, w, torch.where(absx < 2, w2, torch.zeros_like(x)))


def get_indices_weights(in_size, out_size, scale):
    # OpenCV-style half-pixel mapping
    x = torch.arange(out_size, dtype=torch.float32)
    x = (x + 0.5) / scale - 0.5

    x0 = x.floor().long()
    dx = x.unsqueeze(1) - (x0.unsqueeze(1) + torch.arange(-1, 3))

    weights = cubic_kernel(dx) 
    weights = weights / weights.sum(dim=1, keepdim=True)

    indices = x0.unsqueeze(1) + torch.arange(-1, 3)
    indices = indices.clamp(0, in_size - 1)

    return indices, weights


def resize_cubic_1d(x, out_size, dim):
    b, c, h, w = x.shape
    in_size = h if dim == 2 else w
    scale = out_size / in_size

    indices, weights = get_indices_weights(in_size, out_size, scale)

    if dim == 2:
        x = x.permute(0, 1, 3, 2)  
        x = x.reshape(-1, h)  
    else:
        x = x.reshape(-1, w)  

    gathered = x[:, indices]  
    out = (gathered * weights.unsqueeze(0)).sum(dim=2)

    if dim == 2:
        out = out.reshape(b, c, w, out_size).permute(0, 1, 3, 2)
    else:
        out = out.reshape(b, c, h, out_size)

    return out


def resize_cubic(img: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Resize image using OpenCV-equivalent INTER_CUBIC interpolation.
    Implemented in pure PyTorch
    """

    if img.ndim == 3:
        img = img.unsqueeze(0)

    img = img.permute(0, 3, 1, 2)
        
    out_h, out_w = size
    img = resize_cubic_1d(img, out_h, dim=2)
    img = resize_cubic_1d(img, out_w, dim=3)
    return img

def resize_area(img: torch.Tensor, size: tuple) -> torch.Tensor:
    # vectorized implementation for OpenCV's INTER_AREA using pure PyTorch
    original_shape = img.shape
    is_hwc = False

    if img.ndim == 3:
        if img.shape[0] <= 4:
            img = img.unsqueeze(0)
        else:
            is_hwc = True
            img = img.permute(2, 0, 1).unsqueeze(0)
    elif img.ndim == 4:
        pass
    else:
        raise ValueError("Expected image with 3 or 4 dims.")

    B, C, H, W = img.shape
    out_h, out_w = size
    scale_y = H / out_h
    scale_x = W / out_w

    device = img.device

    # compute the grid boundries
    y_start = torch.arange(out_h, device=device).float() * scale_y
    y_end = y_start + scale_y
    x_start = torch.arange(out_w, device=device).float() * scale_x
    x_end = x_start + scale_x

    # for each output pixel, we will compute the range for it
    y_start_int = torch.floor(y_start).long()
    y_end_int = torch.ceil(y_end).long()
    x_start_int = torch.floor(x_start).long()
    x_end_int = torch.ceil(x_end).long()

    # We will build the weighted sums by iterating over contributing input pixels once
    output = torch.zeros((B, C, out_h, out_w), dtype=torch.float32, device=device)
    area = torch.zeros((out_h, out_w), dtype=torch.float32, device=device)
    
    max_kernel_h = int(torch.max(y_end_int - y_start_int).item())
    max_kernel_w = int(torch.max(x_end_int - x_start_int).item())

    for dy in range(max_kernel_h):
        for dx in range(max_kernel_w):
            # compute the weights for this offset for all output pixels

            y_idx = y_start_int.unsqueeze(1) + dy  
            x_idx = x_start_int.unsqueeze(0) + dx  

            # clamp indices to image boundaries
            y_idx_clamped = torch.clamp(y_idx, 0, H - 1)
            x_idx_clamped = torch.clamp(x_idx, 0, W - 1)

            # compute weights by broadcasting
            y_weight = (torch.min(y_end.unsqueeze(1), y_idx_clamped.float() + 1.0) - torch.max(y_start.unsqueeze(1), y_idx_clamped.float())).clamp(min=0)
            x_weight = (torch.min(x_end.unsqueeze(0), x_idx_clamped.float() + 1.0) - torch.max(x_start.unsqueeze(0), x_idx_clamped.float())).clamp(min=0)

            weight = (y_weight * x_weight)

            y_expand = y_idx_clamped.expand(out_h, out_w)
            x_expand = x_idx_clamped.expand(out_h, out_w)


            pixels = img[:, :, y_expand, x_expand]

            # unsqueeze to broadcast
            w = weight.unsqueeze(0).unsqueeze(0)

            output += pixels * w
            area += weight

    # Normalize by area
    output /= area.unsqueeze(0).unsqueeze(0)

    if is_hwc:
        return output[0].permute(1, 2, 0)
    elif img.shape[0] == 1 and original_shape[0] <= 4:
        return output[0]
    else:
        return output


class ImageProcessorV2(nn.Module):
    def __init__(self, size: int = 512, border_ratio: float = None):
        super().__init__()
        
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

        img = resize_cubic(img, size = (self.size, self.size))
        mask = resize_nearest(mask.float(), size = self.size)
        mask = mask[..., torch.newaxis]

        img = to_tensor(img)

        mask = to_tensor(mask)
        mask = mask.permute(0, 3, 1, 2)

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
        result[x2_min:x2_max, y2_min:y2_max] = resize_area(image[x_min:x_max, y_min:y_max], (h2, w2))

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

    """
    implementation speed:  0.24465346336364746
    reference speed:  2.046062469482422
    atol = 4e-2: True
    """

    import time
    import matplotlib.pyplot as plt

    image_processor = ImageProcessorV2(size = 224)
    start = time.time()
    outputs = image_processor(image = r"C:\Users\yrafa\Work\Hunyuan 3D\cat.jpg")
    print(time.time() - start)
    image = outputs["image"]
    print(image.shape)
    plt.imshow(image.squeeze().permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.show()