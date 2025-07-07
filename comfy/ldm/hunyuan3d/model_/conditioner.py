import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov2 import DinoConfig, Dinov2Model

# avoid using torchvision by recreating image processing functions

def resize(img: torch.Tensor, size: int) -> torch.Tensor:
    
    batched = img.ndim == 4

    if not batched:
        img = img.unsqueeze(0)
        
    _, _, h, w = img.shape

    # mantain aspect ratio
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    
    img = F.interpolate(img, size = (new_h, new_w), mode = 'bilinear', align_corners = False, antialias = True ) 

    if not batched:
        img = img.squeeze(0)
        
    return img


def center_crop(img: torch.Tensor, size: int) -> torch.Tensor:

    batched = img.ndim == 4
    if not batched:
        img = img.unsqueeze(0)
    
    _, _, h, w = img.shape
    top = (h - size) // 2
    left = (w - size) // 2

        
    cropped = img[..., top:top + size, left:left + size]

    if not batched:
        cropped = cropped.squeeze(0)

    return cropped

def normalize(img: torch.Tensor, mean: list, std: list) -> torch.Tensor:

    mean = torch.tensor(mean, device = img.device).view(-1, 1, 1)
    std = torch.tensor(std, device = img.device).view(-1, 1, 1)
    return (img - mean) / std

def compose(transforms):
    def apply(img):
        for t in transforms:
            img = t(img)
        return img
    return apply

class ImageEncoder(nn.Module):
    def __init__(
        self,
        config: DinoConfig,
        use_cls_token = True,
        image_size = 518,
        **kwargs,
    ):
        super().__init__()

        self.model = Dinov2Model(config)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
            
        self.model.eval()
        self.model.requires_grad_(False)

        self.use_cls_token = use_cls_token
        self.size = image_size // 14

        self.num_patches = (image_size // 14) ** 2

        if self.use_cls_token:
            self.num_patches += 1

        self.transform = compose([
            lambda x: resize(x, image_size),
            lambda x: center_crop(x, image_size),
            lambda x: normalize(x, mean, std),
        ])

    def forward(self, image, value_range=(-1, 1), **kwargs):

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        inputs = self.transform(image)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        last_hidden_state = self.model(inputs)

        if not self.use_cls_token:
            last_hidden_state = last_hidden_state[:, 1:, :]

        return last_hidden_state

    def unconditional_embedding(self, batch_size, **kwargs):

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        zero = torch.zeros(
            batch_size,
            self.num_patches,
            self.model.config.hidden_size,
            device = device,
            dtype = dtype,
        )

        return zero
    
class SingleImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.main_image_encoder = ImageEncoder(config)

    def forward(self, image, **kwargs):
        outputs = {
            'main': self.main_image_encoder(image, **kwargs),
        }
        return outputs
    
    def unconditional_embedding(self, batch_size, **kwargs):
        outputs = {
            'main': self.main_image_encoder.unconditional_embedding(batch_size, **kwargs),
        }
        return outputs
    
def test_image_encoder():

    torch.manual_seed(2025)
    config = DinoConfig()
    image_encoder = SingleImageEncoder(config)

    image = torch.rand(3, 224, 224)

    outputs = image_encoder(image)

    print(outputs)

if __name__ == "__main__":
    #test_image_encoder()
    conditioner = SingleImageEncoder(DinoConfig())
    torch.manual_seed(2025)
    image = torch.rand(1, 3, 224, 224)
    outputs = conditioner(image)
    print(outputs["main"].size())