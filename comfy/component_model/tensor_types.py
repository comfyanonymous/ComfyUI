from typing import TypedDict

from jaxtyping import Float
from torch import Tensor
from typing_extensions import NotRequired

ImageBatch = Float[Tensor, "batch height width channels"]
LatentBatch = Float[Tensor, "batch channels width height"]
SD15LatentBatch = Float[Tensor, "batch 4 height width"]
SDXLLatentBatch = Float[Tensor, "batch 8 height width"]
SD3LatentBatch = Float[Tensor, "batch 16 height width"]
MaskBatch = Float[Tensor, "batch height width"]
RGBImageBatch = Float[Tensor, "batch height width 3"]
RGBAImageBatch = Float[Tensor, "batch height width 4"]
RGBImage = Float[Tensor, "height width 3"]


class Latent(TypedDict):
    samples: LatentBatch
    noise_mask: NotRequired[LatentBatch]
