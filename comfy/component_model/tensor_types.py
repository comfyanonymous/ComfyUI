from typing import Annotated

from jaxtyping import Float, Shaped
from torch import Tensor


def channels_constraint(n: int):
    def constraint(x: Tensor) -> bool:
        return x.shape[-1] == n

    return constraint


ImageBatch = Float[Tensor, "batch height width channels"]
RGBImageBatch = Annotated[ImageBatch, Shaped[channels_constraint(3)]] | Float[Tensor, "batch height width 3"]
RGBAImageBatch = Annotated[ImageBatch, Shaped[channels_constraint(4)]] | Float[Tensor, "batch height width 4"]
RGBImage = Float[Tensor, "height width 3"]
