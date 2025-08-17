from .utils import FlexibleOptionalInputType, any_type
from .constants import get_category, get_name


class RgthreeImageOrLatentSize:
  """The ImageOrLatentSize Node."""

  NAME = get_name('Image or Latent Size')
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType(any_type),
    }

  RETURN_TYPES = ("INT", "INT")
  RETURN_NAMES = ('WIDTH', 'HEIGHT')
  FUNCTION = "main"

  def main(self, **kwargs):
    """Does the node's work."""
    image_or_latent_or_mask = kwargs.get('input', None)

    if isinstance(image_or_latent_or_mask, dict) and 'samples' in image_or_latent_or_mask:
      count, _, height, width = image_or_latent_or_mask['samples'].shape
      return (width * 8, height * 8)

    batch, height, width, channel = image_or_latent_or_mask.shape
    return (width, height)
