import torch
import comfy.utils
import nodes

from .constants import get_category, get_name


class RgthreeImageResize:
  """Image Resize."""

  NAME = get_name("Image Resize")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "image": ("IMAGE",),
        "measurement": (["pixels", "percentage"],),
        "width": (
          "INT", {
            "default": 0,
            "min": 0,
            "max": nodes.MAX_RESOLUTION,
            "step": 1,
            "tooltip": (
              "The width of the desired resize. A pixel value if measurement is 'pixels' or a"
              " 100% scale percentage value if measurement is 'percentage'. Passing '0' will"
              " calculate the dimension based on the height."
            ),
          },
        ),
        "height": ("INT", {
          "default": 0,
          "min": 0,
          "max": nodes.MAX_RESOLUTION,
          "step": 1
        }),
        "fit": (["crop", "pad", "contain"], {
          "tooltip": (
            "'crop' resizes so the image covers the desired width and height, and center-crops the"
            " excess, returning exactly the desired width and height."
            "\n'pad' resizes so the image fits inside the desired width and height, and fills the"
            " empty space returning exactly the desired width and height."
            "\n'contain' resizes so the image fits inside the desired width and height, and"
            " returns the image with it's new size, with one side liekly smaller than the desired."
            "\n\nNote, if either width or height is '0', the effective fit is 'contain'."
          )
        },
               ),
        "method": (nodes.ImageScale.upscale_methods,),
      },
    }

  RETURN_TYPES = ("IMAGE", "INT", "INT",)
  RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT",)
  FUNCTION = "main"
  DESCRIPTION = """Resize the image."""

  def main(self, image, measurement, width, height, method, fit):
    """Resizes the image."""
    _, H, W, _ = image.shape

    if measurement == "percentage":
      width = round(width * W / 100)
      height = round(height * H / 100)

    if (width == 0 and height == 0) or (width == W and height == H):
      return (image, W, H)

    # If one dimension is 0, then calculate the desired value from the ratio of the set dimension.
    # This also implies a 'contain' fit since the width and height will be scaled with a locked
    # aspect ratio.
    if width == 0 or height == 0:
      width = round(height / H * W) if width == 0 else width
      height = round(width / W * H) if height == 0 else height
      fit = "contain"

    # At this point, width and height are our output height, but our resize sizes will be different.
    resized_width = width
    resized_height = height
    if fit == "crop":
      # If we resize against the opposite ratio, then choose the ratio that has the overhang.
      if (height / H * W) > width:
        resized_width = round(height / H * W)
      elif (width / W * H) > height:
        resized_height = round(width / W * H)
    elif fit == "contain" or fit == "pad":
      # If we resize against the opposite ratio, then choose the ratio that has the overhang.
      if (height / H * W) > width:
        resized_height = round(width / W * H)
      elif (width / W * H) > height:
        resized_width = round(height / H * W)

    out_image = comfy.utils.common_upscale(
      image.clone().movedim(-1, 1), resized_width, resized_height, method, crop="disabled"
    ).movedim(1, -1)
    OB, OH, OW, OC = out_image.shape

    if fit != "contain":
      # First, we crop, then we pad; no need to check fit (other than not 'contain') since the size
      # should already be correct.
      if OW > width:
        out_image = out_image.narrow(-2, (OW - width) // 2, width)
      if OH > height:
        out_image = out_image.narrow(-3, (OH - height) // 2, height)

      OB, OH, OW, OC = out_image.shape
      if width != OW or height != OH:
        padded_image = torch.zeros((OB, height, width, OC), dtype=image.dtype, device=image.device)
        x = (width - OW) // 2
        y = (height - OH) // 2
        for b in range(OB):
          padded_image[b, y:y + OH, x:x + OW, :] = out_image[b]
        out_image = padded_image

    return (out_image, out_image.shape[2], out_image.shape[1])
