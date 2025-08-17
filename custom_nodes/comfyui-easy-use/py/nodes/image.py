import os
import folder_paths
import torch
import numpy as np
import comfy.utils
import comfy.model_management
import shutil
from comfy_extras.nodes_compositing import JoinImageWithAlpha
from server import PromptServer
from nodes import MAX_RESOLUTION, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch.nn.functional as F
from torchvision.transforms import Resize, CenterCrop, GaussianBlur, ToPILImage
from torchvision.transforms.functional import to_pil_image
from ..libs.log import log_node_info
from ..libs.utils import AlwaysEqualProxy, ByPassTypeTuple
from ..libs.cache import cache, update_cache, remove_cache
from ..libs.image import pil2tensor, tensor2pil, ResizeMode, get_new_bounds, RGB2RGBA, image2mask, empty_image, fit_resize_image
from ..libs.colorfix import adain_color_fix, wavelet_color_fix
from ..config import REMBG_DIR, REMBG_MODELS, HUMANPARSING_MODELS, MEDIAPIPE_MODELS, MEDIAPIPE_DIR

any_type = AlwaysEqualProxy("*")
# 图像数量
class imageCount:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
      }
    }

  CATEGORY = "EasyUse/Image"

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("count",)
  FUNCTION = "get_count"

  def get_count(self, images):
    return (images.size(0),)

class imagesCountInDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "directory": ("STRING",),
          "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
          "limit": ("INT", {"default": -1, "min": -1, "max": 10000}),
        }
      }

    CATEGORY = "EasyUse/Image"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "get_count"

    def get_count(self, directory, start_index, limit, **kwargs):
      dir_files = os.listdir(directory)
      valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
      dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
      if limit == -1:
        files_length = len(dir_files)
        total = files_length - start_index if start_index > 0 else files_length
      else:
        total = limit
      return (total,)

# 图像裁切
class imageInsetCrop:

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "image": ("IMAGE",),
        "measurement": (['Pixels', 'Percentage'],),
        "left": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
        "right": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
        "top": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
        "bottom": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
      },
    }

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "crop"

  CATEGORY = "EasyUse/Image"

  # pylint: disable = too-many-arguments
  def crop(self, measurement, left, right, top, bottom, image=None):
    """Does the crop."""

    _, height, width, _ = image.shape

    if measurement == 'Percentage':
      left = int(width - (width * (100 - left) / 100))
      right = int(width - (width * (100 - right) / 100))
      top = int(height - (height * (100 - top) / 100))
      bottom = int(height - (height * (100 - bottom) / 100))

    # Snap to 8 pixels
    left = left // 8 * 8
    right = right // 8 * 8
    top = top // 8 * 8
    bottom = bottom // 8 * 8

    if left == 0 and right == 0 and bottom == 0 and top == 0:
      return (image,)

    inset_left, inset_right, inset_top, inset_bottom = get_new_bounds(width, height, left, right,
                                                                      top, bottom)
    if inset_top > inset_bottom:
      raise ValueError(
        f"Invalid cropping dimensions top ({inset_top}) exceeds bottom ({inset_bottom})")
    if inset_left > inset_right:
      raise ValueError(
        f"Invalid cropping dimensions left ({inset_left}) exceeds right ({inset_right})")

    log_node_info("Image Inset Crop", f'Cropping image {width}x{height} width inset by {inset_left},{inset_right}, ' +
                 f'and height inset by {inset_top}, {inset_bottom}')
    image = image[:, inset_top:inset_bottom, inset_left:inset_right, :]

    return (image,)

# 图像尺寸
class imageSize:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT", "INT")
  RETURN_NAMES = ("width_int", "height_int")
  OUTPUT_NODE = True
  FUNCTION = "image_width_height"

  CATEGORY = "EasyUse/Image"

  def image_width_height(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H

    if width is not None and height is not None:
      result = (width, height)
    else:
      result = (0, 0)
    return {"ui": {"text": "Width: "+str(width)+" , Height: "+str(height)}, "result": result}

# 图像尺寸（最长边）
class imageSizeBySide:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
        "side": (["Longest", "Shortest"],)
      }
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("resolution",)
  OUTPUT_NODE = True
  FUNCTION = "image_side"

  CATEGORY = "EasyUse/Image"

  def image_side(self, image, side):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H
    if width is not None and height is not None:
      if side == "Longest":
         result = (width,) if width > height else (height,)
      elif side == 'Shortest':
         result = (width,) if width < height else (height,)
    else:
      result = (0,)
    return {"ui": {"text": str(result[0])}, "result": result}

# 图像尺寸（最长边）
class imageSizeByLongerSide:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("resolution",)
  OUTPUT_NODE = True
  FUNCTION = "image_longer_side"

  CATEGORY = "EasyUse/Image"

  def image_longer_side(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H
    if width is not None and height is not None:
      if width > height:
         result = (width,)
      else:
         result = (height,)
    else:
      result = (0,)
    return {"ui": {"text": str(result[0])}, "result": result}

# 图像缩放
class imageScaleDown:
  crop_methods = ["disabled", "center"]

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "width": (
          "INT",
          {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
        ),
        "height": (
          "INT",
          {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
        ),
        "crop": (s.crop_methods,),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "EasyUse/Image"
  FUNCTION = "image_scale_down"

  def image_scale_down(self, images, width, height, crop):
    if crop == "center":
      old_width = images.shape[2]
      old_height = images.shape[1]
      old_aspect = old_width / old_height
      new_aspect = width / height
      x = 0
      y = 0
      if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
      elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
      s = images[:, y: old_height - y, x: old_width - x, :]
    else:
      s = images

    results = []
    for image in s:
      img = tensor2pil(image).convert("RGB")
      img = img.resize((width, height), Image.LANCZOS)
      results.append(pil2tensor(img))

    return (torch.cat(results, dim=0),)

# 图像缩放比例
class imageScaleDownBy(imageScaleDown):
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "scale_by": (
          "FLOAT",
          {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01},
        ),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "EasyUse/Image"
  FUNCTION = "image_scale_down_by"

  def image_scale_down_by(self, images, scale_by):
    width = images.shape[2]
    height = images.shape[1]
    new_width = int(width * scale_by)
    new_height = int(height * scale_by)
    return self.image_scale_down(images, new_width, new_height, "center")

# 图像缩放尺寸
class imageScaleDownToSize(imageScaleDownBy):
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "size": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
        "mode": ("BOOLEAN", {"default": True, "label_on": "max", "label_off": "min"}),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "EasyUse/Image"
  FUNCTION = "image_scale_down_to_size"

  def image_scale_down_to_size(self, images, size, mode):
    width = images.shape[2]
    height = images.shape[1]

    if mode:
      scale_by = size / max(width, height)
    else:
      scale_by = size / min(width, height)

    scale_by = min(scale_by, 1.0)
    return self.image_scale_down_by(images, scale_by)

class imageScaleToNormPixels:
  upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image": ("IMAGE",),
        "upscale_method": (s.upscale_methods,),
        "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "scale"
  CATEGORY = "EasyUse/Image"

  def scale(self, image, upscale_method, scale_by):
    height, width = image.shape[1:3]
    width = int(width * scale_by - width * scale_by % 8)
    height = int(height * scale_by - height * scale_by % 8)
    upscale_image_cls = ALL_NODE_CLASS_MAPPINGS['ImageScale']
    image, = upscale_image_cls().upscale(image, upscale_method, width, height, "disabled")
    return (image,)

# 图像比率
class imageRatio:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
  RETURN_NAMES = ("width_ratio_int", "height_ratio_int", "width_ratio_float", "height_ratio_float")
  OUTPUT_NODE = True
  FUNCTION = "image_ratio"

  CATEGORY = "EasyUse/Image"

  def gcf(self, a, b):
    while b:
      a, b = b, a % b
    return a

  def image_ratio(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H

    ratio = self.gcf(width, height)

    if width is not None and height is not None:
      width_ratio = width // ratio
      height_ratio = height // ratio
      result = (width_ratio, height_ratio, width_ratio, height_ratio)
    else:
      width_ratio = 0
      height_ratio = 0
      result = (0, 0, 0.0, 0.0)
    text = f"Image Ratio is {str(width_ratio)}:{str(height_ratio)}"

    return {"ui": {"text": text}, "result": result}


# 图像完美像素
class imagePixelPerfect:
  @classmethod
  def INPUT_TYPES(s):
    RESIZE_MODES = [ResizeMode.RESIZE.value, ResizeMode.INNER_FIT.value, ResizeMode.OUTER_FIT.value]
    return {
      "required": {
        "image": ("IMAGE",),
        "resize_mode": (RESIZE_MODES, {"default": ResizeMode.RESIZE.value})
      }
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("resolution",)
  OUTPUT_NODE = True
  FUNCTION = "execute"

  CATEGORY = "EasyUse/Image"

  def execute(self, image, resize_mode):

    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H

    k0 = float(height) / float(raw_H)
    k1 = float(width) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT.value:
      estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
      estimation = max(k0, k1) * float(min(raw_H, raw_W))

    result = int(np.round(estimation))
    text = f"Width:{str(width)}\nHeight:{str(height)}\nPixelPerfect:{str(result)}"

    return {"ui": {"text": text}, "result": (result,)}

# 图像保存 (简易)
from nodes import PreviewImage, SaveImage
class imageSaveSimple:

  def __init__(self):
    self.output_dir = folder_paths.get_output_directory()
    self.type = "output"
    self.prefix_append = ""
    self.compress_level = 4

  @classmethod
  def INPUT_TYPES(s):
    return {"required":
              {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "only_preview": ("BOOLEAN", {"default": False}),
              },
              "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

  RETURN_TYPES = ()
  FUNCTION = "save"
  OUTPUT_NODE = True
  CATEGORY = "EasyUse/Image"

  def save(self, images, filename_prefix="ComfyUI", only_preview=False, prompt=None, extra_pnginfo=None):
    if only_preview:
      return PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
    else:
      return SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)

# 图像批次合并
class JoinImageBatch:
  """Turns an image batch into one big image."""

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "mode": (("horizontal", "vertical"), {"default": "horizontal"}),
      },
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "join"
  CATEGORY = "EasyUse/Image"

  def join(self, images, mode):
    n, h, w, c = images.shape
    image = None
    if mode == "vertical":
      # for vertical we can just reshape
      image = images.reshape(1, n * h, w, c)
    elif mode == "horizontal":
      # for horizontal we have to swap axes
      image = torch.transpose(torch.transpose(images, 1, 2).reshape(1, n * w, h, c), 1, 2)
    return (image,)

class imageListToImageBatch:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
      "images": ("IMAGE",),
    }}

  INPUT_IS_LIST = True

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "doit"

  CATEGORY = "EasyUse/Image"

  def doit(self, images):
    if len(images) <= 1:
      return (images[0],)
    else:
      image_shape = images[0].shape
      for i, img in enumerate(images):
        if image_shape[1:] == img[1:]:
          continue
        else:
          images[i] = comfy.utils.common_upscale(img.movedim(-1, 1), img.shape[2], image_shape[1], "lanczos",
                                              "center").movedim(1, -1)
      images = torch.cat(images, dim=0)
      return (images,)


class imageBatchToImageList:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {"image": ("IMAGE",), }}

  RETURN_TYPES = ("IMAGE",)
  OUTPUT_IS_LIST = (True,)
  FUNCTION = "doit"

  CATEGORY = "EasyUse/Image"

  def doit(self, image):
    images = [image[i:i + 1, ...] for i in range(image.shape[0])]
    return (images,)

# 图像拆分
class imageSplitList:
  @classmethod

  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
      },
    }

  RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
  RETURN_NAMES = ("images", "images", "images",)
  FUNCTION = "doit"
  CATEGORY = "EasyUse/Image"

  def doit(self, images):
    length = len(images)
    new_images = ([], [], [])
    if length % 3 == 0:
      for index, img in enumerate(images):
        if index % 3 == 0:
          new_images[0].append(img)
        elif (index+1) % 3 == 0:
          new_images[2].append(img)
        else:
          new_images[1].append(img)
    elif length % 2 == 0:
      for index, img in enumerate(images):
        if index % 2 == 0:
          new_images[0].append(img)
        else:
          new_images[1].append(img)
    return new_images

class imageSplitGrid:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "row": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
        "column": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("images",)
  FUNCTION = "doit"
  CATEGORY = "EasyUse/Image"

  def crop(self, image, width, height, x, y):
    x = min(x, image.shape[2] - 1)
    y = min(y, image.shape[1] - 1)
    to_x = width + x
    to_y = height + y
    img = image[:, y:to_y, x:to_x, :]
    return img

  def doit(self, images, row, column):
    _, height, width, _ = images.shape
    sub_width = width // column
    sub_height = height // row
    new_images = []
    for i in range(row):
        for j in range(column):
          new_images.append(self.crop(images, sub_width, sub_height, j * sub_width, i * sub_height))

    return (torch.cat(new_images, dim=0),)

class imageSplitTiles:

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image": ("IMAGE",),
        "overlap_ratio": ("FLOAT", {"default": 0, "min": 0, "max": 0.5, "step": 0.01, }),
        "overlap_offset": ("INT", {"default": 0, "min": - MAX_RESOLUTION // 2, "max": MAX_RESOLUTION // 2, "step": 1, }),
        "tiles_rows": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
        "tiles_cols": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
      },
      "optional": {
        "norm": ("BOOLEAN", {"default": True}),
      }
    }

  RETURN_TYPES = ("IMAGE", "MASK", "OVERLAP", "INT")
  RETURN_NAMES = ("tiles", "masks", "overlap", "total")
  FUNCTION = "doit"
  CATEGORY = "EasyUse/Image"

  def doit(self, image, overlap_ratio, overlap_offset, tiles_rows, tiles_cols, norm=True):
    height, width = image.shape[1:3]

    total = tiles_rows * tiles_cols
    tile_w = int(width // tiles_cols)
    tile_h = int(height // tiles_rows)

    overlap_w = int(tile_w * overlap_ratio) + overlap_offset
    overlap_h = int(tile_h * overlap_ratio) + overlap_offset

    overlap_w = min(tile_w // 2, overlap_w)
    overlap_h = min(tile_h // 2, overlap_h)

    if norm:
      overlap_w = int(overlap_w - overlap_w % 8)
      overlap_h = int(overlap_h - overlap_h % 8)

    if tiles_rows == 1:
      overlap_h = 0
    if tiles_cols == 1:
      overlap_w = 0

    solid_mask_cls = ALL_NODE_CLASS_MAPPINGS['SolidMask']
    feather_mask_cls = ALL_NODE_CLASS_MAPPINGS['FeatherMask']

    tiles, masks = [], []

    x, y = 0, 0
    for i in range(tiles_rows):
      for j in range(tiles_cols):
        y1 = i * tile_h
        x1 = j * tile_w

        if i > 0:
          y1 -= overlap_h
        if j > 0:
          x1 -= overlap_w

        y2 = y1 + tile_h + overlap_h
        x2 = x1 + tile_w + overlap_w

        if y2 > height:
          y2 = height
          y1 = y2 - tile_h - overlap_h
        if x2 > width:
          x2 = width
          x1 = x2 - tile_w - overlap_w

        tile = image[:, y1:y2, x1:x2, :]
        h = tile.shape[1]
        w = tile.shape[2]
        tiles.append(tile)

        fearing_left = overlap_w if overlap_w * j > 0 else 0
        fearing_top = overlap_h if overlap_h * i > 0 else 0
        fearing_right = 0
        fearing_bottom = 0

        mask, = solid_mask_cls().solid(1, w, h)
        mask, = feather_mask_cls().feather(mask, fearing_left, fearing_top, fearing_right, fearing_bottom)
        masks.append(mask)

    tiles = torch.cat(tiles, dim=0)
    masks = torch.cat(masks, dim=0)

    return (tiles, masks, (overlap_w, overlap_h, tile_w, tile_h, tiles_rows, tiles_cols), total)

class imageTilesFromBatch:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "tiles": ("IMAGE",),
        "masks": ("MASK",),
        "overlap": ("OVERLAP",),
        "index":("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
      },
    }

  RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
  RETURN_NAMES = ("image", "mask", "x", "y")
  FUNCTION = "doit"
  CATEGORY = "EasyUse/Image"

  def imageFromBatch(self, image, batch_index, length=1):
    s_in = image
    batch_index = min(s_in.shape[0] - 1, batch_index)
    length = min(s_in.shape[0] - batch_index, length)
    s = s_in[batch_index:batch_index + length].clone()
    return s

  def maskFromBatch(self, mask, start, length=1):
    if length > mask.shape[0]:
        length = mask.shape[0]
    start = min(start, mask.shape[0]-1)
    length = min(mask.shape[0]-start, length)
    return mask[start:start + length]

  def doit(self, tiles, masks, overlap, index):
    tile = self.imageFromBatch(tiles, index)
    mask = self.maskFromBatch(masks, index)
    overlap_w, overlap_h, tile_w, tile_h, tiles_rows, tiles_cols = overlap

    x = tile_w * (index % tiles_cols) - overlap_w if (index % tiles_cols) > 0 else 0
    y = tile_h * (index // tiles_cols) - overlap_h if tiles_rows > 1 and index > tiles_cols - 1 else 0

    return (tile, mask, x, y)



class imagesSplitImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "images": ("IMAGE",),
          }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "image5")
    FUNCTION = "split"
    CATEGORY = "EasyUse/Image"

    def split(self, images,):
      new_images = torch.chunk(images, len(images), dim=0)
      return new_images

class imageConcat:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
      "image1": ("IMAGE",),
      "image2": ("IMAGE",),
      "direction": (['right','down','left','up',],{"default": 'right'}),
      "match_image_size": ("BOOLEAN", {"default": False}),
    }}

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "concat"
  CATEGORY = "EasyUse/Image"

  def concat(self, image1, image2, direction, match_image_size):
    if image1 is None:
      return (image2,)
    elif image2 is None:
      return (image1,)
    if match_image_size:
      # Convert tensor to PIL for proper aspect ratio resizing
      pil_image2 = tensor2pil(image2)
      if direction in ['right', 'left']:
        aspect_ratio = pil_image2.width / pil_image2.height
        new_height = image1.shape[1]
        new_width = int(aspect_ratio * new_height)
        pil_image2 = fit_resize_image(pil_image2, new_width, new_height, 'fill', Image.LANCZOS, '#000000')
      else:  # 'up' or 'down'
        aspect_ratio = pil_image2.height / pil_image2.width
        new_width = image1.shape[2]
        new_height = int(aspect_ratio * new_width)
        pil_image2 = fit_resize_image(pil_image2, new_width, new_height, 'fill', Image.LANCZOS, '#000000')
      image2 = pil2tensor(pil_image2)

    if direction == 'right':
      row = torch.cat((image1, image2), dim=2)
    elif direction == 'down':
      row = torch.cat((image1, image2), dim=1)
    elif direction == 'left':
      row = torch.cat((image2, image1), dim=2)
    elif direction == 'up':
      row = torch.cat((image2, image1), dim=1)
    return (row,)

# 图片背景移除
from ..libs.utils import get_local_filepath, easySave, install_package
class imageRemBg:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "images": ("IMAGE",),
        "rem_mode": (("RMBG-2.0", "RMBG-1.4", "Inspyrenet", "BEN2"), {"default": "RMBG-1.4"}),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "optional":{
        "torchscript_jit": ("BOOLEAN", {"default": False}),
        "add_background": (["none", "white", "black"], {"default": "none"}),
        "refine_foreground": ("BOOLEAN", {"default": False}),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE", "MASK")
  RETURN_NAMES = ("image", "mask")
  FUNCTION = "remove"
  OUTPUT_NODE = True

  CATEGORY = "EasyUse/Image"


  def remove(self, rem_mode, images, image_output, save_prefix, torchscript_jit=False, add_background='none', refine_foreground=False, prompt=None, extra_pnginfo=None):
    new_images = list()
    masks = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rem_mode == "RMBG-2.0":
      if rem_mode in cache:
        _, model = cache[rem_mode][1]
      else:
        repo_id = REMBG_MODELS[rem_mode]['model_url']
        model_path = os.path.join(REMBG_DIR, 'RMBG-2.0')
        if not os.path.exists(model_path):
          from huggingface_hub import snapshot_download
          snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt"])
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        model.to(device)
        model.eval()
        update_cache(rem_mode, 'remove_background', (False, model))

      from torchvision import transforms
      transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      for image in images:
        orig_im = tensor2pil(image)
        input_image = transform_image(orig_im).unsqueeze(0).to(device)

        with torch.no_grad():
          preds = model(input_image)[-1].sigmoid().cpu()
          pred = preds[0].squeeze()

          mask = transforms.ToPILImage()(pred)
          mask = mask.resize(orig_im.size)

          new_im = orig_im.copy()
          new_im.putalpha(mask)

          new_im_tensor = pil2tensor(new_im)
          mask_tensor = pil2tensor(mask)

          new_images.append(new_im_tensor)
          masks.append(mask_tensor)

      torch.cuda.empty_cache()
      new_images = torch.cat(new_images, dim=0)
      masks = torch.cat(masks, dim=0)

    elif rem_mode == "RMBG-1.4":
      from ..modules.briaai.rembg import BriaRMBG, preprocess_image, postprocess_image
      if rem_mode in cache:
        _, net = cache[rem_mode][1]
      else:
        model_url = REMBG_MODELS[rem_mode]['model_url']
        suffix = model_url.split(".")[-1]
        model_path = get_local_filepath(model_url, REMBG_DIR, rem_mode+'.'+suffix)
        net = BriaRMBG()
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        update_cache(rem_mode, 'remove_background', (False, net))

      # prepare input
      model_input_size = [1024, 1024]
      for image in images:
        orig_im = tensor2pil(image)
        w, h = orig_im.size
        image = preprocess_image(orig_im, model_input_size).to(device)
        # inference
        result = net(image)
        result_image = postprocess_image(result[0][0], (h, w))
        mask_im = Image.fromarray(result_image)
        new_im = Image.new("RGBA", mask_im.size, (0,0,0,0))
        new_im.paste(orig_im, mask=mask_im)

        new_images.append(pil2tensor(new_im))
        masks.append(pil2tensor(mask_im))

      new_images = torch.cat(new_images, dim=0)
      masks = torch.cat(masks, dim=0)
    elif rem_mode == "BEN2":
      if rem_mode in cache:
        _, model = cache[rem_mode][1]
      else:
        from ..modules.ben.model import BEN_Base
        model_url = REMBG_MODELS[rem_mode]['model_url']
        model_path = get_local_filepath(model_url, REMBG_DIR)

        model = BEN_Base().to(device).eval()
        model.loadcheckpoints(model_path)
        update_cache(rem_mode, 'remove_background', (False, model))

      for image in images:
        input_image = tensor2pil(image)

        if input_image.mode != 'RGBA':
          input_image = input_image.convert("RGBA")

        mask, new_im = model.inference(input_image, refine_foreground)

        new_im_tensor = pil2tensor(new_im)
        mask_tensor = pil2tensor(mask)

        new_images.append(new_im_tensor)
        masks.append(mask_tensor)

      new_images = torch.cat(new_images, dim=0)
      masks = torch.cat(masks, dim=0)

    elif rem_mode == "Inspyrenet":
      from tqdm import tqdm
      try:
        from transparent_background import Remover
      except:
          install_package("transparent_background")
          from transparent_background import Remover

      if rem_mode in cache:
        _, remover = cache[rem_mode][1]
      else:
        remover = Remover(jit=torchscript_jit)
        update_cache(rem_mode, 'remove_background', (False, remover))

      for img in tqdm(images, "Inspyrenet Rembg"):
        mid = remover.process(tensor2pil(img), type='rgba')
        out = pil2tensor(mid)
        new_images.append(out)
        mask = out[:, :, :, 3]
        masks.append(mask)
      new_images = torch.cat(new_images, dim=0)
      masks = torch.cat(masks, dim=0)

    if add_background != 'none':

      _layer = tensor2pil(new_images)
      _canvas = Image.new('RGB', _layer.size, (255,255,255) if add_background == 'white' else (0, 0, 0))
      _canvas.paste(_layer, mask=_layer)
      new_images = pil2tensor(_canvas)

    results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)

    if image_output in ("Hide", "Hide/Save"):
      return {"ui": {},
              "result": (new_images, masks)}

    return {"ui": {"images": results},
            "result": (new_images, masks)}

# 图像选择器
from ..libs.chooser import wait_for_chooser
class imageChooser(PreviewImage):
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required":{
        "mode": (['Always Pause', 'Keep Last Selection'], {"default": "Always Pause"}),
      },
      "optional": {
        "images": ("IMAGE",),
      },
      "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "chooser"
  OUTPUT_NODE = True
  INPUT_IS_LIST = True
  CATEGORY = "EasyUse/Image"

  last_ic = {}
  @classmethod
  def IS_CHANGED(cls, my_unique_id, **kwargs):
    return cls.last_ic[my_unique_id[0]]

  def tensor_bundle(self, tensor_in: torch.Tensor, picks):
    if tensor_in is not None and len(picks):
      batch = tensor_in.shape[0]
      return torch.cat(tuple([tensor_in[(x) % batch].unsqueeze_(0) for x in picks])).reshape(
        [-1] + list(tensor_in.shape[1:]))
    else:
      return None

  def chooser(self, prompt=None, my_unique_id=None, extra_pnginfo=None, **kwargs):
    id = my_unique_id[0]
    id = id.split('.')[len(id.split('.')) - 1] if "." in id else id

    if (kwargs['images'] is None):
      return (None,)

    images_in = torch.cat(kwargs.pop('images'))
    for x in kwargs: kwargs[x] = kwargs[x][0]

    try:
      pnginfo = extra_pnginfo[0]
    except:
      pnginfo = None
    result = self.save_images(images=images_in, prompt=prompt, extra_pnginfo=pnginfo)
    if "ui" in result and "images" in result['ui']:
      images = result["ui"]["images"]
    else:
      images = []
    try:
      PromptServer.instance.send_sync("easyuse-image-choose", {"id": id, "urls": images})
    except Exception as e:
      pass

    # 获取上次选择
    mode = kwargs.pop('mode', 'Always Pause')
    return wait_for_chooser(id, images_in, mode)

class imageColorMatch(PreviewImage):
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image_ref": ("IMAGE",),
        "image_target": ("IMAGE",),
        "method": (['wavelet', 'adain', 'mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm'],),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  CATEGORY = "EasyUse/Image"

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  OUTPUT_NODE = True
  FUNCTION = "color_match"

  def color_match(self, image_ref, image_target, method, image_output, save_prefix, prompt=None, extra_pnginfo=None):
    if method in ["wavelet", "adain"]:
      result_images = wavelet_color_fix(tensor2pil(image_target), tensor2pil(image_ref)) if method == 'wavelet' else adain_color_fix(tensor2pil(image_target), tensor2pil(image_ref))
      new_images = pil2tensor(result_images)
    else:
      try:
        from color_matcher import ColorMatcher
      except:
        install_package("color-matcher")
        from color_matcher import ColorMatcher
      image_ref = image_ref.cpu()
      image_target = image_target.cpu()
      batch_size = image_target.size(0)
      out = []
      images_target = image_target.squeeze()
      images_ref = image_ref.squeeze()

      image_ref_np = images_ref.numpy()
      images_target_np = images_target.numpy()
      if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
        raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")
      cm = ColorMatcher()
      for i in range(batch_size):
        image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
        image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
        try:
          image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
        except BaseException as e:
          print(f"Error occurred during transfer: {e}")
          break
        out.append(torch.from_numpy(image_result))

      new_images = torch.stack(out, dim=0).to(torch.float32)

    results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)

    if image_output in ("Hide", "Hide/Save"):
      return {"ui": {},
              "result": (new_images,)}

    return {"ui": {"images": results},
            "result": (new_images,)}

class imageDetailTransfer:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "target": ("IMAGE",),
        "source": ("IMAGE",),
        "mode": (["add", "multiply", "screen", "overlay", "soft_light", "hard_light", "color_dodge", "color_burn", "difference", "exclusion", "divide",],{"default": "add"}),
        "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
        "blend_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001, "round": 0.001}),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "optional": {
        "mask": ("MASK",),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  OUTPUT_NODE = True
  FUNCTION = "transfer"
  CATEGORY = "EasyUse/Image"



  def transfer(self, target, source, mode, blur_sigma, blend_factor, image_output, save_prefix, mask=None, prompt=None, extra_pnginfo=None):
    batch_size, height, width, _ = source.shape
    device = comfy.model_management.get_torch_device()
    target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
    source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

    if target.shape[1:] != source.shape[1:]:
      target_tensor = comfy.utils.common_upscale(target_tensor, width, height, "bilinear", "disabled")
    if mask is not None and target.shape[1:] != mask.shape[1:]:
      mask = mask.unsqueeze(1)
      mask = F.interpolate(mask, size=(height, width), mode="bilinear")
      mask = mask.squeeze(1)

    if source.shape[0] < batch_size:
      source = source[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    kernel_size = int(6 * int(blur_sigma) + 1)

    gaussian_blur = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))

    blurred_target = gaussian_blur(target_tensor)
    blurred_source = gaussian_blur(source_tensor)

    if mode == "add":
      new_image = (source_tensor - blurred_source) + blurred_target
    elif mode == "multiply":
      new_image = source_tensor * blurred_target
    elif mode == "screen":
      new_image = 1 - (1 - source_tensor) * (1 - blurred_target)
    elif mode == "overlay":
      new_image = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "soft_light":
      new_image = (1 - 2 * blurred_target) * source_tensor ** 2 + 2 * blurred_target * source_tensor
    elif mode == "hard_light":
      new_image = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "difference":
      new_image = torch.abs(blurred_target - source_tensor)
    elif mode == "exclusion":
      new_image = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
    elif mode == "color_dodge":
      new_image = blurred_target / (1 - source_tensor)
    elif mode == "color_burn":
      new_image = 1 - (1 - blurred_target) / source_tensor
    elif mode == "divide":
      new_image = (source_tensor / blurred_source) * blurred_target
    else:
      new_image = source_tensor

    new_image = torch.lerp(target_tensor, new_image, blend_factor)
    if mask is not None:
      mask = mask.to(device)
      new_image = torch.lerp(target_tensor, new_image, mask)
    new_image = torch.clamp(new_image, 0, 1)
    new_image = new_image.permute(0, 2, 3, 1).cpu().float()

    results = easySave(new_image, save_prefix, image_output, prompt, extra_pnginfo)

    if image_output in ("Hide", "Hide/Save"):
      return {"ui": {},
              "result": (new_image,)}

    return {"ui": {"images": results},
            "result": (new_image,)}

# 图像反推
from ..libs.image import ci
class imageInterrogator:
    @classmethod
    def INPUT_TYPES(self):
        return {
          "required": {
              "image": ("IMAGE",),
              "mode": (['fast','classic','best','negative'],),
              "use_lowvram": ("BOOLEAN", {"default": True}),
          }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "interrogate"
    CATEGORY = "EasyUse/Image"
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = (True,)

    def interrogate(self, image, mode, use_lowvram=False):
      prompt = ci.image_to_prompt(image, mode, low_vram=use_lowvram)
      return (prompt,)

# 人类分割器
class humanSegmentation:

    @classmethod
    def INPUT_TYPES(cls):
        return {
          "required":{
            "image": ("IMAGE",),
            "method": (["selfie_multiclass_256x256", "human_parsing_lip", "human_parts (deeplabv3p)", "segformer_b3_clothes", "segformer_b3_fashion", "face_parsing"],),
            "confidence": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01},),
            "crop_multi": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001},),
            "mask_components":(
               "EASY_COMBO",{
                 "options": [{'label':'Background','value':0}],
                 "multi_select": {
                   "placeholder": "select mask components",
                   "chip": True,
                   "max_selected_labels": 4,
                 }
               }
            )
          },
          "hidden": {
              "prompt": "PROMPT",
              "my_unique_id": "UNIQUE_ID",
          }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX")
    RETURN_NAMES = ("image", "mask", "bbox")
    FUNCTION = "parsing"
    CATEGORY = "EasyUse/Segmentation"

    def get_mediapipe_image(self, image: Image):
      import mediapipe as mp
      # Convert image to NumPy array
      numpy_image = np.asarray(image)
      image_format = mp.ImageFormat.SRGB
      # Convert BGR to RGB (if necessary)
      if numpy_image.shape[-1] == 4:
        image_format = mp.ImageFormat.SRGBA
      elif numpy_image.shape[-1] == 3:
        image_format = mp.ImageFormat.SRGB
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
      return mp.Image(image_format=image_format, data=numpy_image)

    def parsing(self, image, confidence, method, crop_multi, mask_components, prompt=None, my_unique_id=None):
      if method == 'selfie_multiclass_256x256':
        try:
          import mediapipe as mp
        except:
          install_package("mediapipe")
          import mediapipe as mp

        from functools import reduce

        if method in cache:
          _, model_asset_buffer = cache["selfie_multiclass_256x256"][1]
        else:
          model_path = get_local_filepath(MEDIAPIPE_MODELS['selfie_multiclass_256x256']['model_url'], MEDIAPIPE_DIR)
          model_asset_buffer = None
          with open(model_path, "rb") as f:
              model_asset_buffer = f.read()
          update_cache(method, 'human_segmentation', (False, model_asset_buffer))
        image_segmenter_base_options = mp.tasks.BaseOptions(model_asset_buffer=model_asset_buffer)
        options = mp.tasks.vision.ImageSegmenterOptions(
          base_options=image_segmenter_base_options,
          running_mode=mp.tasks.vision.RunningMode.IMAGE,
          output_category_mask=True)
        # Create the image segmenter
        ret_images = []
        ret_masks = []

        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            for img in image:
                _image = torch.unsqueeze(img, 0)
                orig_image = tensor2pil(_image).convert('RGB')
                # Convert the Tensor to a PIL image
                i = 255. * img.cpu().numpy()
                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                # create our foreground and background arrays for storing the mask results
                mask_background_array = np.zeros((image_pil.size[0], image_pil.size[1], 4), dtype=np.uint8)
                mask_background_array[:] = (0, 0, 0, 255)
                mask_foreground_array = np.zeros((image_pil.size[0], image_pil.size[1], 4), dtype=np.uint8)
                mask_foreground_array[:] = (255, 255, 255, 255)
                # Retrieve the masks for the segmented image
                media_pipe_image = self.get_mediapipe_image(image=image_pil)
                segmented_masks = segmenter.segment(media_pipe_image)
                masks = []
                for i, com in enumerate(mask_components):
                    masks.append(segmented_masks.confidence_masks[com])

                image_data = media_pipe_image.numpy_view()
                image_shape = image_data.shape
                # convert the image shape from "rgb" to "rgba" aka add the alpha channel
                if image_shape[-1] == 3:
                    image_shape = (image_shape[0], image_shape[1], 4)
                mask_background_array = np.zeros(image_shape, dtype=np.uint8)
                mask_background_array[:] = (0, 0, 0, 255)
                mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
                mask_foreground_array[:] = (255, 255, 255, 255)
                mask_arrays = []
                if len(masks) == 0:
                    mask_arrays.append(mask_background_array)
                else:
                    for i, mask in enumerate(masks):
                        condition = np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1) > confidence
                        mask_array = np.where(condition, mask_foreground_array, mask_background_array)
                        mask_arrays.append(mask_array)
                # Merge our masks taking the maximum from each
                merged_mask_arrays = reduce(np.maximum, mask_arrays)
                # Create the image
                mask_image = Image.fromarray(merged_mask_arrays)
                # convert PIL image to tensor image
                tensor_mask = mask_image.convert("RGB")
                tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
                tensor_mask = torch.from_numpy(tensor_mask)[None,]
                _mask = tensor_mask.squeeze(3)[..., 0]

                _mask = tensor2pil(tensor_mask).convert('L')

                ret_image = RGB2RGBA(orig_image, _mask)
                ret_images.append(pil2tensor(ret_image))
                ret_masks.append(image2mask(_mask))

            output_image = torch.cat(ret_images, dim=0)
            mask = torch.cat(ret_masks, dim=0)

      elif method == "human_parsing_lip":
        if method in cache:
          _, parsing = cache[method][1]
        else:
          from ..modules.human_parsing.run_parsing import HumanParsing
          onnx_path = os.path.join(folder_paths.models_dir, 'onnx')
          model_path = get_local_filepath(HUMANPARSING_MODELS['parsing_lip']['model_url'], onnx_path)
          parsing = HumanParsing(model_path=model_path)
          update_cache(method, 'human_segmentation', (False, parsing))

        model_image = image.squeeze(0)
        model_image = model_image.permute((2, 0, 1))
        model_image = to_pil_image(model_image)

        map_image, mask = parsing(model_image, mask_components)

        mask = mask[:, :, :, 0]

        alpha = 1.0 - mask

        output_image, = JoinImageWithAlpha().join_image_with_alpha(image, alpha)

      elif method == "human_parts (deeplabv3p)":
        if method in cache:
          _, parsing = cache[method][1]
        else:
          from ..modules.human_parsing.run_parsing import HumanParts
          onnx_path = os.path.join(folder_paths.models_dir, 'onnx')
          human_parts_path = os.path.join(onnx_path, 'human-parts')
          model_path = get_local_filepath(HUMANPARSING_MODELS['human-parts']['model_url'], human_parts_path)
          parsing = HumanParts(model_path=model_path)
          update_cache(method, 'human_segmentation', (False, parsing))

        ret_images = []
        ret_masks = []
        for img in image:
          mask, = parsing(img, mask_components)
          _mask = tensor2pil(mask).convert('L')

          ret_image = RGB2RGBA(tensor2pil(img).convert('RGB'), _mask.convert('L'))
          ret_images.append(pil2tensor(ret_image))
          ret_masks.append(image2mask(_mask))

        output_image = torch.cat(ret_images, dim=0)
        mask = torch.cat(ret_masks, dim=0)

      elif method in ["segformer_b3_clothes", "segformer_b3_fashion", "face_parsing"]:
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

        # 分割
        def get_segmentation_from_model(tensor_image, model, processor):
          cloth = tensor2pil(tensor_image)
          inputs = processor(images=cloth, return_tensors="pt")
          outputs = model(**inputs)
          logits = outputs.logits.cpu()
          upsampled_logits = F.interpolate(logits, size=cloth.size[::-1], mode="bilinear",
                                                       align_corners=False)
          pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
          return pred_seg, cloth


        if method in cache:
          _, (processor, model) = cache[method][1]
        else:
          model_folder_path = os.path.join(folder_paths.models_dir, method)
          if os.path.exists(model_folder_path):
            print(f"Start to load existing model...")
          else:
            from huggingface_hub import snapshot_download
            PromptServer.instance.send_sync("easyuse-toast", {"content": f"Model not found locally. Downloading {method}...", "type": 'loading', "duration": 10000})
            print(f"Model not found locally. Downloading {method}...")
            model_path_cache = os.path.join(folder_paths.models_dir, "cache-"+method)
            snapshot_download(
              repo_id=HUMANPARSING_MODELS[method]['model_name'],
              local_dir=model_path_cache,
              local_dir_use_symlinks=False,
              resume_download=True
            )
            shutil.move(model_path_cache, model_folder_path)
            print(f"Model downloaded to {model_folder_path}...")
          try:
            model_folder_path = os.path.normpath(folder_paths.folder_names_and_paths[method][0][0])
          except:
            pass

          processor = SegformerImageProcessor.from_pretrained(model_folder_path)
          model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)
          update_cache(method, 'human_segmentation', (False, (processor, model)))

        ret_images = []
        ret_masks = []

        if method == "face_parsing":
          import matplotlib
          import torchvision.transforms as T
          transform = ToPILImage()
          colormap = matplotlib.colormaps['viridis']
          device = model.device
          results = []
          images = []
          for img in image:
            size = img.shape[:2]
            inputs = processor(images=transform(img.permute(2, 0, 1)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
              logits,
              size=size,
              mode="bilinear",
              align_corners=False)

            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_seg_np = pred_seg.cpu().detach().numpy().astype(np.uint8)
            results.append(torch.tensor(pred_seg_np))

          results_out = torch.stack(results, dim=0)
          for img, result_item in zip(image, results_out):
              mask = torch.zeros(result_item.shape, dtype=torch.uint8)
              for i in mask_components:
                  mask = mask | torch.where(result_item == i, 1, 0)

              # 将mask转换为numpy数组，并确保数据类型正确
              mask_np = (mask * 255).numpy().astype(np.uint8)
              _mask = Image.fromarray(mask_np)

              # 处理图像输出
              ret_image = RGB2RGBA(tensor2pil(img).convert('RGB'), _mask.convert('L'))
              ret_images.append(pil2tensor(ret_image))
              ret_masks.append(image2mask(_mask))

        else:
          for img in image:
              pred_seg, cloth = get_segmentation_from_model(img, model, processor)
              i = torch.unsqueeze(img, 0)
              i = pil2tensor(tensor2pil(i).convert('RGB'))

              mask = np.isin(pred_seg, mask_components).astype(np.uint8)
              _mask = Image.fromarray(mask * 255)

              ret_image = RGB2RGBA(tensor2pil(img).convert('RGB'), _mask.convert('L'))
              ret_images.append(pil2tensor(ret_image))
              ret_masks.append(image2mask(_mask))

        output_image = torch.cat(ret_images, dim=0)
        mask = torch.cat(ret_masks, dim=0)

      # use crop
      bbox = [[0, 0, 0, 0]]
      if crop_multi > 0.0:
        output_image, mask, bbox = imageCropFromMask().crop(output_image, mask, crop_multi, crop_multi, 1.0)

      return (output_image, mask, bbox)

class imageCropFromMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "image": ("IMAGE",),
              "mask": ("MASK",),
              "image_crop_multi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
              "mask_crop_multi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
              "bbox_smooth_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
          },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX",)
    RETURN_NAMES = ("crop_image", "crop_mask", "bbox",)
    FUNCTION = "crop"
    CATEGORY = "EasyUse/Image"

    def smooth_bbox_size(self, prev_bbox_size, curr_bbox_size, alpha):
        if alpha == 0:
            return prev_bbox_size
        return round(alpha * curr_bbox_size + (1 - alpha) * prev_bbox_size)

    def smooth_center(self, prev_center, curr_center, alpha=0.5):
        if alpha == 0:
            return prev_center
        return (
            round(alpha * curr_center[0] + (1 - alpha) * prev_center[0]),
            round(alpha * curr_center[1] + (1 - alpha) * prev_center[1])
        )

    def image2mask(self, image):
      return image[:, :, :, 0]

    def mask2image(self, mask):
      return mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

    def cropimage(self, original_images, masks, crop_size_mult, bbox_smooth_alpha):

      bounding_boxes = []
      cropped_images = []

      self.max_bbox_width = 0
      self.max_bbox_height = 0

      # First, calculate the maximum bounding box size across all masks
      curr_max_bbox_width = 0
      curr_max_bbox_height = 0
      for mask in masks:
        _mask = tensor2pil(mask)
        non_zero_indices = np.nonzero(np.array(_mask))
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        width = max_x - min_x
        height = max_y - min_y
        curr_max_bbox_width = max(curr_max_bbox_width, width)
        curr_max_bbox_height = max(curr_max_bbox_height, height)

      # Smooth the changes in the bounding box size
      self.max_bbox_width = self.smooth_bbox_size(self.max_bbox_width, curr_max_bbox_width, bbox_smooth_alpha)
      self.max_bbox_height = self.smooth_bbox_size(self.max_bbox_height, curr_max_bbox_height, bbox_smooth_alpha)

      # Apply the crop size multiplier
      self.max_bbox_width = round(self.max_bbox_width * crop_size_mult)
      self.max_bbox_height = round(self.max_bbox_height * crop_size_mult)
      bbox_aspect_ratio = self.max_bbox_width / self.max_bbox_height

      # Then, for each mask and corresponding image...
      for i, (mask, img) in enumerate(zip(masks, original_images)):
        _mask = tensor2pil(mask)
        non_zero_indices = np.nonzero(np.array(_mask))
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

        # Calculate center of bounding box
        center_x = np.mean(non_zero_indices[1])
        center_y = np.mean(non_zero_indices[0])
        curr_center = (round(center_x), round(center_y))

        # If this is the first frame, initialize prev_center with curr_center
        if not hasattr(self, 'prev_center'):
          self.prev_center = curr_center

        # Smooth the changes in the center coordinates from the second frame onwards
        if i > 0:
          center = self.smooth_center(self.prev_center, curr_center, bbox_smooth_alpha)
        else:
          center = curr_center

        # Update prev_center for the next frame
        self.prev_center = center

        # Create bounding box using max_bbox_width and max_bbox_height
        half_box_width = round(self.max_bbox_width / 2)
        half_box_height = round(self.max_bbox_height / 2)
        min_x = max(0, center[0] - half_box_width)
        max_x = min(img.shape[1], center[0] + half_box_width)
        min_y = max(0, center[1] - half_box_height)
        max_y = min(img.shape[0], center[1] + half_box_height)

        # Append bounding box coordinates
        bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

        # Crop the image from the bounding box
        cropped_img = img[min_y:max_y, min_x:max_x, :]

        # Calculate the new dimensions while maintaining the aspect ratio
        new_height = min(cropped_img.shape[0], self.max_bbox_height)
        new_width = round(new_height * bbox_aspect_ratio)

        # Resize the image
        resize_transform = Resize((new_height, new_width))
        resized_img = resize_transform(cropped_img.permute(2, 0, 1))

        # Perform the center crop to the desired size
        crop_transform = CenterCrop((self.max_bbox_height, self.max_bbox_width))  # swap the order here if necessary
        cropped_resized_img = crop_transform(resized_img)

        cropped_images.append(cropped_resized_img.permute(1, 2, 0))

      return cropped_images, bounding_boxes

    def crop(self, image, mask, image_crop_multi, mask_crop_multi, bbox_smooth_alpha):
      cropped_images, bounding_boxes = self.cropimage(image, mask, image_crop_multi, bbox_smooth_alpha)
      cropped_mask_image, _ = self.cropimage(self.mask2image(mask), mask, mask_crop_multi, bbox_smooth_alpha)

      cropped_image_out = torch.stack(cropped_images, dim=0)
      cropped_mask_out = torch.stack(cropped_mask_image, dim=0)

      return (cropped_image_out, cropped_mask_out[:, :, :, 0], bounding_boxes)


class imageUncropFromBBOX:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "original_image": ("IMAGE",),
              "crop_image": ("IMAGE",),
              "bbox": ("BBOX",),
              "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},),
              "use_square_mask": ("BOOLEAN", {"default": True}),
          },
          "optional":{
            "optional_mask": ("MASK",)
          }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "uncrop"
    CATEGORY = "EasyUse/Image"

    def bbox_check(self, bbox, target_size=None):
      if not target_size:
        return bbox

      new_bbox = (
        bbox[0],
        bbox[1],
        min(target_size[0] - bbox[0], bbox[2]),
        min(target_size[1] - bbox[1], bbox[3]),
      )
      return new_bbox

    def bbox_to_region(self, bbox, target_size=None):
      bbox = self.bbox_check(bbox, target_size)
      return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

    def uncrop(self, original_image, crop_image, bbox, border_blending, use_square_mask, optional_mask=None):
      def inset_border(image, border_width=20, border_color=(0)):
        width, height = image.size
        bordered_image = Image.new(image.mode, (width, height), border_color)
        bordered_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(bordered_image)
        draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
        return bordered_image

      if len(original_image) != len(crop_image):
        raise ValueError(
          f"The number of original_images ({len(original_image)}) and cropped_images ({len(crop_image)}) should be the same")

        # Ensure there are enough bboxes, but drop the excess if there are more bboxes than images
      if len(bbox) > len(original_image):
        print(f"Warning: Dropping excess bounding boxes. Expected {len(original_image)}, but got {len(bbox)}")
        bbox = bbox[:len(original_image)]
      elif len(bbox) < len(original_image):
        raise ValueError("There should be at least as many bboxes as there are original and cropped images")


      out_images = []

      for i in range(len(original_image)):
        img = tensor2pil(original_image[i])
        crop = tensor2pil(crop_image[i])
        _bbox = bbox[i]

        bb_x, bb_y, bb_width, bb_height = _bbox
        paste_region = self.bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
        
        # rescale the crop image to fit the paste_region
        crop = crop.resize((round(paste_region[2] - paste_region[0]), round(paste_region[3] - paste_region[1])))
        crop_img = crop.convert("RGB")

        # border blending
        if border_blending > 1.0:
          border_blending = 1.0
        elif border_blending < 0.0:
          border_blending = 0.0

        blend_ratio = (max(crop_img.size) / 2) * float(border_blending)
        blend = img.convert("RGBA")

        if use_square_mask:
          mask = Image.new("L", img.size, 0)
          mask_block = Image.new("L", (paste_region[2] - paste_region[0], paste_region[3] - paste_region[1]), 255)
          mask_block = inset_border(mask_block, round(blend_ratio / 2), (0))
          mask.paste(mask_block, paste_region)
        else:
          if optional_mask is None:
            raise ValueError("optional_mask is required when use_square_mask is False")
          original_mask = tensor2pil(optional_mask)
          original_mask = original_mask.resize((paste_region[2] - paste_region[0], paste_region[3] - paste_region[1]))
          mask = Image.new("L", img.size, 0)
          mask.paste(original_mask, paste_region)

        mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

        blend.paste(crop_img, paste_region)
        blend.putalpha(mask)

        img = Image.alpha_composite(img.convert("RGBA"), blend)
        out_images.append(img.convert("RGB"))

      output_images = torch.cat([pil2tensor(img) for img in out_images], dim=0)
      return (output_images,)



import cv2
import base64
class loadImageBase64:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "base64_data": ("STRING", {"default": ""}),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "optional": {

      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE", "MASK")
  OUTPUT_NODE = True
  FUNCTION = "load_image"
  CATEGORY = "EasyUse/Image/LoadImage"

  def convert_color(self, image,):
    if len(image.shape) > 2 and image.shape[2] >= 4:
      return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  def load_image(self, base64_data, image_output, save_prefix, prompt=None, extra_pnginfo=None):
    nparr = np.frombuffer(base64.b64decode(base64_data), np.uint8)

    result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(result)
    if len(channels) > 3:
      mask = channels[3].astype(np.float32) / 255.0
      mask = torch.from_numpy(mask)
    else:
      mask = torch.ones(channels[0].shape, dtype=torch.float32, device="cpu")

    result = self.convert_color(result)
    result = result.astype(np.float32) / 255.0
    new_images = torch.from_numpy(result)[None,]

    results = easySave(new_images, save_prefix, image_output, None, None)
    mask = mask.unsqueeze(0)

    if image_output in ("Hide", "Hide/Save"):
      return {"ui": {},
              "result": (new_images, mask)}

    return {"ui": {"images": results},
            "result": (new_images, mask)}

class imageToBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "image": ("IMAGE",),
        },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_base64"
    CATEGORY = "EasyUse/Image"
    OUTPUT_NODE = True

    def to_base64(self, image, ):
      import base64
      from io import BytesIO

      # 将张量图像转换为PIL图像
      pil_image = tensor2pil(image)

      buffered = BytesIO()
      pil_image.save(buffered, format="PNG")
      image_bytes = buffered.getvalue()

      base64_str = base64.b64encode(image_bytes).decode("utf-8")
      return {"result": (base64_str,)}

class removeLocalImage:

  def __init__(self):
    self.hasFile = False

  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "any": (any_type,),
          "file_name": ("STRING",{"default":""}),
        },
      }

  RETURN_TYPES = ()
  OUTPUT_NODE = True
  FUNCTION = "remove"
  CATEGORY = "EasyUse/Image"



  def remove(self, any, file_name):
    self.hasFile = False
    def listdir(path, dir_name=''):
      for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
          dir_name = os.path.basename(file_path)
          listdir(file_path, dir_name)
        else:
          file = os.path.join(dir_name, file)
          name_without_extension, file_extension = os.path.splitext(file)
          if name_without_extension == file_name or file == file_name:
            os.remove(os.path.join(folder_paths.input_directory, file))
            self.hasFile = True
            break

    listdir(folder_paths.input_directory, '')

    if self.hasFile:
      PromptServer.instance.send_sync("easyuse-toast", {"content": "Removed SuccessFully", "type":'success'})
    else:
      PromptServer.instance.send_sync("easyuse-toast", {"content": "Removed Failed", "type": 'error'})
    return ()

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except:
    GraphBuilder = None
class loadImagesForLoop:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "directory": ("STRING", {"default": ""}),
      },
      "optional": {
        "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
        "limit": ("INT", {"default":-1, "min":-1, "max": 10000}),
        "initial_value1": (any_type,),
        "initial_value2": (any_type,),
      },
      "hidden": {
        "initial_value0": (any_type,),
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO",
        "unique_id": "UNIQUE_ID"
      }
    }

  RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL", "INT", "IMAGE", "MASK", "STRING", any_type, any_type]))
  RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index", "image", "mask", "name", "value1", "value2"]))

  FUNCTION = "load_images"

  CATEGORY = "image"

  def load_images(self, directory: str, start_index: int = 0, limit: int =-1, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
    print(directory)
    if not os.path.isdir(directory):
      raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

    dir_files = os.listdir(directory)
    if len(dir_files) == 0:
      raise FileNotFoundError(f"No files in directory '{directory}'.")

    # Filter files by extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]

    graph = GraphBuilder()
    index = 0
    # unique_id = unique_id.split('.')[len(unique_id.split('.')) - 1] if "." in unique_id else unique_id
    # update_cache('forloop' + str(unique_id), 'forloop', total)
    if "initial_value0" in kwargs:
      index = kwargs["initial_value0"]
    # start at start_index
    image_path = dir_files[start_index+index]

    name = os.path.splitext(os.path.basename(image_path))[0]

    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    if 'A' in i.getbands():
      mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
      mask = 1. - torch.from_numpy(mask)
    else:
      mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    while_open = graph.node("easy whileLoopStart", condition=True, initial_value0=index, initial_value1=kwargs.get('initial_value1',None), initial_value2=kwargs.get('initial_value2',None))
    outputs = [kwargs.get('initial_value1',None), kwargs.get('initial_value2',None)]

    return {
      "result": tuple(["stub", index, image, mask, name] + outputs),
      "expand": graph.finalize(),
    }

class makeImageForICRepaint:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image_1": ("IMAGE",),
        "direction": (["top-bottom", "left-right"], {"default": "left-right"}),
        "pixels": ("INT", {"default": 0, "max": MAX_RESOLUTION, "min": 0, "step": 8, "tooltip": "The pixel of the output image is not set when it is 0"}),
        "method": (["uniform height", "uniform width", "auto"],{"default": "auto"}),
      },
      "optional": {
        "image_2": ("IMAGE",),
        "mask_1": ("MASK",),
        "mask_2": ("MASK",),
      },
    }

  DESCRIPTION = "make Image for ICLora to Re-paint"
  CATEGORY = "EasyUse/Image"
  FUNCTION = "make"

  RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
  RETURN_NAMES = ("image", "mask", "context_mask", "width", "height", "x", "y")

  def fillMask(self, width, height, mask, box=(0, 0), color=0):
    bg = Image.new("L", (width, height), color)
    bg.paste(mask, box, mask)
    return bg

  def emptyImage(self, width, height, batch_size=1, color=0):
    r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
    g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
    b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
    return torch.cat((r, g, b), dim=-1)

  def resize_image_and_mask(self, image, mask, w, h ,fit='fill'):
      ret_images = []
      ret_masks = []
      _mask = Image.new('L', size=(w, h), color='black')
      _image = Image.new('RGB', size=(w, h), color='black')
      if image is not None and len(image) > 0:
          for i in image:
              _image = tensor2pil(i).convert('RGB')
              _image = fit_resize_image(_image, w, h, fit, Image.LANCZOS, '#000000')
              ret_images.append(pil2tensor(_image))
      if mask is not None and len(mask) > 0:
          for m in mask:
              _mask = tensor2pil(m).convert('L')
              _mask = fit_resize_image(_mask, w, h, fit, Image.LANCZOS).convert('L')
              ret_masks.append(image2mask(_mask))

      if len(ret_images) > 0 and len(ret_masks) > 0:
          return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)
      elif len(ret_images) > 0 and len(ret_masks) == 0:
          return (torch.cat(ret_images, dim=0), None,)
      elif len(ret_images) == 0 and len(ret_masks) > 0:
          return (None, torch.cat(ret_masks, dim=0),)
      else:
        return (None, None)

  def make(self, image_1, direction, pixels, method, image_2=None, mask_1=None, mask_2=None):
    if image_2 is None:
      image_2 = self.emptyImage(image_1.shape[2], image_1.shape[1])
      mask_2 = torch.full((1, image_1.shape[1], image_1.shape[2]), 1, dtype=torch.float32, device="cpu")

    elif image_2 is not None and mask_2 is None:
        mask_2 = torch.full((1, image_2.shape[1], image_2.shape[2]), 1, dtype=torch.float32, device="cpu")

    if pixels > 0:
      _, img2_h, img2_w, _ = image_2.shape
      if method == "uniform height":
          h = pixels
          w = int(img2_w * (pixels / img2_h))
      elif method == "uniform width":
          w = pixels
          h = int(img2_h * (pixels / img2_w))
      else:
          h = pixels if direction == 'left-right' else int(img2_h * (pixels / img2_w))
          w = pixels if direction == 'top-bottom' else int(img2_w * (pixels / img2_h))

      image_2, mask_2 = self.resize_image_and_mask(image_2, mask_2, w, h)

    _, img1_h, img1_w, _ = image_1.shape
    _, img2_h, img2_w, _ = image_2.shape

    image, mask, context_mask = None, None, None

    # resize
    if img1_h != img2_h and img1_w != img2_w:
      width, height = img2_w, img2_h
      fit = 'crop'
      if method != 'uniform width':
        if direction == 'left-right' and img1_h != img2_h:
          scale_factor = img2_h / img1_h
          width = round(img1_w * scale_factor)
        elif direction == 'top-bottom' and img1_w != img2_w:
          scale_factor = img2_w / img1_w
          height = round(img1_h * scale_factor)
        fit = 'fill'
      image_1, mask_1 = self.resize_image_and_mask(image_1, mask_1, width, height, fit)

    if mask_1 is None:
      mask_1 = torch.full((1, image_1.shape[1], image_1.shape[2]), 0, dtype=torch.float32, device="cpu")

    orig_image_1 = tensor2pil(image_1)
    orig_mask_1 = tensor2pil(mask_1).convert('L')

    if orig_mask_1.size != orig_image_1.size:
      orig_mask_1 = orig_mask_1.resize(orig_image_1.size)

    img1_w, img1_h = orig_image_1.size
    image_1 = pil2tensor(orig_image_1)
    image = torch.cat((image_1, image_2), dim=2) if direction == 'left-right' else torch.cat((image_1, image_2),
                                                                                             dim=1)

    context_mask = self.fillMask(image.shape[2], image.shape[1], orig_mask_1)
    context_mask = pil2tensor(context_mask)

    orig_mask_2 = tensor2pil(mask_2).convert('L')
    x = img1_w if direction == 'left-right' else 0
    y = img1_h if direction == 'top-bottom' else 0
    mask = self.fillMask(image.shape[2], image.shape[1], orig_mask_2, (x, y))
    mask = pil2tensor(mask)

    return (image, mask, context_mask, img2_w, img2_h, x, y)


NODE_CLASS_MAPPINGS = {
  "easy imageInsetCrop": imageInsetCrop,
  "easy imageCount": imageCount,
  "easy imagesCountInDirectory": imagesCountInDirectory,
  "easy imageSize": imageSize,
  "easy imageSizeBySide": imageSizeBySide,
  "easy imageSizeByLongerSide": imageSizeByLongerSide,
  "easy imagePixelPerfect": imagePixelPerfect,
  "easy imageScaleDown": imageScaleDown,
  "easy imageScaleDownBy": imageScaleDownBy,
  "easy imageScaleDownToSize": imageScaleDownToSize,
  "easy imageScaleToNormPixels": imageScaleToNormPixels,
  "easy imageRatio": imageRatio,
  "easy imageConcat": imageConcat,
  "easy imageListToImageBatch": imageListToImageBatch,
  "easy imageBatchToImageList": imageBatchToImageList,
  "easy imageSplitList": imageSplitList,
  "easy imageSplitGrid": imageSplitGrid,
  "easy imagesSplitImage": imagesSplitImage,
  "easy imageSplitTiles": imageSplitTiles,
  "easy imageTilesFromBatch": imageTilesFromBatch,
  "easy imageCropFromMask": imageCropFromMask,
  "easy imageUncropFromBBOX": imageUncropFromBBOX,
  "easy imageSave": imageSaveSimple,
  "easy imageRemBg": imageRemBg,
  "easy imageChooser": imageChooser,
  "easy imageColorMatch": imageColorMatch,
  "easy imageDetailTransfer": imageDetailTransfer,
  "easy imageInterrogator": imageInterrogator,
  "easy loadImagesForLoop": loadImagesForLoop,
  "easy loadImageBase64": loadImageBase64,
  "easy imageToBase64": imageToBase64,
  "easy joinImageBatch": JoinImageBatch,
  "easy humanSegmentation": humanSegmentation,
  "easy removeLocalImage": removeLocalImage,
  "easy makeImageForICLora": makeImageForICRepaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "easy imageInsetCrop": "ImageInsetCrop",
  "easy imageCount": "ImageCount",
  "easy imagesCountInDirectory": "imagesCountInDirectory",
  "easy imageSize": "ImageSize",
  "easy imageSizeBySide": "ImageSize (Side)",
  "easy imageSizeByLongerSide": "ImageSize (LongerSide)",
  "easy imagePixelPerfect": "ImagePixelPerfect",
  "easy imageScaleDown": "Image Scale Down",
  "easy imageScaleDownBy": "Image Scale Down By",
  "easy imageScaleDownToSize": "Image Scale Down To Size",
  "easy imageScaleToNormPixels": "ImageScaleToNormPixels",
  "easy imageRatio": "ImageRatio",
  "easy imageHSVMask": "ImageHSVMask",
  "easy imageConcat": "imageConcat",
  "easy imageListToImageBatch": "Image List To Image Batch",
  "easy imageBatchToImageList": "Image Batch To Image List",
  "easy imageSplitList": "imageSplitList",
  "easy imageSplitGrid": "imageSplitGrid",
  "easy imageSplitTiles": "imageSplitTiles",
  "easy imageTilesFromBatch": "imageTilesFromBatch",
  "easy imagesSplitImage": "imagesSplitImage",
  "easy imageCropFromMask": "imageCropFromMask",
  "easy imageUncropFromBBOX": "imageUncropFromBBOX",
  "easy imageSave": "Save Image (Simple)",
  "easy imageRemBg": "Image Remove Bg",
  "easy imageChooser": "Image Chooser",
  "easy imageColorMatch": "Image Color Match",
  "easy imageDetailTransfer": "Image Detail Transfer",
  "easy imageInterrogator": "Image To Prompt",
  "easy joinImageBatch": "JoinImageBatch",
  "easy loadImageBase64": "Load Image (Base64)",
  "easy loadImagesForLoop": "Load Images For Loop",
  "easy imageToBase64": "Image To Base64",
  "easy humanSegmentation": "Human Segmentation",
  "easy removeLocalImage": "Remove Local Image",
  "easy makeImageForICLora": "Make Image For ICLora"
}