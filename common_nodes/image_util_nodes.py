

import torch
import numpy as np

from nodes import MAX_RESOLUTION
from nodes import ImageScale
from framework.app_log import AppLog
import folder_paths
from framework.image_util import ImageUtil



class ConstrainImageMaxSize:
    @classmethod
    def INPUT_TYPES(s):
        upscale_methods = ImageScale.upscale_methods
        return {
            "required": {
                "image": ("IMAGE",),
                "max_width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "max_height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "upscale_method": (upscale_methods, ),
            }
        }
        
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "aiyoh"
    
    
    def execute(self, image, max_width, max_height, upscale_method):
        d1, imgh, imgw, imgd = image.size()
        
        f_h = imgh / float(max_height)
        f_w = imgw / float(max_width)
        if f_h >= f_w and imgh > max_height:
            # resize
            tar_w = int(imgw / f_h)
            tar_h = max_height
            image = ImageScale().upscale(image, width=tar_w, height=tar_h, 
                                         upscale_method=upscale_method, crop="disabled")[0]
        elif f_w > f_h and imgw > max_width:
            # resize
            tar_h = int(imgh / f_w)
            tar_w = max_width
            image = ImageScale().upscale(image, width=tar_w, height=tar_h, 
                                         upscale_method=upscale_method, crop="disabled")[0]
            
        return (image, )
            
            


class ConstrainImageMinSize:
    @classmethod
    def INPUT_TYPES(s):
        upscale_methods = ImageScale.upscale_methods
        return {
            "required": {
                "image": ("IMAGE",),
                "min_width": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "min_height": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "resize_methods": (upscale_methods, ),
            }
        }
        
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "aiyoh"
    
    
    def execute(self, image, min_width, min_height, resize_methods):
        d1, imgh, imgw, imgd = image.size()
        
        f_h = imgh / float(min_height)
        f_w = imgw / float(min_width)
        if f_h <= f_w and imgh < min_height:
            # resize
            tar_w = int(imgw / f_h)
            tar_h = min_height
            image = ImageScale().upscale(image, width=tar_w, height=tar_h, 
                                         upscale_method=resize_methods, crop="disabled")[0]
        elif f_w < f_h and imgw < min_width:
            # resize
            tar_h = int(imgh / f_w)
            tar_w = min_width
            image = ImageScale().upscale(image, width=tar_w, height=tar_h, 
                                         upscale_method=resize_methods, crop="disabled")[0]
            
        return (image, )
        



class ImagePadForOutpaintAdvance:

    @classmethod
    def INPUT_TYPES(s):
        padding_mode = ["empty", "nearest"]
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "padding_mode": (padding_mode, ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "aiyoh"

    def expand_image(self, image, left, top, right, bottom, feathering, padding_mode):
        d1, d2, d3, d4 = image.size()

        new_image = torch.zeros(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        )
        new_image[:, top:top + d2, left:left + d3, :] = image
        
        if padding_mode == "nearest":
            if left > 0 and right > 0:
                left_pixels = new_image[:, :, left:left+1, :]
                new_image[:, :, :left, :] = left_pixels.repeat(1, 1, left, 1)
                right_pixels = new_image[:, :, -right-1:-right, :]
                new_image[:, :, -right:, :] = right_pixels.repeat(1, 1, right, 1)

            # 填充顶部和底部
            if top > 0 and bottom > 0:
                top_pixels = new_image[:, top:top+1, :, :]
                new_image[:, :top, :, :] = top_pixels.repeat(1, top, 1, 1)
                bottom_pixels = new_image[:, -bottom-1:-bottom, :, :]
                new_image[:, -bottom:, :, :] = bottom_pixels.repeat(1, bottom, 1, 1)


        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask)
    

class ImageExpand:
    """
    
    """
    
    @classmethod
    def INPUT_TYPES(s):
        padding_mode = ["empty", "nearest"]
        upscale_methods = ImageScale.upscale_methods
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "target_height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "max_width": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "max_height": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "padding_mode": (padding_mode, ),
                "upscale_method": (upscale_methods,)
                
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "aiyoh"
    
    
    
    def expand_image(self, image, target_width, target_height, left, top, 
                     max_width=1024, max_height=1024, feathering=40, padding_mode="nearest", upscale_method="bilinear"):
        print(type(image))
        print(image.size())
        d1, imgh, imgw, imgd = image.size()
        
        tar_w = target_width
        tar_h = target_height
        _left = left
        _top = top
        
        
        f_h = target_height / float(max_height)
        f_w = target_width / float(max_width)
        if f_h >= f_w and target_height > max_height:
            # resize
            tar_w = int(target_width / f_h)
            tar_h = max_height
            _left = int(_left / f_h)
            _top = int(_top / f_h)
            image = ImageScale().upscale(image, width=int(imgw/f_h), height=int(imgh/f_h), 
                                         upscale_method=upscale_method, crop="disabled")[0]
        elif f_w > f_h and target_width > max_width:
            # resize
            tar_h = int(target_height * max_width / float(target_width))
            tar_w = max_width
            _left = int(_left / f_w)
            _top = int(_top / f_w)
            image = ImageScale().upscale(image, width=int(imgw/f_w), height=int(imgh/f_w), 
                                         upscale_method=upscale_method, crop="disabled")[0]
            
        d2, imgh2, imgw2, imgd2 = image.size()
        _right = max(0, tar_w - _left - imgw2)
        _bottom = max(0, tar_h - _top - imgh2)
        
        new_img, new_mask = ImagePadForOutpaintAdvance().expand_image(image, _left, _top, _right, _bottom, feathering,
                                                                      padding_mode)
        
        
        return (new_img, new_mask)
    




class ImageExpandBy:
    """
    
    """
    
    @classmethod
    def INPUT_TYPES(s):
        padding_mode = ["empty", "nearest"]
        upscale_methods = ImageScale.upscale_methods
        
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("FLOAT", {"default": 2, "min": 0.01, "max": 100, "step": 0.5}),
                "target_height": ("FLOAT", {"default": 2, "min": 0.01, "max": 100, "step": 0.5}),
                "left": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.1}),
                "top": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.1}),
                "max_width": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "max_height": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                
                "feathering": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.001}),
                "padding_mode": (padding_mode, ),
                "upscale_method": (upscale_methods,)
                
                
            }
        }

    RETURN_NAMES = ("IMAGE", "MASK", "width", "height", "left", "top", "right", "bottom")
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    FUNCTION = "expand_image"

    CATEGORY = "aiyoh"
    
    
    
    def expand_image(self, image, target_width, target_height, left, top, 
                     max_width=1024, max_height=1024, feathering=0.05, padding_mode="nearest", upscale_method="bilinear"):
        print(type(image))
        print(image.size())
        d1, imgh, imgw, imgd = image.size()
        
        tar_w = int(target_width * imgw)
        tar_h = int(target_height * imgh)
        _left = int(left * imgw)
        _top = int(top * imgh)
        
        f_h = tar_h / float(max_height)
        f_w = tar_w / float(max_width)
        if f_h >= f_w and tar_h > max_height:
            # resize
            tar_w = int(tar_w / f_h)
            tar_h = max_height
            _left = int(_left / f_h)
            _top = int(_top / f_h)
            image = ImageScale().upscale(image, width=int(imgw/f_h), height=int(imgh/f_h), 
                                         upscale_method=upscale_method, crop="disabled")[0]
        elif f_w > f_h and tar_w > max_width:
            # resize
            tar_h = int(tar_h / f_w)
            tar_w = max_width
            _left = int(_left / f_w)
            _top = int(_top / f_w)
            image = ImageScale().upscale(image, width=int(imgw/f_w), height=int(imgh/f_w), 
                                         upscale_method=upscale_method, crop="disabled")[0]
            
        d2, imgh2, imgw2, imgd2 = image.size()
        _right = max(0, tar_w - _left - imgw2)
        _bottom = max(0, tar_h - _top - imgh2)
        _feathering = int(min(imgh2, imgw2) * feathering)
        
        AppLog.info(f"[ImageExpandBy] left:{_left}, top: {_top}, right:{_right}, bottom:{_bottom}, feathering:{_feathering}")
        new_img, new_mask = ImagePadForOutpaintAdvance().expand_image(image, _left, _top, _right, _bottom, _feathering, padding_mode)
        
        return (new_img, new_mask, imgw2, imgh2, _left, _top, _right, _bottom)
    
    
    
    
ay_color_mapping = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (160, 85, 15),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (102, 102, 102),
    "olive": (128, 128, 0),
    "lime": (0, 128, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 128),
    "aqua": (0, 255, 128),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),    
}

AY_COLORS = ["custom", "white", "black", "red", "green", "blue", "yellow",
          "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
          "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
          "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
          "violet", "coral", "indigo"]

    
class AY_Color:

    @classmethod
    def INPUT_TYPES(s):
                    
        return {"required": {
                    "color": (AY_COLORS,),
                },
                "optional": {
                    "color_hex": ("STRING", {"multiline": False, "default": "#000000"})                
                }
        }

    RETURN_TYPES = ("COLOR", )
    FUNCTION = "get_color"
    CATEGORY = "aiyoh"
    
    def get_color(self, color, color_hex='#000000'):
        
        res_color = ImageUtil.get_color_values(color, color_hex, ay_color_mapping)
        AppLog.info(f"[AY_Color] color is: {res_color}")

        return (res_color, )
    
class DrawTextOnImage:
    """
    
    """
    
    @classmethod
    def INPUT_TYPES(s):
        fonts = folder_paths.get_filename_list('aiyoh_font')
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 256, "min": 1, "max": 10240, "step": 8}),
                "height": ("INT", {"default": 256, "min": 1, "max": 10240, "step": 8}),
                "left": ("INT", {"default": 0, "min": 1, "max": 10240, "step": 8}),
                "top": ("INT", {"default": 0, "min": 1, "max": 10240, "step": 8}),
                
                "font": (fonts, ),
                "font_size": ("INT", {"default": 10 }), 
                "cols": ("INT", {"default": 10}),
                "text": ("STRING", {"multiline": True}),             
                "color": (
                    "COLOR",
                    {"default": "black"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_text"

    CATEGORY = "aiyoh"
    
    
    def draw_text(self, image, width, height, left, top, font, font_size, text, cols, color):
        font_path = folder_paths.get_full_path("aiyoh_font", font)
        AppLog.info(f"[DrawTextOnImage] width: {width}, height: {height}, left:{left}, top:{top}, font:{font_path}, font_size:{font_size}")
        AppLog.info(f"[DrawTextOnImage] text: {text}")
        
        center_x = int(left + width / 2)
        center_y = int(top + height / 2)
        ts_img = image.clone()
        pil_img = ImageUtil.tensor_to_image(ts_img[0])
        pil_img_2 = ImageUtil.draw_text_on_image(pil_img, text, font_path, font_size, color, (center_x, center_y), cols)
        new_image = ImageUtil.image_to_tensor(pil_img_2)
        ts_img[0] = new_image
        return (ts_img, )
    
    
    
    
NODE_CLASS_MAPPINGS = {
    "ConstrainImageMaxSize": ConstrainImageMaxSize,
    "ConstrainImageMinSize": ConstrainImageMinSize,
    "ImageExpand": ImageExpand,
    "ImageExpandBy": ImageExpandBy,
    "ImagePadForOutpaintAdvance": ImagePadForOutpaintAdvance,
    
    "DrawTextOnImage": DrawTextOnImage,
    "AY_Color": AY_Color,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConstrainImageMaxSize": "Constrain Image MaxSize",
    "ConstrainImageMinSize": "Constrain Image MinSize",
    "ImageExpand": "Image Expand", 
    "ImageExpandBy": "Image Expand By",
    "ImagePadForOutpaintAdvance": "ImagePadForOutpaint Advance",
    
    "DrawTextOnImage": "Draw Text On Image",
    "AY_Color": "AY Color"
} 