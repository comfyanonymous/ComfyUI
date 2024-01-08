

import torch

from nodes import MAX_RESOLUTION
from nodes import ImageScale
from framework.app_log import AppLog



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
            left_pixels = new_image[:, :, left:left+1, :]
            new_image[:, :, :left, :] = left_pixels.repeat(1, 1, left, 1)
            right_pixels = new_image[:, :, -right-1:-right, :]
            new_image[:, :, -right:, :] = right_pixels.repeat(1, 1, right, 1)

            # 填充顶部和底部
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
                
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "aiyoh"
    
    
    
    def expand_image(self, image, target_width, target_height, left, top, 
                     max_width=1024, max_height=1024, feathering=40, padding_mode="nearest"):
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
                                         upscale_method="nearest-exact", crop="disabled")[0]
        elif f_w > f_h and target_width > max_width:
            # resize
            tar_h = int(target_height * max_width / float(target_width))
            tar_w = max_width
            _left = int(_left / f_w)
            _top = int(_top / f_w)
            image = ImageScale().upscale(image, width=int(imgw/f_w), height=int(imgh/f_w), 
                                         upscale_method="nearest-exact", crop="disabled")[0]
            
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
                
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "aiyoh"
    
    
    
    def expand_image(self, image, target_width, target_height, left, top, 
                     max_width=1024, max_height=1024, feathering=0.05, padding_mode="nearest"):
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
                                         upscale_method="nearest-exact", crop="disabled")[0]
        elif f_w > f_h and tar_w > max_width:
            # resize
            tar_h = int(tar_h / f_w)
            tar_w = max_width
            _left = int(_left / f_w)
            _top = int(_top / f_w)
            image = ImageScale().upscale(image, width=int(imgw/f_w), height=int(imgh/f_w), 
                                         upscale_method="nearest-exact", crop="disabled")[0]
            
        d2, imgh2, imgw2, imgd2 = image.size()
        _right = max(0, tar_w - _left - imgw2)
        _bottom = max(0, tar_h - _top - imgh2)
        _feathering = int(min(imgh2, imgw2) * feathering)
        
        AppLog.info(f"[ImageExpandBy] left:{_left}, top: {_top}, right:{_right}, bottom:{_bottom}, feathering:{_feathering}")
        new_img, new_mask = ImagePadForOutpaintAdvance().expand_image(image, _left, _top, _right, _bottom, _feathering, padding_mode)
        
        
        return (new_img, new_mask)
    
    
    
    
NODE_CLASS_MAPPINGS = {
    "ConstrainImageMaxSize": ConstrainImageMaxSize,
    "ConstrainImageMinSize": ConstrainImageMinSize,
    "ImageExpand": ImageExpand,
    "ImageExpandBy": ImageExpandBy,
    "ImagePadForOutpaintAdvance": ImagePadForOutpaintAdvance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConstrainImageMaxSize": "Constrain Image MaxSize",
    "ConstrainImageMinSize": "Constrain Image MinSize",
    "ImageExpand": "Image Expand", 
    "ImageExpandBy": "Image Expand By",
    "ImagePadForOutpaintAdvance": "ImagePadForOutpaint Advance"
} 