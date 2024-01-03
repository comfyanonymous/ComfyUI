

from nodes import MAX_RESOLUTION
from nodes import ImageScale, ImagePadForOutpaint


class ImageExpand:
    """
    
    """
    
    @classmethod
    def INPUT_TYPES(s):
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"
    
    
    
    def expand_image(self, image, target_width, target_height, left, top, 
                     max_width=1024, max_height=1024, feathering=40):
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
        
        new_img, new_mask = ImagePadForOutpaint().expand_image(image, _left, _top, _right, _bottom, feathering)
        
        
        return (new_img, new_mask)
    




class ImageExpandBy:
    """
    
    """
    
    @classmethod
    def INPUT_TYPES(s):
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"
    
    
    
    def expand_image(self, image, target_width, target_height, left, top, 
                     max_width=1024, max_height=1024, feathering=0.05):
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
        
        _feathering = min(imgh2, imgw2) * feathering
        new_img, new_mask = ImagePadForOutpaint().expand_image(image, _left, _top, _right, _bottom, _feathering)
        
        
        return (new_img, new_mask)
    
    
    
    
NODE_CLASS_MAPPINGS = {
    
    "ImageExpand": ImageExpand,
    "ImageExpandBy": ImageExpandBy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageExpand": "Image Expand", 
    "ImageExpandBy": "Image Expand By"
} 