

from nodes import PreviewImage, LoadImage
from comfy.model_management import InterruptProcessingException
import os
import torch

from .image_filter_messaging import send_and_wait, Response, TimeoutResponse

HIDDEN = {
            "prompt": "PROMPT", 
            "extra_pnginfo": "EXTRA_PNGINFO", 
            "uid":"UNIQUE_ID",
            "node_identifier": "NID",
        }

class ImageFilter(PreviewImage):
    RETURN_TYPES = ("IMAGE","LATENT","MASK","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("images","latents","masks","extra1","extra2","extra3","indexes")
    FUNCTION = "func"
    CATEGORY = "image_filter"
    OUTPUT_NODE = False
    DESCRIPTION = "Allows you to preview images and choose which, if any to proceed with"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "images" : ("IMAGE", ), 
                "timeout": ("INT", {"default": 600, "min":1, "max":9999999, "tooltip": "Timeout in seconds."}),
                "ontimeout": (["send none", "send all", "send first", "send last"], {}),
            },
            "optional": {
                "latents" : ("LATENT", {"tooltip": "Optional - if provided, will be output"}),
                "masks" : ("MASK", {"tooltip": "Optional - if provided, will be output"}),
                "tip" : ("STRING", {"default":"", "tooltip": "Optional - if provided, will be displayed in popup window"}),
                "extra1" : ("STRING", {"default":""}),
                "extra2" : ("STRING", {"default":""}),
                "extra3" : ("STRING", {"default":""}),
                "pick_list_start" : ("INT", {"default":0, "tooltip":"The number used in pick_list for the first image"}),
                "pick_list" : ("STRING", {"default":"", "tooltip":"If a comma separated list of integers is provided, the images with these indices will be selected automatically."}),
                "video_frames" : ("INT", {"default":1, "min":1, "tooltip": "treat each block of n images as a video"}),
            },
            "hidden": HIDDEN,
        }
    
    @classmethod
    def IS_CHANGED(cls, pick_list, **kwargs):
        return pick_list or float("NaN")
    
    def func(self, images, timeout, ontimeout, uid, node_identifier, tip="", extra1="", extra2="", extra3="", latents=None, masks=None, pick_list_start:int=0, pick_list:str="", video_frames:int=1, **kwargs):
        e1, e2, e3 = extra1, extra2, extra3
        B = images.shape[0]

        if video_frames>B: video_frames=1
            

        try:    images_to_return = [ int(x.strip())%B for x in pick_list.split(',') ] if pick_list else []
        except Exception as e: 
            print(f"{e} parsing pick_list - will manually select")
            images_to_return = []

        if len(images_to_return) == 0:
            all_the_same = ( B and all( (images[i]==images[0]).all() for i in range(1,B) )) 
            urls:list[str] = self.save_images(images=images, **kwargs)['ui']['images']
            payload = {"uid": uid, "urls":urls, "allsame":all_the_same, "extras":[extra1, extra2, extra3], "tip":tip, "video_frames":video_frames}

            response:Response = send_and_wait(payload, timeout, uid, node_identifier)

            if isinstance(response, TimeoutResponse):
                if ontimeout=='send none':  images_to_return = []
                if ontimeout=='send all':   images_to_return = [*range(len(images)//video_frames)]
                if ontimeout=='send first': images_to_return = [0,]
                if ontimeout=='send last':  images_to_return = [(len(images)//video_frames)-1,]
            else:
                e1, e2, e3 = response.get_extras([extra1, extra2, extra3])
                images_to_return = response.selection

        if images_to_return is None or len(images_to_return) == 0: raise InterruptProcessingException()

        if video_frames>1:
            images_to_return = [ key*video_frames + frm  for key in images_to_return for frm in range(video_frames)   ]

        images = torch.stack(list(images[int(i)] for i in images_to_return))
        latents = {"samples": torch.stack(list(latents['samples'][int(i)] for i in images_to_return))} if latents is not None else None
        masks = torch.stack(list(masks[int(i)] for i in images_to_return)) if masks is not None else None

        try: int(pick_list_start)
        except: pick_list_start = '0'
                
        return (images, latents, masks, e1, e2, e3, ",".join(str(int(x)+int(pick_list_start)) for x in images_to_return))
    
class TextImageFilterWithExtras(PreviewImage):
    RETURN_TYPES = ("IMAGE","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("image","text","extra1","extra2","extra3")
    FUNCTION = "func"
    CATEGORY = "image_filter"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image" : ("IMAGE", ), 
                "text" : ("STRING", {"default":""}),
                "timeout": ("INT", {"default": 600, "min":1, "max":9999999, "tooltip": "Timeout in seconds."}),
            },
            "optional": {
                "mask" : ("MASK", {"tooltip": "Optional - if provided, will be overlaid on image"}),
                "tip" : ("STRING", {"default":"", "tooltip": "Optional - if provided, will be displayed in popup window"}),
                "extra1" : ("STRING", {"default":""}),
                "extra2" : ("STRING", {"default":""}),
                "extra3" : ("STRING", {"default":""}),
                "textareaheight" : ("INT", {"default": 150, "min": 50, "max": 500, "tooltip": "Height of text area in pixels"}),
            },
            "hidden": HIDDEN,
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    def func(self, image, text, timeout, uid, node_identifier, extra1="", extra2="", extra3="", mask=None, tip="", textareaheight=None, **kwargs):
        urls:list[str] = self.save_images(images=image, **kwargs)['ui']['images']
        payload = {"uid": uid, "urls":urls, "text":text, "extras":[extra1, extra2, extra3], "tip":tip}
        if textareaheight is not None: payload['textareaheight'] = textareaheight
        if mask is not None: payload['mask_urls'] = self.save_images(images=mask_to_image(mask), **kwargs)['ui']['images']

        response = send_and_wait(payload, timeout, uid, node_identifier)
        if isinstance(response, TimeoutResponse):
            return (image, text, extra1, extra2, extra3)

        return (image, response.text, *response.get_extras([extra1, extra2, extra3])) 

def mask_to_image(mask:torch.Tensor):
    return torch.stack([mask, mask, mask, 1.0-mask], -1)
    
class MaskImageFilter(PreviewImage, LoadImage):
    RETURN_TYPES = ("IMAGE","MASK","STRING","STRING","STRING")
    RETURN_NAMES = ("image","mask","extra1","extra2","extra3")
    FUNCTION = "func"
    CATEGORY = "image_filter"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image" : ("IMAGE", ), 
                "timeout": ("INT", {"default": 600, "min":1, "max":9999999, "tooltip": "Timeout in seconds."}),
                "if_no_mask": (["cancel", "send blank"], {}),
            },
            "optional": {
                "mask" : ("MASK", {"tooltip":"optional initial mask"}),
                "tip" : ("STRING", {"default":"", "tooltip": "Optional - if provided, will be displayed in popup window"}),
                "extra1" : ("STRING", {"default":""}),
                "extra2" : ("STRING", {"default":""}),
                "extra3" : ("STRING", {"default":""}),
            },
            "hidden": HIDDEN,
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs): return True
    
    def func(self, image, timeout, uid, if_no_mask, node_identifier, mask=None, extra1="", extra2="", extra3="", tip="", **kwargs):
        if mask is not None and mask.shape[:3] == image.shape[:3] and not torch.all(mask==0):
            saveable = torch.cat((image, mask.unsqueeze(-1)), dim=-1)
        else:
            saveable = image

        urls:list[str] = self.save_images(images=saveable, **kwargs)['ui']['images']
        payload = {"uid": uid, "urls":urls, "maskedit":True, "extras":[extra1, extra2, extra3], "tip":tip}
        response = send_and_wait(payload, timeout, uid, node_identifier)
        
        if (response.masked_image):
            try:
                return ( *(self.load_image(os.path.join('clipspace', response.masked_image)+" [input]")), *response.get_extras([extra1, extra2, extra3]) ) 
            except FileNotFoundError:
                pass

        if if_no_mask == 'cancel': 
            raise InterruptProcessingException()
        return ( *(self.load_image(urls[0]['filename']+" [temp]")), *response.get_extras([extra1, extra2, extra3]) ) 
