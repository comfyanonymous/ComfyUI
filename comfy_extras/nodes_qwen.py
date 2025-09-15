import node_helpers
import comfy.utils
import math


class TextEncodeQwenImageEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {"vae": ("VAE", ),
                         "image": ("IMAGE", ),}}

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "LATENT")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image=None):
        ref_latent = None
        output_image = None
        
        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            # 修改缩放规则：乘数为8，向下取整
            width = math.floor(samples.shape[3] * scale_by / 8) * 8
            height = math.floor(samples.shape[2] * scale_by / 8) * 8

            # 根据缩放比例选择算法：缩小用area，放大用lanczos
            original_width = samples.shape[3]
            original_height = samples.shape[2]
            
            if width < original_width or height < original_height:
                # 缩小图像，使用area算法保持细节
                upscale_method = "area"
            else:
                # 放大图像，使用lanczos算法获得更好质量
                upscale_method = "lanczos"
            
            s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            output_image = image[:, :, :, :3]
            
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        
        # 将ref_latent包装成ComfyUI LATENT类型要求的格式
        latent_output = {"samples": ref_latent} if ref_latent is not None else None
        
        return (conditioning, output_image, latent_output)


NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEdit": TextEncodeQwenImageEdit,
}
