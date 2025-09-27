import torch
import comfy.model_management

class EmptyLatentHunyuanFoley:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "length": ("INT", {"default": 12, "min": 1, "max": 15, "tooltip": "The length of the audio. The same length as the video."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent audios in the batch."}),
            },
            "optional": {"video": ("VIDEO")}
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/audio"

    def generate(self, length, batch_size, video = None):
        if video is not None:
            _, length = video.get_duration(return_frames = True)
            length /= 25
        shape = (batch_size, 128, int(50 * length))
        latent = torch.randn(shape, device=comfy.model_management.intermediate_device())
        return ({"samples": latent, "type": "hunyuan_foley"}, )

class HunyuanFoleyConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"video_encoding_siglip": ("CONDITIONING",),
                             "video_encoding_synchformer": ("CONDITIONING",),
                             "text_encoding": ("CONDITIONING",)
                },
            }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")

    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, video_encoding_1, video_encoding_2, text_encoding):
        embeds = torch.cat([video_encoding_1, video_encoding_2, text_encoding], dim = 0)
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return (positive, negative)

NODE_CLASS_MAPPINGS = {
    "HunyuanFoleyConditioning": HunyuanFoleyConditioning,
    "EmptyLatentHunyuanFoley": EmptyLatentHunyuanFoley,
}
