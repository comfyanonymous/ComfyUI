import hashlib
import numpy as np


class PrintNode:

    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "text": ("text",),
                "attention": ("ATTENTION",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
            }
        }
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True

    RETURN_TYPES = ()
    FUNCTION = "print_value"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def print_value(self, text=None, latent=None, attention=None, image=None):
        if latent is not None:
            latent_hash = hashlib.sha256(latent["samples"].cpu().numpy().tobytes()).hexdigest()
            print(f"Latent hash: {latent_hash}")
            print(np.array2string(latent["samples"].cpu().numpy(), separator=', '))

        # attention[a][b][c][d]
        # a: number of steps/sigma in this diffusion process
        # b: number of SpatialTransformer or AttentionBlocks used in the middle blocks of the latent diffusion model
        # c: number of transformer layers in the SpatialTransformer or AttentionBlocks
        # d: attn1, attn2
        if attention is not None:
            print(f'attention has {len(attention)} steps')
            print(f'each step has {len(attention[0])} transformer blocks')
            print(f'each block has {len(attention[0][0])} transformer layers')
            print(f'each transformer layer has {len(attention[0][0][0])} attention tensors (attn1, attn2)')
            print(f'the shape of the attention tensors is {attention[0][0][0][0].shape}')
            print(f'the first value of the first attention tensor is {attention[0][0][0][0][:1]}')


        if text is not None:
            print(text)

        if image is not None:
            _, height, width, _ = image.shape
            print(f"Image dimensions: {width}x{height}")

        return {"ui": {"": text}}

NODE_CLASS_MAPPINGS = {
    "PrintNode": PrintNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrintNode": "Print",
}
