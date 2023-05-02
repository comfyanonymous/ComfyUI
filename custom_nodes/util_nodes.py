import hashlib
import os

import numpy as np

import folder_paths


class TextLoader:
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        prompts_dir = folder_paths.prompt_directory
        return {"required":
                    {"prompt_file": (sorted(os.listdir(prompts_dir)),)},
                }

    CATEGORY = "utils"

    RETURN_TYPES = ("TEXT",)
    FUNCTION = "load_text"

    def load_text(self, prompt_file):
        text_file_path = os.path.join(folder_paths.prompt_directory, prompt_file)
        with open(text_file_path, 'r') as f:
            text = f.read()
        return (text,)

    @classmethod
    def IS_CHANGED(s, prompt_file):
        text_file_path = os.path.join(folder_paths.prompt_directory, prompt_file)
        m = hashlib.sha256()
        with open(text_file_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, prompt_file):
        text_file_path = os.path.join(folder_paths.prompt_directory, prompt_file)
        if not os.path.exists(text_file_path):
            return "Invalid text file: {}".format(text_file_path)

        return True

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

        output_text = []

        # attention[a][b][c][d]
        # a: number of steps/sigma in this diffusion process
        # b: number of SpatialTransformer or AttentionBlocks used in the middle blocks of the latent diffusion model
        # c: number of transformer layers in the SpatialTransformer or AttentionBlocks
        # d: attn1, attn2
        if attention is not None:
            output_text.append(f'attention has {len(attention)} steps\n')
            output_text[-1] += f'each step has {len(attention[0])} transformer blocks\n'
            output_text[-1] += f'each block has {len(attention[0][0])} transformer layers\n'
            output_text[-1] += f'each transformer layer has {len(attention[0][0][0])} attention tensors (attn1, attn2)\n'
            output_text[-1] += f'the shape of the attention tensors is {attention[0][0][0][0].shape}\n'
            output_text[-1] += f'the first value of the first attention tensor is {attention[0][0][0][0][:1]}\n'
            print(output_text[-1])

        if text is not None:
            output_text.append(text)
            print(text)

        if image is not None:
            _, height, width, _ = image.shape
            output_text.append(f"Image dimensions: {width}x{height}\n")
            print(output_text[-1])


        return {"ui": {"text": "\n".join(output_text)}}

NODE_CLASS_MAPPINGS = {
    "PrintNode": PrintNode,
    "TextLoader": TextLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrintNode": "Print",
    "TextLoader": "Text Loader",
}
