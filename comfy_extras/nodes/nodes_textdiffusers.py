"""
Adapted from https://github.com/microsoft/unilm/blob/master/textdiffuser-2/inference_textdiffuser2_t2i_full.py#L334

The MIT License (MIT)

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import string
from typing import Optional

from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult
from comfy.sd import CLIP
from comfy.sd1_clip import SDTokenizer


class TextDiffuserTokens(CustomNode):
    ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(alphabet) = 95
    TOKENS = []

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "clip": ("CLIP",)
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "execute"

    def execute(self, clip: CLIP):
        clip = clip.clone()
        if len(TextDiffuserTokens.TOKENS) == 0:
            for i in range(520):
                TextDiffuserTokens.TOKENS.append(f'l{i}</w>')
                TextDiffuserTokens.TOKENS.append(f't{i}</w>')
                TextDiffuserTokens.TOKENS.append(f'r{i}</w>')
                TextDiffuserTokens.TOKENS.append(f'b{i}</w>')
            for c in TextDiffuserTokens.ALPHABET:
                TextDiffuserTokens.TOKENS.append(f'[{c}]</w>')
        tokenizer: SDTokenizer = clip.tokenizer.sd_tokenizer
        existing_vocab = frozenset(tokenizer.tokenizer.get_vocab().keys())
        tokens = [t for t in TextDiffuserTokens.TOKENS if t not in existing_vocab]
        if len(tokens) != 0:
            tokenizer.add_tokens(tokens)

        # todo: assert that the clip's vocab size is what we expect
        return clip,


class TextDiffuserPrepare(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True})
            }
        }

    FUNCTION = "execute"
    RETURN_TYPES = "STRING",
    RETURN_NAMES = "INSTRUCT STRING",

    def execute(self, prompt: str, text: Optional[str] = None, *args, **kwargs) -> ValidatedNodeResult:
        keywords = text.split("\n")
        if len(keywords) > 0:
            # text diffusers does indeed format keywords as
            # ['some', 'word']
            message = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. In addition, we also provide all keywords at random order for reference. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {prompt}. Keywords: {str(keywords)}'
        else:
            message = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. All keywords are included in the caption. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {prompt}'

        return message,


class TextDiffuserDecodeLayout(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "layout_model": ("MODEL", {}),
                "clip": ("CLIP", {}),
                "prompt": ("STRING", {}),
                "instruct_response": ("STRING", {})
            }
        }

    FUNCTION = "execute"
    RETURN_TYPES = "STRING",
    RETURN_NAMES = "CLIP STRING",

    def execute(self, layout_model: TransformersManagedModel, clip: CLIP, prompt: str, instruct_response: str, *args, **kwargs) -> ValidatedNodeResult:
        current_ocr = instruct_response.split('\n')
        words = [clip.tokenizer.sd_tokenizer.tokenizer.eos_token, clip.tokenizer.sd_tokenizer.tokenizer.bos_token]
        for ocr in current_ocr:
            ocr = ocr.strip()

            # .com ??
            if len(ocr) == 0 or '###' in ocr or '.com' in ocr:
                continue

            items = ocr.split()
            pred = ' '.join(items[:-1])
            box = items[-1]

            l, t, r, b = map(int, box.split(','))
            words.extend([f'l{l}', f't{t}', f'r{r}', f'b{b}'])

            char_list = [f'[{i}]' for i in pred]
            words.extend(char_list)
            words.append(clip.tokenizer.sd_tokenizer.tokenizer.eos_token)
        return prompt + ' ' + ' '.join(words),


NODE_CLASS_MAPPINGS = {}
for cls in (
        TextDiffuserDecodeLayout,
        TextDiffuserPrepare,
        TextDiffuserTokens,
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
