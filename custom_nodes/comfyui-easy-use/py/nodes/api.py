import comfy.utils
from ..libs.api.fluxai import fluxaiAPI
from ..libs.api.bizyair import bizyairAPI, encode_data
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

class joyCaption2API:
    API_URL = f"/supernode/joycaption2"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "do_sample": ([True, False],),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "number",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 256,
                        "min": 16,
                        "max": 512,
                        "step": 16,
                        "display": "number",
                    },
                ),
                "caption_type": (
                    [
                        "Descriptive",
                        "Descriptive (Informal)",
                        "Training Prompt",
                        "MidJourney",
                        "Booru tag list",
                        "Booru-like tag list",
                        "Art Critic",
                        "Product Listing",
                        "Social Media Post",
                    ],
                ),
                "caption_length": (
                    ["any", "very short", "short", "medium-length", "long", "very long"]
                    + [str(i) for i in range(20, 261, 10)],
                ),
                "extra_options": (
                    "STRING",
                    {
                        "placeholder": "Extra options(e.g):\nIf there is a person/character in the image you must refer to them as {name}.",
                        "tooltip": "Extra options for the model",
                        "multiline": True,
                    },
                ),
                "name_input": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Name input is only used if an Extra Option is selected that requires it.",
                    },
                ),
                "custom_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
            "optional":{
                "apikey_override": ("STRING", {"default": "", "forceInput": True, "tooltip":"Override the API key in the local config"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)

    FUNCTION = "joycaption"
    OUTPUT_NODE = False

    CATEGORY = "EasyUse/API"

    def joycaption(
            self,
            image,
            do_sample,
            temperature,
            max_tokens,
            caption_type,
            caption_length,
            extra_options,
            name_input,
            custom_prompt,
            apikey_override=None
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        SIZE_LIMIT = 1536
        _, w, h, c = image.shape
        if w > SIZE_LIMIT or h > SIZE_LIMIT:
            node_class = ALL_NODE_CLASS_MAPPINGS['easy imageScaleDownToSize']
            image, = node_class().image_scale_down_to_size(image, SIZE_LIMIT, True)

        payload = {
            "image": None,
            "do_sample": do_sample == True,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "caption_type": caption_type,
            "caption_length": caption_length,
            "extra_options": [extra_options],
            "name_input": name_input,
            "custom_prompt": custom_prompt,
        }

        pbar.update_absolute(30)
        caption = bizyairAPI.joyCaption(payload, image, apikey_override, API_URL=self.API_URL)

        pbar.update_absolute(100)
        return (caption,)

class joyCaption3API(joyCaption2API):
    API_URL = f"/supernode/joycaption3"

NODE_CLASS_MAPPINGS = {
    "easy joyCaption2API": joyCaption2API,
    "easy joyCaption3API": joyCaption3API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy joyCaption2API": "JoyCaption2 (BizyAIR)",
    "easy joyCaption3API": "JoyCaption3 (BizyAIR)",
}