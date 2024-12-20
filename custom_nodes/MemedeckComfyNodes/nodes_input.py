from comfy_extras.nodes_custom_sampler import Noise_RandomNoise


class MD_VideoInputs:
    """One node to load all input parameters for video generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": (
                    "STRING",
                ),
                "length": ("INT", {
                    "default": 121,
                    "description": "The length of the video."
                }),
                "steps": ("INT", {
                    "default": 25,
                    "description": "Number of steps to generate the video."
                }),
                "width": ("INT", {
                    "default": 768,
                    "description": "The width of the video."
                }),
                "height": ("INT", {
                    "default": 768,
                    "description": "The height of the video."
                }),
                "crf": ("INT", {
                    "default": 28,
                    "min": 0,
                    "max": 51,
                    "step": 1
                }),
                "terminal": ("FLOAT", {
                    "default": 0.1,
                    "step": 0.01,
                    "description": "The terminal values of the sigmas after stretching."
                }),
            },
            "optional": {
                "seed": (
                    "INT",
                ),
                "user_prompt": (
                    "STRING",
                ),
                "pre_prompt": (
                    "STRING",
                ),
                "post_prompt": (
                    "STRING",
                ),
                "negative_prompt": (
                    "STRING",
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "INT", "FLOAT", "STRING", "STRING", "STRING", "STRING", "NOISE",)
    RETURN_NAMES = ("image_url", "length", "steps", "width", "height", "crf", "terminal", "user_prompt", "pre_prompt", "post_prompt", "negative_prompt", "seed")
    FUNCTION = "load_inputs"
    CATEGORY = "MemeDeck"

    def load_inputs(self, image_url, length=121, steps=25, width=768, height=768, crf=28, terminal=0.1, user_prompt="", pre_prompt="", post_prompt="", negative_prompt="", seed=None):
      return (image_url, length, steps, width, height, crf, terminal, user_prompt, pre_prompt, post_prompt, negative_prompt, Noise_RandomNoise(seed))