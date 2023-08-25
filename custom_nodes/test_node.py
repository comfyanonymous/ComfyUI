from transformers.models import clip


class TestNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "image": ("IMAGE",),
                "int_field": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "dong!"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "test"

    CATEGORY = "inflamously"

    def test(self, clip, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = 0.5 - image
        tokens = clip.tokenize("test")
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

NODE_CLASS_MAPPINGS = {
    "TestNode": TestNode,
    "TestNode2": TestNode,
}