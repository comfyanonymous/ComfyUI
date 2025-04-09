from inspect import cleandoc
class IdeogramTextToImage:
    """
    Generates images synchronously based on a given prompt and optional parameters.

    Images links are available for a limited period of time; if you would like to keep the image, you must download it.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                    "default": "", "tooltip": "Prompt for the image generation"}),
                "model": (["V_2", "V_2_TURBO", "V_1", "V_1_TURBO"], {"default": "V_2"}),
            },
            "optional": {
                "aspect_ratio": (["ASPECT_1_1", "ASPECT_4_3", "ASPECT_3_4", "ASPECT_16_9", "ASPECT_9_16", 
                                "ASPECT_2_1", "ASPECT_1_2", "ASPECT_3_2", "ASPECT_2_3", "ASPECT_4_5", "ASPECT_5_4"], {
                    "default": "ASPECT_1_1",
                    "tooltip": "The aspect ratio for image generation. Cannot be used with resolution"
                }),
                "resolution": (["1024x1024", "1024x1792", "1792x1024"], {
                    "default": "1024x1024",
                    "tooltip": "The resolution for image generation (V2 only). Cannot be used with aspect_ratio"
                }),
                "magic_prompt_option": (["AUTO", "ON", "OFF"], {
                    "default": "AUTO",
                    "tooltip": "Determine if MagicPrompt should be used in generation"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
                "style_type": (["NONE", "ANIME", "CINEMATIC", "CREATIVE", "DIGITAL_ART", "PHOTOGRAPHIC"], {
                    "default": "NONE",
                    "tooltip": "Style type for generation (V2+ only)"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of what to exclude from the image (V1/V2 only)"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number"
                }),
                "color_palette": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Color palette preset name or hex colors with weights (V2/V2_TURBO only)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "api_call"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"

    def api_call(self, prompt, model, aspect_ratio=None, resolution=None, 
                 magic_prompt_option="AUTO", seed=0, style_type="NONE", 
                 negative_prompt="", num_images=1, color_palette=""):
        import requests
        import torch
        from PIL import Image
        import io
        import numpy as np
        import time

        # Build payload with all available parameters
        payload = {
            "image_request": {
                "prompt": prompt,
                "model": model,
                "num_images": num_images,
                "seed": seed,
            }
        }

        # Make API request
        headers = {
            "Authorization": "Bearer TBD", # TODO(robin): add authorization key
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "http://localhost:8080/proxy/ideogram/generate",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")

        # Parse response
        response_data = response.json()
        
        # Get the image URL from the response
        image_url = response_data["data"][0]["url"]
        
        # Time the image download
        download_start = time.time()
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            raise Exception("Failed to download the image")
        download_time = (time.time() - download_start) * 1000  # Convert to milliseconds
        print(f"Image download time: {download_time:.2f}ms")

        # Time the conversion process
        conversion_start = time.time()
        img = Image.open(io.BytesIO(img_response.content))
        img = img.convert("RGB")  # Ensure RGB format
        
        # Convert to numpy array, normalize to float32 between 0 and 1
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to torch tensor and add batch dimension
        img_tensor = torch.from_numpy(img_array)[None,]
        conversion_time = (time.time() - conversion_start) * 1000  # Convert to milliseconds
        print(f"Image conversion time: {conversion_time:.2f}ms")

        return (img_tensor,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


class RunwayVideoNode:
    """
    Generates videos synchronously based on a given image, prompt, and optional parameters using Runway's API.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_image": ("IMAGE",),  # Will need to handle image URL conversion
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt to guide the video generation"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4294967295,
                    "step": 1,
                    "display": "number"
                }),
                "model": (["gen3a_turbo"], {
                    "default": "gen3a_turbo",
                    "tooltip": "Model to use for video generation"
                }),
                "duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Duration of the generated video in seconds"
                }),
                "ratio": (["1280:768", "768:1280"], {
                    "default": "1280:768",
                    "tooltip": "Aspect ratio of the output video"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to include watermark in the output"
                }),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Generates videos from images using Runway's API"
    FUNCTION = "generate_video"
    CATEGORY = "video"

    def generate_video(self, prompt_image, prompt_text, seed=0, model="gen3a_turbo", 
                      duration=5.0, ratio="1280:768", watermark=False):
        import requests
        import torch
        import time
        import os

        # Hardcoded API key (temporary solution)
        api_key = "key_e861661aa0b307e07e8cc269c1f42cf56fcce876ed6511a507e185ee51f695291da21f4777be1326b4467c34be5a6498b72dc27c9780e483250c692aa410d4c6"  # Replace with actual API key

        # Convert torch tensor image to URL (you'll need to implement this part)
        # This is a placeholder - you'll need to either save the image temporarily
        # or upload it to a service that can host it
        image_url = "http://example.com"  # Placeholder

        # Build payload
        payload = {
            "promptImage": image_url,
            "promptText": prompt_text,
            "seed": seed,
            "model": model,
            "watermark": watermark,
            "duration": duration,
            "ratio": ratio
        }

        # Make API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": "2024-11-06"
        }

        # Time the API request
        api_start = time.time()
        response = requests.post(
            "https://api.dev.runwayml.com/v1/image_to_video",
            headers=headers,
            json=payload
        )
        api_time = (time.time() - api_start) * 1000  # Convert to milliseconds
        print(f"API request time: {api_time:.2f}ms")

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")

        # Parse response
        response_data = response.json()
        
        # Note: You'll need to implement the actual video handling here
        # This is a placeholder return
        return (None,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "IdeogramTextToImage": IdeogramTextToImage,
    "RunwayVideoNode": RunwayVideoNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTextToImage": "Ideogram Text to Image",
    "RunwayVideoNode": "Runway Video Generator"
}
