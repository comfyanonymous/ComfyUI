import os
import folder_paths
from pathlib import Path

BASE_RESOLUTIONS = [
    ("width", "height"),
    (512, 512),
    (512, 768),
    (576, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (816, 1920),
    (832, 1152),
    (832, 1216),
    (896, 1152),
    (896, 1088),
    (1024, 1024),
    (1024, 576),
    (1024, 768),
    (1080, 1920),
    (1440, 2560),
    (1088, 896),
    (1216, 832),
    (1152, 832),
    (1152, 896),
    (1280, 768),
    (1344, 768),
    (1536, 640),
    (1536, 768),
    (1920, 816),
    (1920, 1080),
    (2560, 1440),
]
MAX_SEED_NUM = 1125899906842624


RESOURCES_DIR = os.path.join(Path(__file__).parent.parent, "resources")

# inpaint
INPAINT_DIR = os.path.join(folder_paths.models_dir, "inpaint")
FOOOCUS_STYLES_DIR = os.path.join(Path(__file__).parent.parent, "styles")
FOOOCUS_STYLES_SAMPLES = 'https://raw.githubusercontent.com/lllyasviel/Fooocus/main/sdxl_styles/samples/'
FOOOCUS_INPAINT_HEAD = {
    "fooocus_inpaint_head": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"
    }
}
FOOOCUS_INPAINT_PATCH = {
    "inpaint_v26 (1.32GB)": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"
    },
    "inpaint_v25 (2.58GB)": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch"
    },
    "inpaint (1.32GB)": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch"
    },
}
BRUSHNET_MODELS = {
    "random_mask": {
        "sd1": {
            "model_url": "https://huggingface.co/Kijai/BrushNet-fp16/resolve/main/brushnet_random_mask_fp16.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/yolain/brushnet/resolve/main/brushnet_random_mask_sdxl.safetensors"
        }
    },
    "segmentation_mask": {
        "sd1": {
            "model_url": "https://huggingface.co/Kijai/BrushNet-fp16/resolve/main/brushnet_segmentation_mask_fp16.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/yolain/brushnet/resolve/main/brushnet_segmentation_mask_sdxl.safetensors"
        }
    }
}
POWERPAINT_MODELS = {
    "base_fp16": {
        "model_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.fp16.safetensors"
    },
    "v2.1": {
        "model_url": "https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/resolve/main/PowerPaint_Brushnet/diffusion_pytorch_model.safetensors",
        "clip_url": "https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/resolve/main/PowerPaint_Brushnet/pytorch_model.bin",
    }
}

# layerDiffuse
LAYER_DIFFUSION_DIR = os.path.join(folder_paths.models_dir, "layer_model")
LAYER_DIFFUSION_VAE = {
    "encode": {
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors"
        }
    },
    "decode": {
        "sd1": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_vae_transparent_decoder.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors"
        }
    }
}
LAYER_DIFFUSION = {
    "Attention Injection": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_transparent_attn.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors"
        },
    },
    "Conv Injection": {
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors"
        },
        "sd1": {
            "model_url": None
        }
    },
    "Everything": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_joint.safetensors"
        },
        "sdxl": {
            "model_url": None
        }
    },
    "Foreground": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_fg2bg.safetensors"
        },
        "sdxl": {
           "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors"
        }
    },
    "Foreground to Background": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_fg2bg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fgble2bg.safetensors"
        }
    },
    "Background": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_bg2fg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors"
        }
    },
    "Background to Foreground": {
        "sd1": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_bg2fg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bgble2fg.safetensors"
        }
    },
}

# IC Light
IC_LIGHT_MODELS = {
    "Foreground": {
        "sd1": {
            "model_url": "https://huggingface.co/huchenlei/IC-Light-ldm/resolve/main/iclight_sd15_fc_unet_ldm.safetensors"
        },
        "sdxl": {
            "model_url": None
        }
    },
    "Foreground&Background": {
        "sd1": {
            "model_url": "https://huggingface.co/huchenlei/IC-Light-ldm/resolve/main/iclight_sd15_fbc_unet_ldm.safetensors"
        },
        "sdxl": {
            "model_url": None
        }
    }
}


# REMBG
REMBG_DIR = os.path.join(folder_paths.models_dir, "rembg")
REMBG_MODELS = {
    "RMBG-1.4": {
        "model_url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth"
    },
    "RMBG-2.0": {
        "model_url": "briaai/RMBG-2.0"
    },
    "BEN2": {
        "model_url": "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.pth"
    }
}

#ipadapter
IPADAPTER_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
IPADAPTER_MODELS = {
    "LIGHT - SD1.5 only (low strength)": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin"
        },
        "sdxl": {
            "model_url": ""
        }
    },
    "STANDARD (medium strength)": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"
        }
    },
    "VIT-G (medium strength)": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"
        }
    },
    "PLUS (high strength)": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
        }
    },
    "PLUS (kolors genernal)": {
        "sd1": {
            "model_url": ""
        },
        "sdxl": {
            "model_url":"https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/ip_adapter_plus_general.bin"
        }
    },
    "REGULAR - FLUX and SD3.5 only (high strength)": {
       "flux": {
           "model_url": "https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/resolve/main/ip-adapter.bin",
           "model_file_name": "ip-adapter_flux_1_dev.bin",
       },
       "sd3": {
           "model_url": "https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter/resolve/main/ip-adapter.bin",
           "model_file_name": "ip-adapter_sd35.bin",
       },
    },
    "PLUS FACE (portraits)": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"
        }
    },
    "FULL FACE - SD1.5 only (portraits stronger)": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors"
        },
        "sdxl": {
            "model_url": ""
        }
    },
    "FACEID": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors"
        }
    },
    "FACEID PLUS - SD1.5 only": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15_lora.safetensors"
        },
        "sdxl": {
            "model_url": "",
            "lora_url": ""
        }
    },
    "FACEID PLUS V2": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
        }
    },
    "FACEID PLUS KOLORS":{
        "sd1":{

        },
        "sdxl":{
            "model_url":"https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus/resolve/main/ipa-faceid-plus.bin"
        }
    },
    "FACEID PORTRAIT (style transfer)": {
        "sd1": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin",
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin",
        }
    },
    "FACEID PORTRAIT UNNORM - SDXL only (strong)": {
        "sd1": {
            "model_url":""
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl_unnorm.bin",
        }
    },
    "COMPOSITION": {
        "sd1": {
            "model_url": "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors"
        }
    }
}
IPADAPTER_CLIPVISION_MODELS = {
    "clip-vit-large-patch14-336":{
        "model_url": "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/pytorch_model.bin"
    },
    "clip-vit-h-14-laion2B-s32B-b79K":{
        "model_url": "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_model.safetensors"
    },
    "sigclip_vision_patch14_384":{
        "model_url": "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors"
    }
}

# dynamiCrafter
DYNAMICRAFTER_DIR = os.path.join(folder_paths.models_dir, "dynamicrafter_models")
DYNAMICRAFTER_MODELS = {
    "dynamicrafter_unet_512 (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_512.safetensors",
        "vae_url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
        "clip_url": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/text_encoder/model.safetensors",
        "clip_vision_url": "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_model.safetensors",
    },
    "dynamicrafter_unet_512_interp (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_512_interp.safetensors"
    },
    "dynamicrafter_unet_1024 (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_1024.safetensors"
    },
    "dynamicrafter_unet_256 (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_256.safetensors"
    },
}

#humanParsing
HUMANPARSING_MODELS = {
    "parsing_lip": {
        "model_url": "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx",
    },
    "human-parts":{
        "model_url":"https://huggingface.co/Metal3d/deeplabv3p-resnet50-human/resolve/main/deeplabv3p-resnet50-human.onnx",
    },
    "segformer_b3_clothes":{
        "model_name": "sayeed99/segformer_b3_clothes",
    },
    "segformer_b3_fashion":{
        "model_name": "sayeed99/segformer-b3-fashion",
    },
    "face_parsing":{
        "model_name": "jonathandinu/face-parsing"
    }
}

#mediapipe
MEDIAPIPE_DIR = os.path.join(folder_paths.models_dir, "mediapipe")
MEDIAPIPE_MODELS = {
    "selfie_multiclass_256x256": {
        "model_url": "https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite"
    }
}


#prompt template
PROMPT_TEMPLATE = {
    "prefix": ["Detailed photo of", "Amateur photo of", "Flicker 2008 photo of", "Fantastic artwork of",
               "Vintage photograph of", "Unreal 5 render of", "Surrealist painting of",
               "Professional advertising design of"],
    "subject": ["a man", "a woman", "a young man", "a young woman", "a handsome man", "a beautiful woman", "a monster", "a toy", "a product", "a buddha", "a dog", "a cat"],
    "action": ["looking at viewer", "looking away", "looking up", "looking down", "looking back", "open mouth", "half-closed mouth", "closed mouth", "open eyes", "half-closed eyes", "closed eyes", "wink", "standing", "sitting", "lying", "walking", "running", "adjusting hair", "waving", "hand on hip", "crossed arms", "smile", "sad", "angry", "sleepy", "tired", "expressionless"],
    "clothes": ["underwear", "clothed", "casual", "dress", "swimsuit", "uniform", "bikini", "one-piece swimsuit", "shirt", "blouse", "sweater", "hoodie", "jeans", "pants", "shorts", "skirt", "vest", "coat", "trenchoat", "jacket", "short dress", "long dress", "off-shoulder", "backless", "hairbow", "hair ribbon", "hair tie", "hairband", "cap", "beanie",  "bucket hat", "sun hat", "straw hat", "rice hat", "witch hat", "crown", "chain necklace", "tooth necklace", "choker", "pendant", "bracelet", "watch", "ring", "earring", "anklet", "belt", "scarf", "gloves", "mittens", "socks", "stockings", "tights", "leggings", "boots", "sneakers", "heels", "sandals", "flip-flops", "slippers", "loafers", "mules", "oxfords", "brogues", "derbies", "monk shoes", "chelsea boots", "combat boots", "riding boots", "rain boots", "wedge heels", "platform heels", "stilettos", "block heels", "kitten heels", "moccasins", "espadrilles", "pumps", "flats", "ballet flats", "mary janes", "slingbacks", "peep-toe", "mule sandals", "gladiator sandals", "thong sandals", "slide sandals", "espadrille sandals", "wedge sandals", "platform sandals", "ankle boots", "knee-high boots", "over-the-knee boots", "thigh-high boots", "wellington boots", "chukka boots", "desert boots", "chelsea boots", "hiking boots", "work boots", "snow boots", "rain boots", "riding boots", "cowboy boots", "combat boots", "biker boots", "duck boots", "military boots", "western boots", "ankle strap heels", "block heels", "chunky heels", "cone heels", "kitten heels", "platform heels", "pumps", "slingback heels", "stiletto heels", "wedge heels", "mules", "slingbacks", "slides", "thong sandals", "gladiator sandals", "espadrilles", "wedge sandals", "platform sandals", "ankle boots", "knee-high boots", "over-the-knee boots", "thigh-high boots", "wellington boots", "chukka boots", "desert boots", "chelsea boots", "hiking boots", "work boots", "snow boots", "rain boots", "riding boots", "cowboy boots", "combat boots", "biker boots", "duck boots", "military boots", "western boots", "ankle strap heels", "block heels" ],
    "environment": ["sunshine from window", "neon night, city", "sunset over sea", "golden time", "sci-fi RGB glowing, cyberpunk", "natural lighting", "warm atmosphere, at home, bedroom", "magic lit", "evil, gothic, in a cave", "light and shadow", "shadow from window", "soft studio lighting", "home atmosphere, cozy bedroom illumination", "neon, Wong Kar-wai, warm", "moonlight through curtains", "stormy sky lighting", "underwater glow, deep sea", "foggy forest at dawn", "golden hour in a meadow", "rainbow reflections, neon", "cozy candlelight", "apocalyptic, smoky atmosphere", "red glow, emergency lights", "mystical glow, enchanted forest", "campfire light", "harsh, industrial lighting", "sunrise in the mountains", "evening glow in the desert", "moonlight in a dark alley", "golden glow at a fairground", "midnight in the forest", "purple and pink hues at twilight", "foggy morning, muted light", "candle-lit room, rustic vibe", "fluorescent office lighting", "lightning flash in storm", "night, cozy warm light from fireplace", "ethereal glow, magical forest", "dusky evening on a beach", "afternoon light filtering through trees", "blue neon light, urban street", "red and blue police lights in rain", "aurora borealis glow, arctic landscape", "sunrise through foggy mountains", "golden hour on a city skyline", "mysterious twilight, heavy mist", "early morning rays, forest clearing", "colorful lantern light at festival", "soft glow through stained glass", "harsh spotlight in dark room", "mellow evening glow on a lake", "crystal reflections in a cave", "vibrant autumn lighting in a forest", "gentle snowfall at dusk", "hazy light of a winter morning", "soft, diffused foggy glow", "underwater luminescence", "rain-soaked reflections in city lights", "golden sunlight streaming through trees", "fireflies lighting up a summer night", "glowing embers from a forge", "dim candlelight in a gothic castle", "midnight sky with bright starlight", "warm sunset in a rural village", "flickering light in a haunted house", "desert sunset with mirage-like glow", "golden beams piercing through storm clouds"],
    "background": ["cars and people", "a cozy bed and a lamp", "a forest clearing with mist", "a bustling marketplace", "a quiet beach at dusk", "an old, cobblestone street", "a futuristic cityscape", "a tranquil lake with mountains", "a mysterious cave entrance", "bookshelves and plants in the background", "an ancient temple in ruins", "tall skyscrapers and neon signs", "a starry sky over a desert", "a bustling café", "rolling hills and farmland", "a modern living room with a fireplace", "an abandoned warehouse", "a picturesque mountain range", "a starry night sky", "the interior of a futuristic spaceship", "the cluttered workshop of an inventor", "the glowing embers of a bonfire", "a misty lake surrounded by trees", "an ornate palace hall", "a busy street market", "a vast desert landscape", "a peaceful library corner", "bustling train station", "a mystical, enchanted forest", "an underwater reef with colorful fish", "a quiet rural village", "a sandy beach with palm trees", "a vibrant coral reef, teeming with life", "snow-capped mountains in distance", "a stormy ocean, waves crashing", "a rustic barn in open fields", "a futuristic lab with glowing screens", "a dark, abandoned castle", "the ruins of an ancient civilization", "a bustling urban street in rain", "an elegant grand ballroom", "a sprawling field of wildflowers", "a dense jungle with sunlight filtering through", "a dimly lit, vintage bar", "an ice cave with sparkling crystals", "a serene riverbank at sunset", "a narrow alley with graffiti walls", "a peaceful zen garden with koi pond", "a high-tech control room", "a quiet mountain village at dawn", "a lighthouse on a rocky coast", "a rainy street with flickering lights", "a frozen lake with ice formations", "an abandoned theme park", "a small fishing village on a pier", "rolling sand dunes in a desert", "a dense forest with towering redwoods", "a snowy cabin in the mountains", "a mystical cave with bioluminescent plants", "a castle courtyard under moonlight", "a bustling open-air night market", "an old train station with steam", "a tranquil waterfall surrounded by trees", "a vineyard in the countryside", "a quaint medieval village", "a bustling harbor with boats", "a high-tech futuristic mall", "a lush tropical rainforest"],
    "nsfw": ["nude", "breast", "small breast", "middle breast", "large breast", "nipples", "clothes lift", "pussy juice trail", "pussy juice puddle", "small testicles", "medium testicles", "large testicles", "disembodied penis", "cum on body", "cum inside", "cum outside", "fingering", "handjob", "fellatio", "licking penis", "paizuri",  "doggystyle", "cowgirl", "reversed cowgirl", "piledriver", "suspended congress", "full nelson",],
}

NEW_SCHEDULERS = ['align_your_steps', 'gits']
