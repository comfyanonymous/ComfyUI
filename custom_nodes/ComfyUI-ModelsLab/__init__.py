"""
ComfyUI-ModelsLab: ModelsLab API Integration for ComfyUI

This package provides custom nodes for ComfyUI to integrate with ModelsLab's
comprehensive AI generation APIs, including text-to-image, image-to-image,
video generation, and text-to-speech capabilities.

ModelsLab offers:
- High-quality image generation with Flux, SDXL, and community models
- Competitive pricing compared to premium API providers
- Multi-modal capabilities (text, image, video, audio)
- Uncensored models for creative freedom
- Professional-grade API with async processing

Get your API key at: https://modelslab.com
Documentation: https://docs.modelslab.com
"""

from .modules.text_to_image_node import (
    ModelsLabTextToImage,
    ModelsLabFlux,
    ModelsLabSDXL,
    ModelsLabCommunityModel,
)

# Import additional modules when implemented
try:
    from .modules.image_to_image_node import (
        ModelsLabImageToImage,
        ModelsLabUpscale,
        ModelsLabInpaint,
        ModelsLabBackgroundRemoval,
    )
    IMAGE_TO_IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_TO_IMAGE_AVAILABLE = False

try:
    from .modules.video_node import (
        ModelsLabTextToVideo,
        ModelsLabImageToVideo,
        ModelsLabVideoToVideo,
    )
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False

try:
    from .modules.tts_node import (
        ModelsLabTextToSpeech,
        ModelsLabVoiceClone,
        ModelsLabMusicGeneration,
    )
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from .modules.config_nodes import (
        ModelsLabAPIConfig,
        ModelsLabModelInfo,
        ModelsLabUsageStats,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


# Core node mappings (always available)
NODE_CLASS_MAPPINGS = {
    # Text-to-Image Generation
    "ModelsLabTextToImage": ModelsLabTextToImage,
    "ModelsLabFlux": ModelsLabFlux,
    "ModelsLabSDXL": ModelsLabSDXL,
    "ModelsLabCommunityModel": ModelsLabCommunityModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Text-to-Image Generation
    "ModelsLabTextToImage": "🎨 ModelsLab Text to Image",
    "ModelsLabFlux": "⚡ ModelsLab Flux",
    "ModelsLabSDXL": "🖼️ ModelsLab SDXL",
    "ModelsLabCommunityModel": "👥 ModelsLab Community Model",
}

# Add optional modules if available
if IMAGE_TO_IMAGE_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "ModelsLabImageToImage": ModelsLabImageToImage,
        "ModelsLabUpscale": ModelsLabUpscale,
        "ModelsLabInpaint": ModelsLabInpaint,
        "ModelsLabBackgroundRemoval": ModelsLabBackgroundRemoval,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "ModelsLabImageToImage": "🔄 ModelsLab Image to Image",
        "ModelsLabUpscale": "📈 ModelsLab Upscale",
        "ModelsLabInpaint": "🎯 ModelsLab Inpaint",
        "ModelsLabBackgroundRemoval": "✂️ ModelsLab Background Removal",
    })

if VIDEO_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "ModelsLabTextToVideo": ModelsLabTextToVideo,
        "ModelsLabImageToVideo": ModelsLabImageToVideo,
        "ModelsLabVideoToVideo": ModelsLabVideoToVideo,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "ModelsLabTextToVideo": "🎬 ModelsLab Text to Video",
        "ModelsLabImageToVideo": "🎥 ModelsLab Image to Video",
        "ModelsLabVideoToVideo": "🔁 ModelsLab Video to Video",
    })

if TTS_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "ModelsLabTextToSpeech": ModelsLabTextToSpeech,
        "ModelsLabVoiceClone": ModelsLabVoiceClone,
        "ModelsLabMusicGeneration": ModelsLabMusicGeneration,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "ModelsLabTextToSpeech": "🗣️ ModelsLab Text to Speech",
        "ModelsLabVoiceClone": "🎤 ModelsLab Voice Clone",
        "ModelsLabMusicGeneration": "🎵 ModelsLab Music Generation",
    })

if CONFIG_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "ModelsLabAPIConfig": ModelsLabAPIConfig,
        "ModelsLabModelInfo": ModelsLabModelInfo,
        "ModelsLabUsageStats": ModelsLabUsageStats,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "ModelsLabAPIConfig": "⚙️ ModelsLab API Config",
        "ModelsLabModelInfo": "ℹ️ ModelsLab Model Info",
        "ModelsLabUsageStats": "📊 ModelsLab Usage Stats",
    })


# Web directory for custom CSS/JS (if any)
WEB_DIRECTORY = "./web"

# Package metadata
__version__ = "1.0.0"
__author__ = "ModelsLab Integration Team"
__description__ = "ModelsLab API integration for ComfyUI"

# Export all mappings
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def get_extension_info():
    """Return extension information for ComfyUI-Manager."""
    return {
        "name": "ComfyUI-ModelsLab",
        "description": "ModelsLab API integration for advanced AI image generation in ComfyUI",
        "version": __version__,
        "author": __author__,
        "repository": "https://github.com/modelslab/ComfyUI-ModelsLab",
        "nodes": list(NODE_CLASS_MAPPINGS.keys()),
        "categories": [
            "ModelsLab/Generation",
            "ModelsLab/Processing", 
            "ModelsLab/Video",
            "ModelsLab/Audio",
            "ModelsLab/Community",
            "ModelsLab/Config"
        ],
        "requirements": [
            "requests>=2.28.0",
            "pillow>=9.0.0",
            "torch>=1.11.0",
            "numpy>=1.21.0",
        ],
        "features": [
            "High-quality text-to-image generation",
            "Flux, SDXL, and community models support",
            "Image-to-image processing",
            "Video generation capabilities",
            "Text-to-speech synthesis",
            "Competitive API pricing",
            "Uncensored model options",
            "Professional async processing"
        ],
        "tags": [
            "generation", "ai", "image", "video", "audio", "api", "cloud",
            "flux", "sdxl", "stable-diffusion", "text-to-image", "modelslab"
        ]
    }


# Print startup information
print("\n" + "="*60)
print("🎨 ComfyUI-ModelsLab v{} Loaded".format(__version__))
print("="*60)
print(f"📦 Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")

# Group nodes by category for display
categories = {}
for node_key, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    category = "Generation" if any(x in node_key for x in ["TextToImage", "Flux", "SDXL"]) else \
              "Processing" if any(x in node_key for x in ["ImageToImage", "Upscale", "Inpaint"]) else \
              "Video" if "Video" in node_key else \
              "Audio" if any(x in node_key for x in ["Speech", "Voice"]) else \
              "Config"
    
    if category not in categories:
        categories[category] = []
    categories[category].append(f"  • {display_name}")

for category, nodes in categories.items():
    print(f"\n{category}:")
    for node in sorted(nodes):
        print(node)

print(f"\n🔑 API Key Configuration:")
print("  • Set MODELSLAB_API_KEY environment variable, or")
print("  • Create config.ini with [modelslab] section")
print("  • Get your API key at: https://modelslab.com")

print(f"\n📖 Documentation:")
print("  • ModelsLab API Docs: https://docs.modelslab.com")
print("  • GitHub Repository: https://github.com/modelslab/ComfyUI-ModelsLab")

print("\n" + "="*60)
print("✨ Ready to generate with ModelsLab!")
print("="*60 + "\n")