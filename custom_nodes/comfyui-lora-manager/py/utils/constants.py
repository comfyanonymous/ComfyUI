NSFW_LEVELS = {
    "PG": 1,
    "PG13": 2,
    "R": 4,
    "X": 8,
    "XXX": 16,
    "Blocked": 32, # Probably not actually visible through the API without being logged in on model owner account?
}

# Node type constants
NODE_TYPES = {
    "Lora Loader (LoraManager)": 1,
    "Lora Stacker (LoraManager)": 2,
    "WanVideo Lora Select (LoraManager)": 3
}

# Default ComfyUI node color when bgcolor is null
DEFAULT_NODE_COLOR = "#353535"

# preview extensions
PREVIEW_EXTENSIONS = [
    '.webp',
    '.preview.webp',
    '.preview.png',
    '.preview.jpeg',
    '.preview.jpg',
    '.preview.mp4',
    '.png',
    '.jpeg',
    '.jpg',
    '.mp4',
    '.gif',
    '.webm'
]

# Card preview image width
CARD_PREVIEW_WIDTH = 480

# Width for optimized example images
EXAMPLE_IMAGE_WIDTH = 832

# Supported media extensions for example downloads
SUPPORTED_MEDIA_EXTENSIONS = {
    'images': ['.jpg', '.jpeg', '.png', '.webp', '.gif'],
    'videos': ['.mp4', '.webm']
}

# Valid Lora types
VALID_LORA_TYPES = ['lora', 'locon', 'dora']

# Auto-organize settings
AUTO_ORGANIZE_BATCH_SIZE = 50  # Process models in batches to avoid overwhelming the system

# Civitai model tags in priority order for subfolder organization
CIVITAI_MODEL_TAGS = [
    'character', 'style', 'concept', 'clothing', 
    # 'base model', # exclude 'base model'
    'poses', 'background', 'tool', 'vehicle', 'buildings', 
    'objects', 'assets', 'animal', 'action'
]