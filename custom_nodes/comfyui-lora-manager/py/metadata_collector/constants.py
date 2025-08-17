"""Constants used by the metadata collector"""

# Metadata categories
MODELS = "models"
PROMPTS = "prompts"
SAMPLING = "sampling"
LORAS = "loras"
SIZE = "size"
IMAGES = "images"
IS_SAMPLER = "is_sampler"  # New constant to mark sampler nodes

# Complete list of categories to track
METADATA_CATEGORIES = [MODELS, PROMPTS, SAMPLING, LORAS, SIZE, IMAGES]
