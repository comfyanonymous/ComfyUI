"""Constants used across recipe parsers."""

# Import VALID_LORA_TYPES from utils.constants
from ..utils.constants import VALID_LORA_TYPES

# Constants for generation parameters
GEN_PARAM_KEYS = [
    'prompt',
    'negative_prompt', 
    'steps',
    'sampler',
    'cfg_scale',
    'seed',
    'size',
    'clip_skip',
]
