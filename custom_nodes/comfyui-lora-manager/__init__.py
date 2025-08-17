from .py.lora_manager import LoraManager
from .py.nodes.lora_loader import LoraManagerLoader
from .py.nodes.trigger_word_toggle import TriggerWordToggle
from .py.nodes.lora_stacker import LoraStacker
from .py.nodes.save_image import SaveImage
from .py.nodes.debug_metadata import DebugMetadata
from .py.nodes.wanvideo_lora_select import WanVideoLoraSelect
# Import metadata collector to install hooks on startup
from .py.metadata_collector import init as init_metadata_collector

NODE_CLASS_MAPPINGS = {
    LoraManagerLoader.NAME: LoraManagerLoader,
    TriggerWordToggle.NAME: TriggerWordToggle,
    LoraStacker.NAME: LoraStacker,
    SaveImage.NAME: SaveImage,
    DebugMetadata.NAME: DebugMetadata,
    WanVideoLoraSelect.NAME: WanVideoLoraSelect
}

WEB_DIRECTORY = "./web/comfyui"

# Initialize metadata collector
init_metadata_collector()

# Register routes on import
LoraManager.add_routes()
__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']
