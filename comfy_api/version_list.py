from comfy_api.latest import ComfyAPI_latest
from comfy_api.v0_0_2 import ComfyAPIAdapter_v0_0_2
from comfy_api.v0_0_1 import ComfyAPIAdapter_v0_0_1
from comfy_api.internal import ComfyAPIBase
from typing import List, Type

supported_versions: List[Type[ComfyAPIBase]] = [
    ComfyAPI_latest,
    ComfyAPIAdapter_v0_0_2,
    ComfyAPIAdapter_v0_0_1,
]

