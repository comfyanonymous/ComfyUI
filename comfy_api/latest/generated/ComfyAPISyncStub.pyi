from typing import Any, Dict, List, Optional, Tuple, Union, Set, Sequence, cast, NamedTuple
from comfy_api.latest import ComfyAPI_latest
from PIL.Image import Image
from torch import Tensor
class ComfyAPISyncStub:
    def __init__(self) -> None: ...

    class ExecutionSync:
        def __init__(self) -> None: ...
        """
        Update the progress bar displayed in the ComfyUI interface.

        This function allows custom nodes and API calls to report their progress
        back to the user interface, providing visual feedback during long operations.

        Migration from previous API: comfy.utils.PROGRESS_BAR_HOOK
        """
        def set_progress(self, value: float, max_value: float, node_id: Union[str, None] = None, preview_image: Union[Image, Tensor, None] = None, ignore_size_limit: bool = False) -> None: ...

    execution: ExecutionSync
