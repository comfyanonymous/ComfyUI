import logging
import spandrel

logger = logging.getLogger(__name__)

# This file is deprecated and will be removed in a future version.
# Please use the spandrel library directly instead.


def load_state_dict(state_dict):
    logger.warning("comfy_extras.chainner_models is deprecated and has been replaced by the spandrel library.")
    return spandrel.ModelLoader().load_from_state_dict(state_dict).eval()
