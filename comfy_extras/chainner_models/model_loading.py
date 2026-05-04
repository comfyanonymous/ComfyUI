import logging
from spandrel import ModelLoader

def load_state_dict(state_dict):
    logging.warning("comfy_extras.chainner_models is deprecated and has been replaced by the spandrel library.")
    return ModelLoader().load_from_state_dict(state_dict).eval()
