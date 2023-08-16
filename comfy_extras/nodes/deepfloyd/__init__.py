from transformers import logging as transformers_logging
from diffusers import logging as diffusers_logging
from warnings import filterwarnings
import logging

from .deep_floyd import *

transformers_logging.set_verbosity_error()
diffusers_logging.set_verbosity_error()
logging.getLogger("xformers").addFilter(lambda r: "A matching Triton is not available" not in r.getMessage())
filterwarnings("ignore", category=FutureWarning, message="The `reduce_labels` parameter is deprecated")
filterwarnings("ignore", category=UserWarning, message="You seem to be using the pipelines sequentially on GPU")
filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

NODE_CLASS_MAPPINGS = {
    # DeepFloyd
    "IF Loader": Loader,
    "IF Encoder": Encoder,
    "IF Stage I": StageI,
    "IF Stage II": StageII,
    "IF Stage III": StageIII,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF Loader": "IF Loader",
    "IF Encoder": "IF Encoder",
    "IF Stage I": "IF Stage I",
    "IF Stage II": "IF Stage II",
    "IF Stage III": "IF Stage III",
}
