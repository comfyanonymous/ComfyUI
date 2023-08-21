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
    "IFLoader": IFLoader,
    "IFEncoder": IFEncoder,
    "IFStageI": IFStageI,
    "IFStageII": IFStageII,
    "IFStageIII": IFStageIII,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IFLoader": "DeepFloyd IF Loader",
    "IFEncoder": "DeepFloyd IF Encoder",
    "IFStageI": "DeepFloyd IF Stage I",
    "IFStageII": "DeepFloyd IF Stage II",
    "IFStageIII": "DeepFloyd IF Stage III",
}
