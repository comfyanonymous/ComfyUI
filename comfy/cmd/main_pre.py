import os

from .. import options

import warnings
import logging

options.enable_args_parsing()
if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")

from ..cli_args import args

if args.cuda_device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    logging.info("Set cuda device to:", args.cuda_device)

if args.deterministic:
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

__all__ = ["args"]
