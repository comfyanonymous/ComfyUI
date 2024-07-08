import random
import logging
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
import math
import torch.nn.functional as F
import os

def get_trainable_params(model):
    params = model.parameters()
    params = [p for p in params if p.requires_grad]
    return params


def set_seeds(seed_list, device=None):
    if isinstance(seed_list, (tuple, list)):
        seed = sum(seed_list)
    else:
        seed = seed_list
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return torch.Generator(device).manual_seed(seed)

def get_start_epoch(resume_path, ckpt, steps_per_epoch):
    if 'epoch' in ckpt:
        start_epoch = ckpt['epoch']
    else:
        start_epoch = 0
    if 'steps' in ckpt:
        train_steps = ckpt['steps']
    else:
        try:
            train_steps = int(Path(resume_path).stem)
        except:
            train_steps = start_epoch * steps_per_epoch

    start_epoch_step = train_steps % steps_per_epoch + 1
    return start_epoch, start_epoch_step, train_steps

def assert_shape(*args):
    if len(args) < 2:
        return
    cond = True
    fail_str = f"{args[0] if isinstance(args[0], (list, tuple)) else args[0].shape}"
    for i in range(1, len(args)):
        shape1 = args[i] if isinstance(args[i], (list, tuple)) else args[i].shape
        shape2 = args[i - 1] if isinstance(args[i - 1], (list, tuple)) else args[i - 1].shape
        cond = cond and (shape1 == shape2)
        fail_str += f" vs {args[i] if isinstance(args[i], (list, tuple)) else args[i].shape}"
    assert cond, fail_str


def create_logger(logging_dir=None, logging_file=None, ddp=True):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not ddp or (ddp and dist.get_rank() == 0):  # real logger
        if logging_file is not None:
            file_handler = [logging.FileHandler(logging_file)]
        elif logging_dir is not None:
            file_handler = [logging.FileHandler(f"{logging_dir}/log.txt")]
        else:
            file_handler = []
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler()] + file_handler
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def create_exp_folder(args, rank):
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
    existed_experiments = list(Path(args.results_dir).glob("*dit*"))
    if len(existed_experiments) == 0:
        experiment_index = 1
    else:
        existed_experiments.sort()
        print('existed_experiments', existed_experiments)
        experiment_index = max([int(x.stem.split('-')[0]) for x in existed_experiments]) + 1
    dist.barrier()
    model_string_name = args.task_flag if args.task_flag else args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"       # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"                                        # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger()
        experiment_dir = ""

    return experiment_dir, checkpoint_dir, logger


def model_resume(args, model, ema, logger, len_loader):
    """
    Load pretrained weights.
    """
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    resume_path = args.resume
    if not Path(resume_path).exists():
        raise FileNotFoundError(f"    Cannot find checkpoint from {resume_path}")

    logger.info(f"    Resume deepspeed={args.resume_deepspeed}, "
                f"Resume split={args.resume_split}, "
                f"Resume from checkpoint {resume_path}")
    # Resume model and ema states (not include optimizer states) from a checkpoint saved by Deepspeed version of DIT.
    if args.resume_deepspeed:
        assert 'mp_rank_00_model_states.pt' in os.listdir(resume_path), f'    Cannot find dp chkpt from {resume_path}'
        resume_ckpt = torch.load(os.path.join(resume_path, 'mp_rank_00_model_states.pt'),
                                 map_location=lambda storage, loc: storage)
        # Resume main model
        if args.ema_to_module:
            logger.info("    Resume main model from the ema states.")
            model.load_state_dict(resume_ckpt['ema'], strict=args.strict)
        else:
            logger.info("    Resume main model from the main states.")
            model.load_state_dict(resume_ckpt['module'], strict=args.strict)
        # Resume EMA model
        if args.use_ema:
            if args.module_to_ema:
                logger.info("    Resume EMA model from the main states.")
                ema.load_state_dict(resume_ckpt['module'], strict=args.strict)
            else:
                logger.info("    Resume EMA model from the EMA states.")
                ema.load_state_dict(resume_ckpt['ema'], strict=args.strict)
        if not args.reset_loader:
            start_epoch, start_epoch_step, train_steps = get_start_epoch(args.resume, resume_ckpt, len_loader)
    # Resume model and ema states (not include optimizer states) from two checkpoints separated from DeepSpeed ckpt.
    elif args.resume_split:
        # Resume main model
        if args.ema_to_module:
            assert 'pytorch_model_ema.pt' in os.listdir(
                resume_path), f'    Cannot find pytorch_model_ema.pt from {resume_path}'
            logger.info(f"    Resume main model from ema states.")
            resume_ckpt_ema = torch.load(os.path.join(resume_path, 'pytorch_model_ema.pt'),
                                         map_location=lambda storage, loc: storage)
            model.load_state_dict(resume_ckpt_ema, strict=args.strict)
        else:
            assert 'pytorch_model_module.pt' in os.listdir(
                resume_path), f'    Cannot find pytorch_model_module.pt from {resume_path}'
            logger.info(f"    Resume main model from main states.")
            resume_ckpt_module = torch.load(os.path.join(resume_path, 'pytorch_model_module.pt'),
                                            map_location=lambda storage, loc: storage)
            model.load_state_dict(resume_ckpt_module, strict=args.strict)
        # Resume ema model
        if args.use_ema:
            if args.module_to_ema:
                if "resume_ckpt_module" in locals():
                    logger.info(f"    Resume ema model from main states.")
                    ema.load_state_dict(resume_ckpt_module, strict=args.strict)
                else:
                    assert 'pytorch_model_module.pt' in os.listdir(
                        resume_path), f'    Cannot find pytorch_model_module.pt from {resume_path}'
                    logger.info(f"    Resume ema model from module states.")
                    resume_ckpt_module = torch.load(os.path.join(resume_path, 'pytorch_model_module.pt'),
                                                    map_location=lambda storage, loc: storage)
                    ema.load_state_dict(resume_ckpt_module, strict=args.strict)
            else:
                if "resume_ckpt_ema" in locals():
                    logger.info(f"    Resume ema model from EMA states.")
                    ema.load_state_dict(resume_ckpt_ema, strict=args.strict)
                else:
                    assert 'pytorch_model_ema.pt' in os.listdir(
                        resume_path), f'    Cannot find pytorch_model_ema.pt from {resume_path}'
                    logger.info(f"    Resume ema model from EMA states.")
                    resume_ckpt_ema = torch.load(os.path.join(resume_path, 'pytorch_model_ema.pt'),
                                                 map_location=lambda storage, loc: storage)
                    ema.load_state_dict(resume_ckpt_ema, strict=args.strict)
    else:
        raise ValueError("    “If `resume` is True, then either `resume_split` or `resume_deepspeed` must be true.”")

    return model, ema, start_epoch, start_epoch_step, train_steps