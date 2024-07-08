from collections import OrderedDict
from copy import deepcopy

import torch
from deepspeed.utils import instrument_w_nvtx
from pathlib import Path

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class EMA(object):
    def __init__(self, args, model, device, logger):
        if args.ema_dtype == 'fp32':
            self.warmup = args.ema_warmup
            self.update_after_step = 0
            self.max_value = args.ema_decay if args.ema_decay is not None else 0.9999
            self.inv_gamma = 1.0
            self.power = args.ema_warmup_power if args.ema_warmup_power is not None else 2 / 3
            self.min_value = 0.0
        else:
            self.warmup = args.ema_warmup
            self.update_after_step = 0
            self.max_value = args.ema_decay if args.ema_decay is not None else 0.992
            self.inv_gamma = 1.0
            self.power = args.ema_warmup_power if args.ema_warmup_power is not None else 0.446249
            # 0.446249 == math.log(1 - 0.992) / math.log(50000)
            self.min_value = 0.0

        self.ema_reset_decay = args.ema_reset_decay
        self.decay_steps = 0

        if args.ema_dtype == 'none':
            ema_dtype = 'fp16' if args.use_fp16 else 'fp32'
        else:
            ema_dtype = args.ema_dtype
        
        # 由于module.half()和module.float()会发生inplace类型修改，因此需要先copy后修改类型
        self.ema_model = deepcopy(model)
        if ema_dtype == 'fp16':
            self.ema_model = self.ema_model.half().to(device)
        elif ema_dtype == 'fp32':
            self.ema_model = self.ema_model.float().to(device)
        else:
            raise ValueError(f"Unknown EMA dtype {ema_dtype}.")

        requires_grad(self.ema_model, False)

        logger.info(f"    Using EMA with date type {args.ema_dtype} "
                    f"(decay={args.ema_decay}, warmup={args.ema_warmup}, warmup_power={args.ema_warmup_power}, "
                    f"reset_decay={args.ema_reset_decay}).")

    def get_decay(self):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).

        @jarvizhang's notes on EMA max_value when enabling FP16:
            If using FP16 for EMA, max_value=0.995 is better (Don't larger than 0.999, unless you know
            what you are doing). This is because FP16 has less precision than FP32, so the EMA value can
            be pushed out of the range of FP16.

            gamma=1, power=0.446249 are good values for models (reaches decay factor 0.99 at 30K steps,
            0.992 at 50K steps).
        """
        if self.warmup:
            step = max(0, self.decay_steps - self.update_after_step - 1)
            value = 1 - (1 + step / self.inv_gamma) ** -self.power

            if step <= 0:
                return 0.0

            return max(self.min_value, min(value, self.max_value))
        else:
            return self.max_value

    @torch.no_grad()
    @instrument_w_nvtx
    def update(self, model, step, decay=None):
        """
        Step the EMA model towards the current model.

        Parameters
        ----------
        model: nn.Module
            The current model
        step: int
            The current training step. This is used to determine the decay factor. If you want to control
            the decay, you can pass in a custom step instead.
            For example, if you want to restart the EMA decay, you can pass in step=0 at start and increase
            step by step.
        decay: float
            The decay factor. If None, will be determined by the current step.
        """
        if decay is None:
            if self.ema_reset_decay:
                self.decay_steps += 1
            else:
                self.decay_steps = step
            decay = self.get_decay()

        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

        return None

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.ema_model.load_state_dict(*args, **kwargs)

    def train(self):
        self.ema_model.train()

    def eval(self):
        self.ema_model.eval()
