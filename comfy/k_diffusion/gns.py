import torch
from torch import nn


class DDPGradientStatsHook:
    def __init__(self, ddp_module):
        try:
            ddp_module.register_comm_hook(self, self._hook_fn)
        except AttributeError:
            raise ValueError('DDPGradientStatsHook does not support non-DDP wrapped modules')
        self._clear_state()

    def _clear_state(self):
        self.bucket_sq_norms_small_batch = []
        self.bucket_sq_norms_large_batch = []

    @staticmethod
    def _hook_fn(self, bucket):
        buf = bucket.buffer()
        self.bucket_sq_norms_small_batch.append(buf.pow(2).sum())
        fut = torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.AVG, async_op=True).get_future()
        def callback(fut):
            buf = fut.value()[0]
            self.bucket_sq_norms_large_batch.append(buf.pow(2).sum())
            return buf
        return fut.then(callback)

    def get_stats(self):
        sq_norm_small_batch = sum(self.bucket_sq_norms_small_batch)
        sq_norm_large_batch = sum(self.bucket_sq_norms_large_batch)
        self._clear_state()
        stats = torch.stack([sq_norm_small_batch, sq_norm_large_batch])
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.AVG)
        return stats[0].item(), stats[1].item()


class GradientNoiseScale:
    """Calculates the gradient noise scale (1 / SNR), or critical batch size,
    from _An Empirical Model of Large-Batch Training_,
    https://arxiv.org/abs/1812.06162).

    Args:
        beta (float): The decay factor for the exponential moving averages used to
            calculate the gradient noise scale.
            Default: 0.9998
        eps (float): Added for numerical stability.
            Default: 1e-8
    """

    def __init__(self, beta=0.9998, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.ema_sq_norm = 0.
        self.ema_var = 0.
        self.beta_cumprod = 1.
        self.gradient_noise_scale = float('nan')

    def state_dict(self):
        """Returns the state of the object as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the object's state.
        Args:
            state_dict (dict): object state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def update(self, sq_norm_small_batch, sq_norm_large_batch, n_small_batch, n_large_batch):
        """Updates the state with a new batch's gradient statistics, and returns the
        current gradient noise scale.

        Args:
            sq_norm_small_batch (float): The mean of the squared 2-norms of microbatch or
                per sample gradients.
            sq_norm_large_batch (float): The squared 2-norm of the mean of the microbatch or
                per sample gradients.
            n_small_batch (int): The batch size of the individual microbatch or per sample
                gradients (1 if per sample).
            n_large_batch (int): The total batch size of the mean of the microbatch or
                per sample gradients.
        """
        est_sq_norm = (n_large_batch * sq_norm_large_batch - n_small_batch * sq_norm_small_batch) / (n_large_batch - n_small_batch)
        est_var = (sq_norm_small_batch - sq_norm_large_batch) / (1 / n_small_batch - 1 / n_large_batch)
        self.ema_sq_norm = self.beta * self.ema_sq_norm + (1 - self.beta) * est_sq_norm
        self.ema_var = self.beta * self.ema_var + (1 - self.beta) * est_var
        self.beta_cumprod *= self.beta
        self.gradient_noise_scale = max(self.ema_var, self.eps) / max(self.ema_sq_norm, self.eps)
        return self.gradient_noise_scale

    def get_gns(self):
        """Returns the current gradient noise scale."""
        return self.gradient_noise_scale

    def get_stats(self):
        """Returns the current (debiased) estimates of the squared mean gradient
        and gradient variance."""
        return self.ema_sq_norm / (1 - self.beta_cumprod), self.ema_var / (1 - self.beta_cumprod)
