"""
Implementation of learning rate schedules.

Taken and modified from PyTorch v1.0.1 source
https://github.com/pytorch/pytorch/blob/v1.1.0/torch/optim/lr_scheduler.py
"""

import argparse
from torch.optim import Optimizer
import math

LR_SCHEDULE = 'lr_schedule'
LR_RANGE_TEST = 'LRRangeTest'
ONE_CYCLE = 'OneCycle'
WARMUP_LR = 'WarmupLR'
WARMUP_DECAY_LR = 'WarmupDecayLR'
VALID_LR_SCHEDULES = [LR_RANGE_TEST, ONE_CYCLE, WARMUP_LR, WARMUP_DECAY_LR]

LR_RANGE_TEST_MIN_LR = 'lr_range_test_min_lr'
LR_RANGE_TEST_STEP_RATE = 'lr_range_test_step_rate'
LR_RANGE_TEST_STEP_SIZE = 'lr_range_test_step_size'
LR_RANGE_TEST_STAIRCASE = 'lr_range_test_staircase'

EDGE_VALUE = 'edge_value'
MID_VALUE = 'mid_value'

CYCLE_FIRST_STEP_SIZE = 'cycle_first_step_size'
CYCLE_FIRST_STAIR_COUNT = 'cycle_first_stair_count'
CYCLE_SECOND_STEP_SIZE = 'cycle_second_step_size'
CYCLE_SECOND_STAIR_COUNT = 'cycle_second_stair_count'
DECAY_STEP_SIZE = 'decay_step_size'

CYCLE_MIN_LR = 'cycle_min_lr'
CYCLE_MAX_LR = 'cycle_max_lr'
DECAY_LR_RATE = 'decay_lr_rate'

CYCLE_MIN_MOM = 'cycle_min_mom'
CYCLE_MAX_MOM = 'cycle_max_mom'
DECAY_MOM_RATE = 'decay_mom_rate'

WARMUP_MIN_LR = 'warmup_min_lr'
WARMUP_MAX_LR = 'warmup_max_lr'
WARMUP_NUM_STEPS = 'warmup_num_steps'
WARMUP_TYPE = 'warmup_type'
WARMUP_LOG_RATE = 'log'
WARMUP_LINEAR_RATE = 'linear'

TOTAL_NUM_STEPS = 'total_num_steps'


def add_tuning_arguments(parser):
    group = parser.add_argument_group('Convergence Tuning', 'Convergence tuning configurations')

    # LR scheduler
    group.add_argument('--lr_schedule', type=str, default=None, help='LR schedule for training.')

    # Learning rate range test
    group.add_argument("--lr_range_test_min_lr", type=float, default=0.001, help='Starting lr value.')
    group.add_argument("--lr_range_test_step_rate", type=float, default=1.0, help='scaling rate for LR range test.')
    group.add_argument("--lr_range_test_step_size", type=int, default=1000, help='training steps per LR change.')
    group.add_argument("--lr_range_test_staircase",
                       type=bool,
                       default=False,
                       help='use staircase scaling for LR range test.')

    # OneCycle schedule
    group.add_argument("--cycle_first_step_size",
                       type=int,
                       default=1000,
                       help='size of first step of 1Cycle schedule (training steps).')
    group.add_argument("--cycle_first_stair_count",
                       type=int,
                       default=-1,
                       help='first stair count for 1Cycle schedule.')
    group.add_argument("--cycle_second_step_size",
                       type=int,
                       default=-1,
                       help='size of second step of 1Cycle schedule (default first_step_size).')
    group.add_argument("--cycle_second_stair_count",
                       type=int,
                       default=-1,
                       help='second stair count for 1Cycle schedule.')
    group.add_argument("--decay_step_size",
                       type=int,
                       default=1000,
                       help='size of intervals for applying post cycle decay (training steps).')

    # 1Cycle LR
    group.add_argument("--cycle_min_lr", type=float, default=0.01, help='1Cycle LR lower bound.')
    group.add_argument("--cycle_max_lr", type=float, default=0.1, help='1Cycle LR upper bound.')
    group.add_argument("--decay_lr_rate", type=float, default=0.0, help='post cycle LR decay rate.')

    # 1Cycle Momentum
    group.add_argument('--cycle_momentum', default=False, action='store_true', help='Enable 1Cycle momentum schedule.')
    group.add_argument("--cycle_min_mom", type=float, default=0.8, help='1Cycle momentum lower bound.')
    group.add_argument("--cycle_max_mom", type=float, default=0.9, help='1Cycle momentum upper bound.')
    group.add_argument("--decay_mom_rate", type=float, default=0.0, help='post cycle momentum decay rate.')

    # Warmup LR
    group.add_argument('--warmup_min_lr', type=float, default=0, help='WarmupLR minimum/initial LR value')
    group.add_argument('--warmup_max_lr', type=float, default=0.001, help='WarmupLR maximum LR value.')
    group.add_argument('--warmup_num_steps', type=int, default=1000, help='WarmupLR step count for LR warmup.')
    group.add_argument('--warmup_type',
                       type=str,
                       default=WARMUP_LOG_RATE,
                       help='WarmupLR increasing function during warmup')
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = add_tuning_arguments(parser)

    lr_sched_args, unknown_args = parser.parse_known_args()
    return lr_sched_args, unknown_args


def override_lr_range_test_params(args, params):
    if hasattr(args, LR_RANGE_TEST_MIN_LR) and args.lr_range_test_min_lr is not None:
        params[LR_RANGE_TEST_MIN_LR] = args.lr_range_test_min_lr

    if hasattr(args, LR_RANGE_TEST_STEP_RATE) and args.lr_range_test_step_rate is not None:
        params[LR_RANGE_TEST_STEP_RATE] = args.lr_range_test_step_rate

    if hasattr(args, LR_RANGE_TEST_STEP_SIZE) and args.lr_range_test_step_size is not None:
        params[LR_RANGE_TEST_STEP_SIZE] = args.lr_range_test_step_size

    if hasattr(args, LR_RANGE_TEST_STAIRCASE) and args.lr_range_test_staircase is not None:
        params[LR_RANGE_TEST_STAIRCASE] = args.lr_range_test_staircase


def override_1cycle_params(args, params):
    if hasattr(args, CYCLE_FIRST_STEP_SIZE) and args.cycle_first_step_size is not None:
        params[CYCLE_FIRST_STEP_SIZE] = args.cycle_first_step_size

    if hasattr(args, CYCLE_FIRST_STAIR_COUNT) and args.cycle_first_stair_count is not None:
        params[CYCLE_FIRST_STAIR_COUNT] = args.cycle_first_stair_count

    if hasattr(args, CYCLE_SECOND_STEP_SIZE) and args.cycle_second_step_size is not None:
        params[CYCLE_SECOND_STEP_SIZE] = args.cycle_second_step_size

    if hasattr(args, CYCLE_SECOND_STAIR_COUNT) and args.cycle_second_stair_count is not None:
        params[CYCLE_SECOND_STAIR_COUNT] = args.cycle_second_stair_count

    if hasattr(args, DECAY_STEP_SIZE) and args.decay_step_size is not None:
        params[DECAY_STEP_SIZE] = args.decay_step_size

    # 1Cycle LR params
    if hasattr(args, CYCLE_MIN_LR) and args.cycle_min_lr is not None:
        params[CYCLE_MIN_LR] = args.cycle_min_lr

    if hasattr(args, CYCLE_MAX_LR) and args.cycle_max_lr is not None:
        params[CYCLE_MAX_LR] = args.cycle_max_lr

    if hasattr(args, DECAY_LR_RATE) and args.decay_lr_rate is not None:
        params[DECAY_LR_RATE] = args.decay_lr_rate

    # 1Cycle MOM params
    if hasattr(args, CYCLE_MIN_MOM) and args.cycle_min_mom is not None:
        params[CYCLE_MIN_MOM] = args.cycle_min_mom

    if hasattr(args, CYCLE_MAX_MOM) and args.cycle_max_mom is not None:
        params[CYCLE_MAX_MOM] = args.cycle_max_mom

    if hasattr(args, DECAY_MOM_RATE) and args.decay_mom_rate is not None:
        params[DECAY_MOM_RATE] = args.decay_mom_rate


def override_warmupLR_params(args, params):
    if hasattr(args, WARMUP_MIN_LR) and args.warmup_min_lr is not None:
        params[WARMUP_MIN_LR] = args.warmup_min_lr

    if hasattr(args, WARMUP_MAX_LR) and args.warmup_max_lr is not None:
        params[WARMUP_MAX_LR] = args.warmup_max_lr

    if hasattr(args, WARMUP_NUM_STEPS) and args.warmup_num_steps is not None:
        params[WARMUP_NUM_STEPS] = args.warmup_num_steps

    if hasattr(args, WARMUP_TYPE) and args.warmup_type is not None:
        params[WARMUP_TYPE] = args.warmup_type


def override_params(args, params):
    # LR range test params
    override_lr_range_test_params(args, params)

    # 1Cycle params
    override_1cycle_params(args, params)

    # WarmupLR params
    override_warmupLR_params(args, params)


def get_config_from_args(args):
    if not hasattr(args, LR_SCHEDULE) or args.lr_schedule is None:
        return None, '--{} not specified on command line'.format(LR_SCHEDULE)

    if not args.lr_schedule in VALID_LR_SCHEDULES:
        return None, '{} is not supported LR schedule'.format(args.lr_schedule)

    config = {}
    config['type'] = args.lr_schedule
    config['params'] = {}

    if args.lr_schedule == LR_RANGE_TEST:
        override_lr_range_test_params(args, config['params'])
    elif args.lr_schedule == ONE_CYCLE:
        override_1cycle_params(args, config['params'])
    else:
        override_warmupLR_params(args, config['params'])

    return config, None


def get_lr_from_config(config):
    if not 'type' in config:
        return None, 'LR schedule type not defined in config'

    if not 'params' in config:
        return None, 'LR schedule params not defined in config'

    lr_schedule = config['type']
    lr_params = config['params']

    if not lr_schedule in VALID_LR_SCHEDULES:
        return None, '{} is not a valid LR schedule'.format(lr_schedule)

    if lr_schedule == LR_RANGE_TEST:
        return lr_params[LR_RANGE_TEST_MIN_LR], ''
    if lr_schedule == ONE_CYCLE:
        return lr_params[CYCLE_MAX_LR], ''
    # Warmup LR
    return lr_params[WARMUP_MAX_LR], ''


"""
Only optimizers that are subclass of torch.optim.Optimizer are supported. So check the passed optimizer and wrapped
optimizer to see if requirement is satisfied.
TODO: Looking under the hood to examine the wrapped optimizer is a hack that requires a better long-term fix.
"""


def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError('{} is not a subclass of torch.optim.Optimizer'.format(type(optimizer).__name__))


class LRRangeTest(object):
    """Sets the learning rate of each parameter group according to
    learning rate range test (LRRT) policy. The policy increases learning
    rate starting from a base value with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters: Part1`_.

    LRRT policy is used for finding maximum LR that trains a model without divergence, and can be used to
    configure the LR boundaries for Cyclic LR schedules.

    LRRT changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_range_test_min_lr (float or list): Initial learning rate which is the
            lower boundary in the range test for each parameter group.
        lr_range_test_step_size (int): Interval of training steps to increase learning rate. Default: 2000
        lr_range_test_step_rate (float): Scaling rate for range test. Default: 1.0
        lr_range_test_staircase (bool): Scale in staircase fashion, rather than continuous. Default: False.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = LRRangeTest(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

        _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay:
        https://arxiv.org/abs/1803.09820
"""

    def __init__(self,
                 optimizer: Optimizer,
                 lr_range_test_min_lr: float = 1e-3,
                 lr_range_test_step_size: int = 2000,
                 lr_range_test_step_rate: float = 1.0,
                 lr_range_test_staircase: bool = False,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        if isinstance(lr_range_test_min_lr, list) or isinstance(lr_range_test_min_lr, tuple):
            if len(lr_range_test_min_lr) != len(self.optimizer.param_groups):
                raise ValueError("expected {} lr_range_test_min_lr, got {}".format(len(self.optimizer.param_groups),
                                                                                   len(lr_range_test_min_lr)))
            self.min_lr = list(lr_range_test_min_lr)
        else:
            self.min_lr = [lr_range_test_min_lr] * len(self.optimizer.param_groups)

        self.step_size = lr_range_test_step_size
        self.step_rate = lr_range_test_step_rate
        self.last_batch_iteration = last_batch_iteration
        self.staircase = lr_range_test_staircase
        self.interval_fn = self._staircase_interval if lr_range_test_staircase else self._continuous_interval

        if last_batch_iteration == -1:
            self._update_optimizer(self.min_lr)

    def _staircase_interval(self):
        return math.floor(float(self.last_batch_iteration + 1) / self.step_size)

    def _continuous_interval(self):
        return float(self.last_batch_iteration + 1) / self.step_size

    def _get_increase(self):
        return (1 + self.step_rate * self.interval_fn())

    def get_lr(self):
        lr_increase = self._get_increase()
        return [lr_range_test_min_lr * lr_increase for lr_range_test_min_lr in self.min_lr]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def _update_optimizer(self, group_lrs):
        for param_group, lr in zip(self.optimizer.param_groups, group_lrs):
            param_group['lr'] = lr

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        self._update_optimizer(self.get_lr())
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']



class OneCycle(object):
    """Sets the learning rate of each parameter group according to
    1Cycle learning rate policy (1CLR). 1CLR is a variation of the
    Cyclical Learning Rate (CLR) policy that involves one cycle followed by
    decay. The policy simultaneously cycles the learning rate (and momentum)
    between two boundaries with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters`_.

    1CLR policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This implementation was adapted from the github repo: `pytorch/pytorch`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        cycle_min_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        cycle_max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_lr - cycle_min_lr).
            The lr at any cycle is the sum of cycle_min_lr
            and some scaling of the amplitude; therefore
            cycle_max_lr may not actually be reached depending on
            scaling function.
        decay_lr_rate(float): Decay rate for learning rate. Default: 0.
        cycle_first_step_size (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        cycle_second_step_size (int): Number of training iterations in the
            decreasing half of a cycle. If cycle_second_step_size is None,
            it is set to cycle_first_step_size. Default: None
        cycle_first_stair_count(int): Number of stairs in first half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        cycle_second_stair_count(int): Number of stairs in second half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        decay_step_size (int): Intervals for applying decay in decay phase. Default: 0, means no decay.
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'cycle_min_mom' and 'cycle_max_mom'.
            Default: True
        cycle_min_mom (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        cycle_max_mom (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_mom - cycle_min_mom).
            The momentum at any cycle is the difference of cycle_max_mom
            and some scaling of the amplitude; therefore
            cycle_min_mom may not actually be reached depending on
            scaling function. Default: 0.9
        decay_mom_rate (float): Decay rate for momentum. Default: 0.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = OneCycle(optimizer, 0.0001, 0.0010)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay: https://arxiv.org/abs/1803.09820
    """

    def __init__(self,
                 optimizer,
                 cycle_min_lr,
                 cycle_max_lr,
                 decay_lr_rate=0.,
                 cycle_first_step_size=2000,
                 cycle_second_step_size=None,
                 cycle_first_stair_count=0,
                 cycle_second_stair_count=None,
                 decay_step_size=0,
                 cycle_momentum=True,
                 cycle_min_mom=0.8,
                 cycle_max_mom=0.9,
                 decay_mom_rate=0.,
                 last_batch_iteration=-1):

        self.optimizer = get_torch_optimizer(optimizer)

        # Initialize cycle shape
        self._initialize_cycle(cycle_first_step_size, cycle_second_step_size, cycle_first_stair_count,
                               cycle_second_stair_count, decay_step_size)

        # Initialize cycle lr
        self._initialize_lr(self.optimizer, cycle_min_lr, cycle_max_lr, decay_lr_rate, last_batch_iteration)

        # Initialize cyclic momentum
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            self._initialize_momentum(self.optimizer, cycle_min_mom, cycle_max_mom, decay_mom_rate,
                                      last_batch_iteration)

        # Initialize batch iteration tracker
        self.last_batch_iteration = last_batch_iteration

    # Configure cycle shape

    def _initialize_cycle(self, cycle_first_step_size, cycle_second_step_size, cycle_first_stair_count,
                          cycle_second_stair_count, decay_step_size):
        cycle_first_step_size = float(cycle_first_step_size)
        cycle_second_step_size = float(
            cycle_second_step_size) if cycle_second_step_size is not None else cycle_first_step_size

        self.total_size = cycle_first_step_size + cycle_second_step_size
        self.step_ratio = cycle_first_step_size / self.total_size
        self.first_stair_count = cycle_first_stair_count
        self.second_stair_count = cycle_first_stair_count if cycle_second_stair_count is None else cycle_second_stair_count
        self.decay_step_size = decay_step_size

        if math.isclose(self.decay_step_size, 0):
            self.skip_lr_decay = True
            self.skip_mom_decay = True
        else:
            self.skip_lr_decay = False
            self.skip_mom_decay = False

    # Configure lr schedule
    def _initialize_lr(self, optimizer, cycle_min_lr, cycle_max_lr, decay_lr_rate, last_batch_iteration):
        self.min_lrs = [cycle_min_lr] * len(optimizer.param_groups)
        if last_batch_iteration == -1:
            for lr, group in zip(self.min_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = [cycle_max_lr] * len(optimizer.param_groups)
        self.decay_lr_rate = decay_lr_rate

        if math.isclose(self.decay_lr_rate, 0):
            self.skip_lr_decay = True

    # Configure momentum schedule
    def _initialize_momentum(self, optimizer, cycle_min_mom, cycle_max_mom, decay_mom_rate, last_batch_iteration):
        if 'betas' not in optimizer.defaults:
            optimizer_name = type(optimizer).__name__
            print(
                f"cycle_momentum is disabled because optimizer {optimizer_name} does not support momentum, no betas attribute in defaults"
            )
            self.cycle_momentum = False
            return

        self.decay_mom_rate = decay_mom_rate
        self.min_moms = [(cycle_min_mom, 0.99)] * len(optimizer.param_groups)
        self.max_moms = [(cycle_max_mom, 0.99)] * len(optimizer.param_groups)

        if last_batch_iteration == -1:
            for momentum, group in zip(self.min_moms, optimizer.param_groups):
                group['betas'] = momentum

        if math.isclose(self.decay_mom_rate, 0):
            self.skip_mom_decay = True

    def _get_scale_factor(self):
        batch_iteration = (self.last_batch_iteration + 1)
        cycle = math.floor(1 + batch_iteration / self.total_size)
        x = 1. + batch_iteration / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        return scale_factor

    def _get_cycle_mom(self):
        scale_factor = self._get_scale_factor()
        momentums = []
        for base_betas, max_betas in zip(self.min_moms, self.max_moms):
            cycle_min_mom = base_betas[0]
            cycle_max_mom = max_betas[0]
            base_height = (cycle_max_mom - cycle_min_mom) * scale_factor
            momentum = cycle_max_mom - base_height
            momentums.append((momentum, base_betas[1]))
        return momentums

    def _get_cycle_lr(self):
        scale_factor = self._get_scale_factor()
        lrs = []
        for cycle_min_lr, cycle_max_lr in zip(self.min_lrs, self.max_lrs):
            base_height = (cycle_max_lr - cycle_min_lr) * scale_factor
            lr = cycle_min_lr + base_height
            lrs.append(lr)

        return lrs

    def _get_decay_mom(self, decay_batch_iteration):
        if self.skip_mom_decay:
            return self.max_moms

        decay_interval = decay_batch_iteration / self.decay_step_size
        mom_decay_factor = (1 + self.decay_mom_rate * decay_interval)
        momentums = [(beta0 * mom_decay_factor, beta1) for beta0, beta1 in self.max_moms]

        return momentums

    def _get_decay_lr(self, decay_batch_iteration):
        """Calculates the learning rate at batch index. This function is used
        after the cycle completes and post cycle decaying of lr/mom is enabled.
        This function treats `self.last_batch_iteration` as the last batch index.
        """
        if self.skip_lr_decay:
            return self.min_lrs

        decay_interval = decay_batch_iteration / self.decay_step_size
        lr_decay_factor = (1 + self.decay_lr_rate * decay_interval)
        lrs = [cycle_min_lr / lr_decay_factor for cycle_min_lr in self.min_lrs]

        return lrs

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_batch_iteration` as the last batch index.
        """
        if self.last_batch_iteration < self.total_size:
            return self._get_cycle_lr()
        return self._get_decay_lr(self.last_batch_iteration - self.total_size + 1)

    def get_mom(self):
        """Calculates the momentum at batch index. This function treats
        `self.last_batch_iteration` as the last batch index.
        """
        if not self.cycle_momentum:
            return None

        if self.last_batch_iteration < self.total_size:
            return self._get_cycle_mom()
        return self._get_decay_mom(self.last_batch_iteration - self.total_size + 1)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, batch_iteration=None):
        """ Updates the optimizer with the learning rate for the last batch index.
        `self.last_batch_iteration` is treated as the last batch index.

        If self.cycle_momentum is true, also updates optimizer momentum.
        """
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1

        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        if self.cycle_momentum:
            momentums = self.get_mom()
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['betas'] = momentum

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']



class WarmupLR(object):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then fix at max lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            warmup_type {‘log’, ‘linear’}: increasing function from min_lr to max_lr during warmup. Default: log
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = WarmupLR(optimizer)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 warmup_type: str = WARMUP_LOG_RATE,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        self.min_lrs = self._format_param(self.optimizer, warmup_min_lr, "min_lr")
        self.max_lrs = self._format_param(self.optimizer, warmup_max_lr, "max_lr")
        self.delta_lrs = [big - small for big, small in zip(self.max_lrs, self.min_lrs)]
        self.warmup_num_steps = max(2, warmup_num_steps)
        # Currently only support linear and log function
        if warmup_type not in {WARMUP_LOG_RATE, WARMUP_LINEAR_RATE}:
            print(f"Using unknown warmup_type: {warmup_type}. The increasing function "
                           f"is set to default (log)")
            warmup_type = WARMUP_LOG_RATE
        self.warmup_type = warmup_type
        self.inverse_log_warm_up = 1.0 / math.log(self.warmup_num_steps)
        self.last_batch_iteration = last_batch_iteration

    def get_lr(self):
        if self.last_batch_iteration < 0:
            print("Attempting to get learning rate from scheduler before it has started")
            return [0.0]
        gamma = self._get_gamma()
        return [min_lr + (delta_lr * gamma) for min_lr, delta_lr in zip(self.min_lrs, self.delta_lrs)]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        return 1.0

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(len(optimizer.param_groups), param_name,
                                                                           FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)



class WarmupDecayLR(WarmupLR):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then decay at linear rate over the remaining training steps.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_num_steps (int): total number of training steps
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            warmup_type {‘log’, ‘linear’}: increasing function from min_lr to max_lr during warmup. Default: log
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = WarmupDecayLR(optimizer, 1000000)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """

    def __init__(self,
                 optimizer: Optimizer,
                 total_num_steps: int,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 warmup_type: str = WARMUP_LOG_RATE,
                 last_batch_iteration: int = -1):

        self.total_num_steps = total_num_steps
        super(WarmupDecayLR, self).__init__(optimizer, warmup_min_lr, warmup_max_lr, warmup_num_steps, warmup_type,
                                            last_batch_iteration)
        if self.total_num_steps < self.warmup_num_steps:
            print('total_num_steps {} is less than warmup_num_steps {}'.format(
                total_num_steps, warmup_num_steps))

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        return max(
            0.0,
            float(self.total_num_steps - self.last_batch_iteration) /
            float(max(1.0, self.total_num_steps - self.warmup_num_steps)))
