from __future__ import annotations
from typing import TYPE_CHECKING, Union

from comfy_api.latest import io, ComfyExtension
import comfy.patcher_extension
import logging
import torch
import math
import comfy.model_patcher
if TYPE_CHECKING:
    from uuid import UUID



def easysortblock_predict_noise_wrapper(executor, *args, **kwargs):
    # get values from args
    x: torch.Tensor = args[0]
    timestep: float = args[1]
    model_options: dict[str] = args[2]
    easycache: EasySortblockHolder = model_options["transformer_options"]["easycache"]

    # initialize predict_ratios
    if easycache.initial_step:
        sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
        relevant_sigmas = []
        for i,sigma in enumerate(sample_sigmas):
            if easycache.check_if_within_timesteps(sigma):
                relevant_sigmas.append((i, sigma))
        start_index = relevant_sigmas[0][0]
        end_index = relevant_sigmas[-1][0]
        easycache.predict_ratios = torch.linspace(easycache.start_predict_ratio, easycache.end_predict_ratio, end_index - start_index + 1)
        easycache.predict_start_index = start_index

    easycache.skip_current_step = False
    if easycache.is_past_end_timestep(timestep):
        return executor(*args, **kwargs)
    # prepare next x_prev
    next_x_prev = x
    input_change = None
    do_easycache = easycache.should_do_easycache(timestep)
    if do_easycache:
        easycache.check_metadata(x)
        if easycache.has_x_prev_subsampled():
            if easycache.has_x_prev_subsampled():
                input_change = (easycache.subsample(x, clone=False) - easycache.x_prev_subsampled).flatten().abs().mean()
            if easycache.has_output_prev_norm() and easycache.has_relative_transformation_rate():
                approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / easycache.output_prev_norm
                easycache.cumulative_change_rate += approx_output_change_rate
                if easycache.cumulative_change_rate < easycache.reuse_threshold:
                    if easycache.verbose:
                        logging.info(f"EasySortblock [verbose] - skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                    # other conds should also skip this step
                    easycache.skip_current_step = True
                    easycache.steps_skipped.append(easycache.step_count)
                else:
                    if easycache.verbose:
                        logging.info(f"EasySortblock [verbose] - NOT skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                    easycache.cumulative_change_rate = 0.0
    output: torch.Tensor = executor(*args, **kwargs)
    if easycache.has_output_prev_norm():
        output_change = (easycache.subsample(output, clone=False) - easycache.output_prev_subsampled).flatten().abs().mean()
        if easycache.verbose:
            output_change_rate = output_change / easycache.output_prev_norm
            easycache.output_change_rates.append(output_change_rate.item())
        if easycache.has_relative_transformation_rate():
            approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / easycache.output_prev_norm
            easycache.approx_output_change_rates.append(approx_output_change_rate.item())
            if easycache.verbose:
                logging.info(f"EasySortblock [verbose] - approx_output_change_rate: {approx_output_change_rate}")
        if input_change is not None:
            easycache.relative_transformation_rate = output_change / input_change
        if easycache.verbose:
            logging.info(f"EasySortblock [verbose] - output_change_rate: {output_change_rate}")
    easycache.x_prev_subsampled = easycache.subsample(next_x_prev)
    easycache.output_prev_subsampled = easycache.subsample(output)
    easycache.output_prev_norm = output.flatten().abs().mean()
    if easycache.verbose:
        logging.info(f"EasySortblock [verbose] - x_prev_subsampled: {easycache.x_prev_subsampled.shape}")

    # increment step count
    easycache.step_count += 1
    easycache.initial_step = False

    return output


def easysortblock_outer_sample_wrapper(executor, *args, **kwargs):
    """
    This OUTER_SAMPLE wrapper makes sure EasySortblock is prepped for current run, and all memory usage is cleared at the end.
    """
    try:
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        # clone and prepare timesteps
        guider.model_options["transformer_options"]["easycache"] = guider.model_options["transformer_options"]["easycache"].clone().prepare_timesteps(guider.model_patcher.model.model_sampling)
        easycache: EasySortblockHolder = guider.model_options['transformer_options']['easycache']
        logging.info(f"{easycache.name} enabled - threshold: {easycache.reuse_threshold}, start_percent: {easycache.start_percent}, end_percent: {easycache.end_percent}")
        return executor(*args, **kwargs)
    finally:
        easycache = guider.model_options['transformer_options']['easycache']
        output_change_rates = easycache.output_change_rates
        approx_output_change_rates = easycache.approx_output_change_rates
        if easycache.verbose:
            logging.info(f"{easycache.name} [verbose] - output_change_rates {len(output_change_rates)}: {output_change_rates}")
            logging.info(f"{easycache.name} [verbose] - approx_output_change_rates {len(approx_output_change_rates)}: {approx_output_change_rates}")
        total_steps = len(args[3])-1
        logging.info(f"{easycache.name} - skipped {len(easycache.steps_skipped)}/{total_steps} steps")# ({total_steps/(total_steps-easycache.total_steps_skipped):.2f}x speedup).")
        logging.info(f"{easycache.name} - skipped steps: {easycache.steps_skipped}")
        easycache.reset()
        guider.model_options = orig_model_options


def model_forward_wrapper(executor, *args, **kwargs):
    # TODO: make work with batches of conds
    transformer_options: dict[str] = args[-1]
    if not isinstance(transformer_options, dict):
        transformer_options = kwargs.get("transformer_options")
        if not transformer_options:
            transformer_options = args[-2]
    sigmas = transformer_options["sigmas"]
    sb_holder: EasySortblockHolder = transformer_options["easycache"]

    # if initial step, prepare everything for Sortblock
    if sb_holder.initial_step:
        logging.info(f"EasySortblock: inside model {executor.class_obj.__class__.__name__}")
        # TODO: generalize for other models
        # these won't stick around past this step; should store on sb_holder instead
        logging.info(f"EasySortblock: preparing {len(executor.class_obj.double_blocks)} double blocks and {len(executor.class_obj.single_blocks)} single blocks")
        if hasattr(executor.class_obj, "double_blocks"):
            for block in executor.class_obj.double_blocks:
                prepare_block(block, sb_holder)
        if hasattr(executor.class_obj, "single_blocks"):
            for block in executor.class_obj.single_blocks:
                prepare_block(block, sb_holder)
        if hasattr(executor.class_obj, "blocks"):
            for block in executor.class_obj.block:
                prepare_block(block, sb_holder)

    if sb_holder.skip_current_step:
        predict_index = max(0, sb_holder.step_count - sb_holder.predict_start_index)
        predict_ratio = sb_holder.predict_ratios[predict_index]
        logging.info(f"EasySortblock: skipping step {sb_holder.step_count}, predict_ratio: {predict_ratio}")
        # reuse_ratio = 1.0 - predict_ratio
        for block_type, blocks in sb_holder.blocks_per_type.items():
            for block in blocks:
                cache: BlockCache = block.__block_cache
                cache.allowed_to_skip = False
            sorted_blocks = sorted(blocks, key=lambda x: (x.__block_cache.consecutive_skipped_steps, x.__block_cache.prev_change_rate))
            # for block in sorted_blocks:
            #     pass
            threshold_index = int(len(sorted_blocks) * predict_ratio)
            # blocks with lower similarity are marked for recomputation
            for block in sorted_blocks[:threshold_index]:
                cache: BlockCache = block.__block_cache
                cache.allowed_to_skip = True
                logging.info(f"EasySortblock: skip block {block.__class__.__name__} - consecutive_skipped_steps: {block.__block_cache.consecutive_skipped_steps}, prev_change_rate: {block.__block_cache.prev_change_rate}, index: {block.__block_cache.block_index}")
            not_skipped  = [block for block in blocks if not block.__block_cache.allowed_to_skip]
            for block in not_skipped:
                logging.info(f"EasySortblock: reco block {block.__class__.__name__} - consecutive_skipped_steps: {block.__block_cache.consecutive_skipped_steps}, prev_change_rate: {block.__block_cache.prev_change_rate}, index: {block.__block_cache.block_index}")
            logging.info(f"EasySortblock: for {block_type}, selected {len(sorted_blocks[:threshold_index])} blocks for prediction and {len(sorted_blocks[threshold_index:])} blocks for recomputation")
        # return executor(*args, **kwargs)
    to_return = executor(*args, **kwargs)

    return to_return





def block_forward_factory(func, block):
    def block_forward_wrapper(*args, **kwargs):
        transformer_options: dict[str] = kwargs.get("transformer_options")
        sigmas = transformer_options["sigmas"]
        sb_holder: EasySortblockHolder = transformer_options["easycache"]
        cache: BlockCache = block.__block_cache
        # make sure stream count is properly set for this block
        if sb_holder.initial_step:
            sb_holder.add_to_blocks_per_type(block, transformer_options['block'][0])
            cache.block_index = transformer_options['block'][1]
            cache.stream_count = transformer_options['block'][2]

        if sb_holder.is_past_end_timestep(sigmas):
            return func(*args, **kwargs)
        # do sortblock stuff
        x = cache.get_next_x_prev(args, kwargs)
        # prepare next_x_prev
        next_x_prev = cache.get_next_x_prev(args, kwargs, clone=True)
        input_change = None
        do_sortblock = sb_holder.should_do_easycache(sigmas)
        if do_sortblock:
            # TODO: checkmetadata
            if cache.has_x_prev_subsampled():
                input_change = (cache.subsample(x, clone=False) - cache.x_prev_subsampled).flatten().abs().mean()
            if cache.has_output_prev_norm() and cache.has_relative_transformation_rate():
                approx_output_change_rate = (cache.relative_transformation_rate * input_change) / cache.output_prev_norm
                cache.cumulative_change_rate += approx_output_change_rate
                if cache.allowed_to_skip:
                # if cache.cumulative_change_rate < sb_holder.reuse_threshold:
                    # accumulate error + skip block
                    # cache.want_to_skip = True
                    # if cache.allowed_to_skip:
                    cache.consecutive_skipped_steps += 1
                    cache.prev_change_rate = approx_output_change_rate
                    return cache.apply_cache_diff(x, sb_holder)
                else:
                    # reset error; NOT skipping block and recalculating
                    cache.cumulative_change_rate = 0.0
                    cache.prev_change_rate = approx_output_change_rate
                    cache.want_to_skip = False
                    cache.consecutive_skipped_steps = 0
        # output_raw is expected to have cache.stream_count elements if count is greaater than 1 (double block, etc.)
        output_raw: Union[torch.Tensor, tuple[torch.Tensor, ...]] = func(*args, **kwargs)
        # if more than one stream from block, only use first one
        if isinstance(output_raw, tuple):
            output = output_raw[0]
        else:
            output = output_raw
        if cache.has_output_prev_norm():
            output_change = (cache.subsample(output, clone=False) - cache.output_prev_subsampled).flatten().abs().mean()
            # if verbose in future
            output_change_rate = output_change / cache.output_prev_norm
            cache.output_change_rates.append(output_change_rate.item())
            if cache.has_relative_transformation_rate():
                approx_output_change_rate = (cache.relative_transformation_rate * input_change) / cache.output_prev_norm
                cache.approx_output_change_rates.append(approx_output_change_rate.item())
            if input_change is not None:
                cache.relative_transformation_rate = output_change / input_change
        # TODO: allow cache_diff to be offloaded
        cache.update_cache_diff(output_raw, next_x_prev)
        cache.x_prev_subsampled = cache.subsample(next_x_prev)
        cache.output_prev_subsampled = cache.subsample(output)
        cache.output_prev_norm = output.flatten().abs().mean()
        return output_raw
    return block_forward_wrapper

def prepare_block(block, sb_holder: EasySortblockHolder, stream_count: int=1):
    sb_holder.add_to_all_blocks(block)
    block.__original_forward = block.forward
    block.forward = block_forward_factory(block.__original_forward, block)
    block.__block_cache = BlockCache(subsample_factor=sb_holder.subsample_factor, verbose=sb_holder.verbose)

def clean_block(block):
    block.forward = block.__original_forward
    del block.__original_forward
    del block.__block_cache

class BlockCache:
    def __init__(self, subsample_factor: int=8, verbose: bool=False):
        self.subsample_factor = subsample_factor
        self.verbose = verbose
        self.stream_count = 1
        self.block_index = 0
        # control values
        self.relative_transformation_rate: float = None
        self.cumulative_change_rate = 0.0
        self.prev_change_rate = 0.0
        # cached values
        self.x_prev_subsampled: torch.Tensor = None
        self.output_prev_subsampled: torch.Tensor = None
        self.output_prev_norm: torch.Tensor = None
        self.cache_diff: list[torch.Tensor] = []
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.steps_skipped: list[int] = []
        self.consecutive_skipped_steps = 0
        # self.state_metadata = None
        self.want_to_skip = False
        self.allowed_to_skip = False

    def has_cache_diff(self) -> bool:
        return self.cache_diff[0] is not None

    def has_x_prev_subsampled(self) -> bool:
        return self.x_prev_subsampled is not None

    def has_output_prev_subsampled(self) -> bool:
        return self.output_prev_subsampled is not None

    def has_output_prev_norm(self) -> bool:
        return self.output_prev_norm is not None

    def has_relative_transformation_rate(self) -> bool:
        return self.relative_transformation_rate is not None

    def get_next_x_prev(self, d_args: tuple[torch.Tensor, ...], d_kwargs: dict[str, torch.Tensor], clone: bool=False) -> tuple[torch.Tensor, ...]:
        if self.stream_count == 1:
            if clone:
                return d_args[0].clone()
            return d_args[0]
        keys = list(d_kwargs.keys())[:self.stream_count]
        orig_inputs = []
        for key in keys:
            if clone:
                orig_inputs.append(d_kwargs[key].clone())
            else:
                orig_inputs.append(d_kwargs[key])
        return tuple(orig_inputs)

    def subsample(self, x: Union[torch.Tensor, tuple[torch.Tensor, ...]], clone: bool = True) -> torch.Tensor:
        # subsample only the first compoenent
        if isinstance(x, tuple):
            return self.subsample(x[0], clone)
        if self.subsample_factor > 1:
            to_return = x[..., ::self.subsample_factor, ::self.subsample_factor]
            if clone:
                return to_return.clone()
            return to_return
        if clone:
            return x.clone()
        return x

    def apply_cache_diff(self, x: Union[torch.Tensor, tuple[torch.Tensor, ...]], sb_holder: EasySortblockHolder):
        self.steps_skipped.append(sb_holder.step_count)
        if not isinstance(x, tuple):
            x = (x, )
        to_return = tuple([x[i] + self.cache_diff[i] for i in range(self.stream_count)])
        if len(to_return) == 1:
            return to_return[0]
        return to_return

    def update_cache_diff(self, output_raw: Union[torch.Tensor, tuple[torch.Tensor, ...]], x: Union[torch.Tensor, tuple[torch.Tensor, ...]]):
        if not isinstance(output_raw, tuple):
            output_raw = (output_raw, )
        if not isinstance(x, tuple):
            x = (x, )
        self.cache_diff = tuple([output_raw[i] - x[i] for i in range(self.stream_count)])

    def reset(self):
        self.relative_transformation_rate = 0.0
        self.cumulative_change_rate = 0.0
        self.prev_change_rate = 0.0
        self.x_prev_subsampled = None
        self.output_prev_subsampled = None
        self.output_prev_norm = None
        self.cache_diff = []
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.steps_skipped = []
        self.consecutive_skipped_steps = 0
        self.want_to_skip = False
        self.allowed_to_skip = False
        return self


class EasySortblockHolder:
    def __init__(self, reuse_threshold: float, start_predict_ratio: float, end_predict_ratio: float, max_skipped_steps: int,
                       start_percent: float, end_percent: float, subsample_factor: int, verbose: bool=False):
        self.name = "EasySortblock"
        self.reuse_threshold = reuse_threshold
        self.start_predict_ratio = start_predict_ratio
        self.end_predict_ratio = end_predict_ratio
        self.max_skipped_steps = max_skipped_steps
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.subsample_factor = subsample_factor
        self.verbose = verbose
        # timestep values
        self.start_t = 0.0
        self.end_t = 0.0
        # control values
        self.relative_transformation_rate: float = None
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        self.step_count = 0
        self.predict_ratios = []
        self.skip_current_step = False
        self.predict_start_index = 0
        # cache values
        self.x_prev_subsampled: torch.Tensor = None
        self.output_prev_subsampled: torch.Tensor = None
        self.output_prev_norm: torch.Tensor = None
        self.steps_skipped: list[int] = []
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.state_metadata = None
        self.all_blocks = []
        self.blocks_per_type = {}

    def add_to_all_blocks(self, block):
        self.all_blocks.append(block)

    def add_to_blocks_per_type(self, block, block_type: str):
        self.blocks_per_type.setdefault(block_type, []).append(block)

    def is_past_end_timestep(self, timestep: float) -> bool:
        return not (timestep[0] > self.end_t).item()

    def should_do_easycache(self, timestep: float) -> bool:
        return (timestep[0] <= self.start_t).item()

    def check_if_within_timesteps(self, timestep: Union[float, torch.Tensor]) -> bool:
        return (timestep <= self.start_t).item() and (timestep > self.end_t).item()

    def has_x_prev_subsampled(self) -> bool:
        return self.x_prev_subsampled is not None

    def has_output_prev_subsampled(self) -> bool:
        return self.output_prev_subsampled is not None

    def has_output_prev_norm(self) -> bool:
        return self.output_prev_norm is not None

    def has_relative_transformation_rate(self) -> bool:
        return self.relative_transformation_rate is not None

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def subsample(self, x: torch.Tensor, clone: bool = True) -> torch.Tensor:
        if self.subsample_factor > 1:
            to_return = x[..., ::self.subsample_factor, ::self.subsample_factor]
            if clone:
                return to_return.clone()
            return to_return
        if clone:
            return x.clone()
        return x

    def check_metadata(self, x: torch.Tensor) -> bool:
        metadata = (x.device, x.dtype, x.shape)
        if self.state_metadata is None:
            self.state_metadata = metadata
            return True
        if metadata == self.state_metadata:
            return True
        logging.warning(f"{self.name} - Tensor shape, dtype or device changed, resetting state")
        self.reset()
        return False

    def reset(self):
        logging.info(f"EasySortblock: resetting {len(self.all_blocks)} blocks")
        for block in self.all_blocks:
            clean_block(block)
        self.relative_transformation_rate = 0.0
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        self.step_count = 0
        self.predict_ratios = []
        self.skip_current_step = False
        self.predict_start_index = 0
        self.x_prev_subsampled = None
        self.output_prev_subsampled = None
        self.output_prev_norm = None
        self.steps_skipped = []
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.state_metadata = None
        self.all_blocks = []
        self.blocks_per_type = {}
        return self

    def clone(self):
        return EasySortblockHolder(self.reuse_threshold, self.start_predict_ratio, self.end_predict_ratio, self.max_skipped_steps,
                                   self.start_percent, self.end_percent, self.subsample_factor, self.verbose)

class EasySortblockScaledNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasySortblockScaled",
            display_name="EasySortblockScaled",
            description="A homebrew version of EasyCache - even 'easier' version of EasyCache to implement. Overall works worse than EasyCache, but better in some rare cases AND universal compatibility with everything in ComfyUI.",
            category="advanced/debug/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to add Sortblock to."),
                io.Float.Input("reuse_threshold", min=0.0, default=0.2, max=3.0, step=0.01, tooltip="The threshold for reusing cached steps."),
                io.Float.Input("start_predict_ratio", min=0.0, default=0.2, max=1.0, step=0.01, tooltip="The ratio of blocks to predict."),
                io.Float.Input("end_predict_ratio", min=0.0, default=0.9, max=1.0, step=0.01, tooltip="The ratio of blocks to predict."),
                io.Int.Input("policy_refresh_interval", min=3, default=5, max=100, step=1, tooltip="The interval at which to refresh the policy."),
                io.Float.Input("start_percent", min=0.0, default=0.15, max=1.0, step=0.01, tooltip="The relative sampling step to begin use of Sortblock."),
                io.Float.Input("end_percent", min=0.0, default=0.95, max=1.0, step=0.01, tooltip="The relative sampling step to end use of Sortblock."),
                io.Boolean.Input("verbose", default=False, tooltip="Whether to log verbose information."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with Sortblock."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, reuse_threshold: float, start_predict_ratio: float, end_predict_ratio: float, policy_refresh_interval: int, start_percent: float, end_percent: float, verbose: bool) -> io.NodeOutput:
        # TODO: check for specific flavors of supported models
        model = model.clone()
        model.model_options["transformer_options"]["easycache"] = EasySortblockHolder(reuse_threshold, start_predict_ratio, end_predict_ratio, policy_refresh_interval, start_percent, end_percent, subsample_factor=8, verbose=verbose)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, "sortblock", easysortblock_predict_noise_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "sortblock", easysortblock_outer_sample_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "sortblock", model_forward_wrapper)
        return io.NodeOutput(model)


class EasySortblockExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            # EasySortblockNode,
            EasySortblockScaledNode,
        ]

def comfy_entrypoint():
    return EasySortblockExtension()

