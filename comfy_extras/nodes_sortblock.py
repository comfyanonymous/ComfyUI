from __future__ import annotations
from typing import TYPE_CHECKING, Union

from scipy.sparse.linalg._dsolve.linsolve import is_pydata_spmatrix
from comfy_api.latest import io, ComfyExtension
import comfy.patcher_extension
import logging
import torch
import comfy.model_patcher
if TYPE_CHECKING:
    from uuid import UUID

def outer_sample_wrapper(executor, *args, **kwargs):
    try:
        logging.info("Sortblock: inside outer_sample!")
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        # clone and prepare timesteps
        guider.model_options["transformer_options"]["sortblock"] = guider.model_options["transformer_options"]["sortblock"].clone().prepare_timesteps(guider.model_patcher.model.model_sampling)
        sb_holder: SortblockHolder = guider.model_options["transformer_options"]["sortblock"]
        logging.info(f"Sortblock: enabled - threshold: {sb_holder.reuse_threshold}, start_percent: {sb_holder.start_percent}, end_percent: {sb_holder.end_percent}")
        return executor(*args, **kwargs)
    finally:
        sb_holder = guider.model_options["transformer_options"]["sortblock"]
        sb_holder.print_block_info(0)
        sb_holder.reset()
        guider.model_options = orig_model_options


def model_forward_wrapper(executor, *args, **kwargs):
    timestep: float = args[1]
    transformer_options: dict[str] = args[-1]
    if not isinstance(transformer_options, dict):
        transformer_options = kwargs.get("transformer_options")
        if not transformer_options:
            transformer_options = args[-2]
    logging.info(f"Sortblock: inside model {executor.class_obj.__class__.__name__}")
    sb_holder: SortblockHolder = transformer_options["sortblock"]
    sb_holder.update_should_do_sortblock(timestep)
    sb_holder.update_is_past_end_timestep(timestep)
    if sb_holder.initial_step:
        transformer_options["total_double_block"] = len(executor.class_obj.double_blocks)
        transformer_options["total_single_block"] = len(executor.class_obj.single_blocks)
        # save the original forwards on the blocks
        logging.info(f"Sortblock: preparing {transformer_options['total_double_block']} double blocks and {transformer_options['total_single_block']} single blocks")
        for block in executor.class_obj.double_blocks:
            prepare_block(block, sb_holder)
        for block in executor.class_obj.single_blocks:
            prepare_block(block, sb_holder)
    try:
        return executor(*args, **kwargs)
    finally:
        sb_holder: SortblockHolder = transformer_options["sortblock"]
        # do double blocks
        total_double_block = len(executor.class_obj.double_blocks)
        total_single_block = len(executor.class_obj.single_blocks)
        perform_sortblock(sb_holder.blocks[:total_double_block])
        #perform_sortblock(sb_holder.blocks[total_double_block:])
        if sb_holder.initial_step:
            sb_holder.initial_step = False

def perform_sortblock(blocks: list):
    candidate_blocks = []
    for block in blocks:
        cache: BlockCache = getattr(block, "__block_cache")
        cache.allowed_to_skip = False
        if cache.want_to_skip:
            candidate_blocks.append(block)
    if len(candidate_blocks) > 0:
        percentage_to_skip = 1.0
        candidate_blocks.sort(key=lambda x: getattr(x, "__block_cache").cumulative_change_rate)
        blocks_to_skip = int(len(candidate_blocks) * percentage_to_skip)
        for block in candidate_blocks[:blocks_to_skip]:
            cache: BlockCache = getattr(block, "__block_cache")
            cache.allowed_to_skip = True



def prepare_block(block, sb_holder: SortblockHolder, stream_count: int=1):
    sb_holder.add_block(block)
    block.__original_forward = block.forward
    block.forward = block_forward_factory(block.__original_forward, block)
    block.__block_cache = BlockCache(subsample_factor=sb_holder.subsample_factor, verbose=sb_holder.verbose)


def clean_block(block):
    block.forward = block.__original_forward
    del block.__original_forward
    del block.__block_cache


def block_forward_factory(func, block):
    def block_forward_wrapper(*args, **kwargs):
        transformer_options: dict[str] = kwargs.get("transformer_options", None)
        #logging.info(f"Sortblock: inside block {transformer_options['block']}")
        sb_holder: SortblockHolder = transformer_options["sortblock"]
        cache: BlockCache = block.__block_cache
        if sb_holder.initial_step:
            cache.stream_count = transformer_options['block'][2]
        if sb_holder.is_past_end_timestep():
            return func(*args, **kwargs)
        # do sortblock stuff
        keys = list(kwargs.keys())
        x = cache.get_next_x_prev(kwargs)
        timestep: float = sb_holder.curr_t
        # prepare next_x_prev
        next_x_prev = cache.get_next_x_prev(kwargs)
        input_change = None
        do_sortblock = sb_holder.should_do_sortblock()
        if do_sortblock:
            # TODO: checkmetadata
            if cache.has_x_prev_subsampled():
                input_change = (cache.subsample(x, clone=False) - cache.x_prev_subsampled).flatten().abs().mean()
            if cache.has_output_prev_norm() and cache.has_relative_transformation_rate():
                approx_output_change_rate = (cache.relative_transformation_rate * input_change) / cache.output_prev_norm
                cache.cumulative_change_rate += approx_output_change_rate
                if cache.cumulative_change_rate < sb_holder.reuse_threshold:
                    # accumulate error + skip block
                    cache.want_to_skip = True
                    if cache.allowed_to_skip:
                        return cache.apply_cache_diff(x)
                else:
                    # reset error; NOT skipping block and recalculating
                    cache.cumulative_change_rate = 0.0
                    cache.want_to_skip = False
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


class SortblockHolder:
    def __init__(self, reuse_threshold: float, start_percent: float, end_percent: float, subsample_factor: int=8, verbose: bool=False):
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.subsample_factor = subsample_factor
        self.verbose = verbose
        # timestep values
        self.start_t = 0.0
        self.end_t = 0.0
        self.curr_t = 0.0
        # control values
        self.past_timestep = False
        self.do_sortblock = False
        self.initial_step = True
        # cache values
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def update_is_past_end_timestep(self, timestep: float) -> bool:
        self.past_timestep = not (timestep[0] > self.end_t).item()
        return self.past_timestep

    def is_past_end_timestep(self) -> bool:
        return self.past_timestep

    def update_should_do_sortblock(self, timestep: float) -> bool:
        self.do_sortblock = (timestep[0] <= self.start_t).item()
        self.curr_t = timestep
        return self.do_sortblock

    def should_do_sortblock(self) -> bool:
        return self.do_sortblock

    def print_block_info(self, index: int):
        block = self.blocks[index]
        cache = getattr(block, "__block_cache")
        logging.info(f"Sortblock: block {index} output_change_rates: {cache.output_change_rates}")
        logging.info(f"Sortblock: block {index} approx_output_change_rates: {cache.approx_output_change_rates}")

    def reset(self):
        self.initial_step = True
        self.curr_t = 0.0
        logging.info(f"Sortblock: resetting {len(self.blocks)} blocks")
        for block in self.blocks:
            clean_block(block)
        self.blocks = []
        return self

    def clone(self):
        return SortblockHolder(self.reuse_threshold, self.start_percent, self.end_percent, self.subsample_factor, self.verbose)


class BlockCache:
    def __init__(self, subsample_factor: int=8, stream_count: int=1, verbose: bool=False):
        self.subsample_factor = subsample_factor
        self.stream_count = stream_count
        self.verbose = verbose
        # control values
        self.relative_transformation_rate: float = None
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        # cache values
        self.x_prev_subsampled: torch.Tensor = None
        self.output_prev_subsampled: torch.Tensor = None
        self.output_prev_norm: torch.Tensor = None
        self.cache_diff: list[torch.Tensor] = [None for _ in range(stream_count)]
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.total_steps_skipped = 0
        self.state_metadata = None
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

    def get_next_x_prev(self, d_kwargs: dict[str, torch.Tensor]) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        keys = list(d_kwargs.keys())
        if self.stream_count == 1:
            return d_kwargs[keys[0]]
        return tuple([d_kwargs[keys[i]] for i in range(self.stream_count)])

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

    def apply_cache_diff(self, x: Union[torch.Tensor, tuple[torch.Tensor, ...]]):
        self.total_steps_skipped += 1
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

    def check_metadata(self, x: torch.Tensor) -> bool:
        # TODO: make sure shapes are correct
        metadata = (x.device, x.dtype, x.shape)
        if self.state_metadata is None:
            self.state_metadata = metadata
            return True
        if metadata == self.state_metadata:
            return True
        logging.warn(f"{self.name} - Tensor shape, dtype or device changed, resetting state")
        self.reset()
        return False

    def reset(self):
        self.relative_transformation_rate = 0.0
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        self.cache_diff = [None for _ in range(self.stream_count)]
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.total_steps_skipped = 0
        self.state_metadata = None
        self.want_to_skip = False
        self.allowed_to_skip = False

class SortblockNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sortblock",
            display_name="Sortblock",
            description="A homebrew version of EasyCache - even 'easier' version of EasyCache to implement. Overall works worse than EasyCache, but better in some rare cases AND universal compatibility with everything in ComfyUI.",
            category="advanced/debug/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to add Sortblock to."),
                io.Float.Input("reuse_threshold", min=0.0, default=0.2, max=3.0, step=0.01, tooltip="The threshold for reusing cached blocks."),
                io.Float.Input("start_percent", min=0.0, default=0.15, max=1.0, step=0.01, tooltip="The relative sampling step to begin use of Sortblock."),
                io.Float.Input("end_percent", min=0.0, default=0.95, max=1.0, step=0.01, tooltip="The relative sampling step to end use of Sortblock."),
                io.Boolean.Input("verbose", default=False, tooltip="Whether to log verbose information."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with Sortblock."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, reuse_threshold: float, start_percent: float, end_percent: float, verbose: bool) -> io.NodeOutput:
        # TODO: check for specific flavors of supported models
        model = model.clone()
        model.model_options["transformer_options"]["sortblock"] = SortblockHolder(reuse_threshold, start_percent, end_percent, subsample_factor=8, verbose=verbose)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "sortblock", outer_sample_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "sortblock", model_forward_wrapper)
        # model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "lazycache", easycache_sample_wrapper)
        # model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, "lazycache", lazycache_predict_noise_wrapper)
        return io.NodeOutput(model)


class SortblockExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SortblockNode,
        ]

def comfy_entrypoint():
    return SortblockExtension()
