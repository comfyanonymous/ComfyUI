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


def prepare_noise_wrapper(executor, *args, **kwargs):
    try:
        return executor(*args, **kwargs)
    finally:
        sb_holder: SortblockHolder = executor.class_obj.model_options["transformer_options"]["sortblock"]
        sb_holder.step_count += 1


def outer_sample_wrapper(executor, *args, **kwargs):
    try:
        logging.info("Sortblock: inside outer_sample!")
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        # clone and prepare timesteps
        sb_holder = guider.model_options["transformer_options"]["sortblock"]
        guider.model_options["transformer_options"]["sortblock"] = sb_holder.clone().prepare_timesteps(guider.model_patcher.model.model_sampling)
        sb_holder: SortblockHolder = guider.model_options["transformer_options"]["sortblock"]
        logging.info(f"Sortblock: enabled - threshold: {sb_holder.predict_ratio}, start_percent: {sb_holder.start_percent}, end_percent: {sb_holder.end_percent}")
        return executor(*args, **kwargs)
    finally:
        sb_holder = guider.model_options["transformer_options"]["sortblock"]
        logging.info(f"Sortblock: final step count: {sb_holder.step_count}")
        sb_holder.reset()
        guider.model_options = orig_model_options


def model_forward_wrapper(executor, *args, **kwargs):
    # TODO: make work with batches of conds
    transformer_options: dict[str] = args[-1]
    if not isinstance(transformer_options, dict):
        transformer_options = kwargs.get("transformer_options")
        if not transformer_options:
            transformer_options = args[-2]
    sigmas = transformer_options["sigmas"]
    sb_holder: SortblockHolder = transformer_options["sortblock"]
    sb_holder.update_should_do_sortblock(sigmas)

    # if initial step, prepare everything for Sortblock
    if sb_holder.initial_step:
        logging.info(f"Sortblock: inside model {executor.class_obj.__class__.__name__}")
        # TODO: generalize for other models
        # these won't stick around past this step; should store on sb_holder instead
        logging.info(f"Sortblock: preparing {len(executor.class_obj.double_blocks)} double blocks and {len(executor.class_obj.single_blocks)} single blocks")
        if hasattr(executor.class_obj, "double_blocks"):
            for block in executor.class_obj.double_blocks:
                prepare_block(block, sb_holder)
        if hasattr(executor.class_obj, "single_blocks"):
            for block in executor.class_obj.single_blocks:
                prepare_block(block, sb_holder)
        if hasattr(executor.class_obj, "blocks"):
            for block in executor.class_obj.block:
                prepare_block(block, sb_holder)

    # when 0: Initialization(1)
    if sb_holder.step_modulus == 0:
        logging.info(f"Sortblock: for step {sb_holder.step_count}, all blocks are marked for recomputation")
        # all features are computed, input-outputs changes for all DiT blocks are stored for relative step 'k'
        sb_holder.activated_steps.append(sb_holder.step_count)
        for block in sb_holder.all_blocks:
            cache: BlockCache = block.__block_cache
            cache.mark_recompute()

    # all block operations are performed in forward pass of model
    to_return = executor(*args, **kwargs)

    # when 1: Select DiT blocks(4)
    if sb_holder.step_modulus == 1:
        logging.info(f"Sortblock: for step {sb_holder.step_count}, selecting blocks for recomputation and prediction")
        predict_ratio = 1.0 - sb_holder.predict_ratio
        for block_type, blocks in sb_holder.blocks_per_type.items():
            sorted_blocks = sorted(blocks, key=lambda x: x.__block_cache.cosine_similarity)
            threshold_index = int(len(sorted_blocks) * predict_ratio)
            # blocks with lower similarity are marked for recomputation
            for block in sorted_blocks[:threshold_index]:
                cache: BlockCache = block.__block_cache
                cache.mark_recompute()
            # blocks with higher similarity are marked for prediction
            for block in sorted_blocks[threshold_index:]:
                cache: BlockCache = block.__block_cache
                cache.mark_predict()
            logging.info(f"Sortblock: for {block_type}, selected {len(sorted_blocks[:threshold_index])} blocks for recomputation and {len(sorted_blocks[threshold_index:])} blocks for prediction")

    if sb_holder.initial_step:
        sb_holder.initial_step = False
    return to_return

def block_forward_factory(func, block):
    def block_forward_wrapper(*args, **kwargs):
        transformer_options: dict[str] = kwargs.get("transformer_options")
        sb_holder: SortblockHolder = transformer_options["sortblock"]
        cache: BlockCache = block.__block_cache
        # make sure stream count is properly set for this block
        if sb_holder.initial_step:
            sb_holder.add_to_blocks_per_type(block, transformer_options['block'][0])
            cache.block_index = transformer_options['block'][1]
            cache.stream_count = transformer_options['block'][2]
        # do sortblock stuff
        if cache.recompute and sb_holder.step_modulus != 1:
            # clone relevant inputs
            orig_inputs = cache.get_orig_inputs(args, kwargs, clone=True)
            # get block outputs
            # NOTE: output_raw is expected to have cache.stream_count elements if count is greaater than 1 (double block, etc.)
            if cache.stream_count == 1:
                zzz = 10
            output_raw: Union[torch.Tensor, tuple[torch.Tensor, ...]] = func(*args, **kwargs)
            # perform derivative approximation;
            cache.derivative_approximation(sb_holder, output_raw, orig_inputs)
            # if step_modulus is 0, input-output changes for DiT block are stored
            if sb_holder.step_modulus == 0:
                cache.cache_previous_residual(output_raw, orig_inputs)
        else:
            # if not to recompute, predict features for current timestep
            orig_inputs = cache.get_orig_inputs(args, kwargs, clone=False)
            # when 1: Linear Prediction(2)
            # if step_modulus is 1, store block residuals as 'current' after applying taylor_formula
            if sb_holder.step_modulus == 1:
                cache.cache_current_residual(sb_holder)
            # based on features computed in last timestep, all features for current timestep are predicted using Eq. 4,
            # input-output changes for all DiT blocks are stored for relative step 'k+1'
            output_raw = cache.apply_linear_prediction(sb_holder, orig_inputs)

        # when 1: Identify Changes(3)
        if sb_holder.step_modulus == 1:
            # based on features computed in last timestep, all features for current timestep are predicted using Eq. 4,
            # input-output changes for all DiT blocks are stored for relative step 'k+1'
            cache.calculate_cosine_similarity()

        # return output_raw
        return output_raw
    return block_forward_wrapper


def perform_sortblock(blocks: list):
    ...

def prepare_block(block, sb_holder: SortblockHolder, stream_count: int=1):
    sb_holder.add_to_all_blocks(block)
    block.__original_forward = block.forward
    block.forward = block_forward_factory(block.__original_forward, block)
    block.__block_cache = BlockCache(subsample_factor=sb_holder.subsample_factor, verbose=sb_holder.verbose)

def clean_block(block):
    block.forward = block.__original_forward
    del block.__original_forward
    del block.__block_cache

def subsample(x: torch.Tensor, factor: int, clone: bool=True) -> torch.Tensor:
    if factor > 1:
        to_return = x[..., ::factor, ::factor]
        if clone:
            return to_return.clone()
        return to_return
    if clone:
        return x.clone()
    return x

class BlockCache:
    def __init__(self, subsample_factor: int=8, verbose: bool=False):
        self.subsample_factor = subsample_factor
        self.verbose = verbose
        self.stream_count = 1
        self.recompute = False
        self.block_index = 0
        # cached values
        self.previous_residual_subsampled: torch.Tensor = None
        self.current_residual_subsampled: torch.Tensor = None
        self.cosine_similarity: float = None
        self.previous_taylor_factors: dict[int, torch.Tensor] = {}
        self.current_taylor_factors: dict[int, torch.Tensor] = {}

    def mark_recompute(self):
        self.recompute = True

    def mark_predict(self):
        self.recompute = False

    def cache_previous_residual(self, output_raw: Union[torch.Tensor, tuple[torch.Tensor, ...]], orig_inputs: Union[torch.Tensor, tuple[torch.Tensor, ...]]):
        if isinstance(output_raw, tuple):
            output_raw = output_raw[0]
        if isinstance(orig_inputs, tuple):
            orig_inputs = orig_inputs[0]
        del self.previous_residual_subsampled
        self.previous_residual_subsampled = subsample(output_raw - orig_inputs, self.subsample_factor, clone=True)

    def cache_current_residual(self, sb_holder: SortblockHolder):
        del self.current_residual_subsampled
        self.current_residual_subsampled = subsample(self.use_taylor_formula(sb_holder)[0], self.subsample_factor, clone=True)

    def get_orig_inputs(self, d_args: tuple, d_kwargs: dict, clone: bool=True) -> tuple[torch.Tensor, ...]:
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

    def apply_linear_prediction(self, sb_holder: SortblockHolder, orig_inputs: Union[torch.Tensor, tuple[torch.Tensor, ...]]) -> None:
        drop_tuple = False
        if not isinstance(orig_inputs, tuple):
            orig_inputs = (orig_inputs,)
            drop_tuple = True
        taylor_results = self.use_taylor_formula(sb_holder)
        for output, taylor_result in zip(orig_inputs, taylor_results):
            if output.shape != taylor_result.shape:
                zzz = 10
            output += taylor_result
        if drop_tuple:
            orig_inputs = orig_inputs[0]
        return orig_inputs

    def calculate_cosine_similarity(self) -> None:
        self.cosine_similarity = torch.nn.functional.cosine_similarity(self.previous_residual_subsampled, self.current_residual_subsampled, dim=-1).mean().item()

    def derivative_approximation(self, sb_holder: SortblockHolder, output_raw: Union[torch.Tensor, tuple[torch.Tensor, ...]], orig_inputs: Union[torch.Tensor, tuple[torch.Tensor, ...]]):
        activation_distance = sb_holder.activated_steps[-1] - sb_holder.activated_steps[-2]
        # make tuple if not already tuple, so that works with both single and double blocks
        if not isinstance(output_raw, tuple):
            output_raw = (output_raw,)
        if not isinstance(orig_inputs, tuple):
            orig_inputs = (orig_inputs,)

        for i, (output, x) in enumerate(zip(output_raw, orig_inputs)):
            feature = output.clone() - x
            has_previous_taylor_factor = self.previous_taylor_factors.get(i, None) is not None
            # NOTE: not sure why - 2, but that's what's in the original implementation. Maybe consider changing values?
            if has_previous_taylor_factor and sb_holder.step_count > (sb_holder.first_enhance - 2):
                self.current_taylor_factors[i] = (
                    feature - self.previous_taylor_factors[i]
                ) / activation_distance

            self.previous_taylor_factors[i] = feature

    def use_taylor_formula(self, sb_holder: SortblockHolder) -> tuple[torch.Tensor, ...]:
        step_distance = sb_holder.step_count - sb_holder.activated_steps[-1]

        output_predicted = []

        for key in self.previous_taylor_factors.keys():
            previous_tf = self.previous_taylor_factors[key]
            current_tf = self.current_taylor_factors[key]
            predicted = taylor_formula(previous_tf, 0, step_distance)
            predicted += taylor_formula(current_tf, 1, step_distance)
            output_predicted.append(predicted)

        return tuple(output_predicted)

    def reset(self):
        self.recompute = False
        self.current_residual_subsampled = None
        self.previous_residual_subsampled = None
        self.cosine_similarity = None
        self.previous_taylor_factors = {}
        self.current_taylor_factors = {}

def taylor_formula(taylor_factor: torch.Tensor, i: int, step_distance: int):
    return (
        (1 / math.factorial(i))
        * taylor_factor
        * (step_distance ** i)
    )

class SortblockHolder:
    def __init__(self, predict_ratio: float, policy_refresh_interval: int, start_percent: float, end_percent: float, subsample_factor: int=8, verbose: bool=False):
        self.predict_ratio = predict_ratio
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.subsample_factor = subsample_factor
        self.verbose = verbose

        # NOTE: number represents steps
        self.policy_refresh_interval = policy_refresh_interval
        self.active_policy_refresh_interval = 1
        self.first_enhance = 3  # NOTE: this value is 2 higher than the one actually used in code (subtracted by 2 in derivative_approximation)
        # timestep values
        self.start_t = 0.0
        self.end_t = 0.0
        self.curr_t = 0.0
        # control values
        self.initial_step = True
        self.step_count = 0
        self.activated_steps: list[int] = [0]
        self.step_modulus = 0
        # cache values
        self.all_blocks = []
        self.blocks_per_type = {}

    def add_to_all_blocks(self, block):
        self.all_blocks.append(block)

    def add_to_blocks_per_type(self, block, block_type: str):
        self.blocks_per_type.setdefault(block_type, []).append(block)

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def update_should_do_sortblock(self, timestep: float) -> bool:
        self.do_sortblock = (timestep[0] <= self.start_t).item() and (timestep[0] > self.end_t).item()
        self.curr_t = timestep
        if self.do_sortblock:
            self.active_policy_refresh_interval = self.policy_refresh_interval
        else:
            self.active_policy_refresh_interval = 1
        self.update_step_modulus()
        return self.do_sortblock

    def update_step_modulus(self):
        self.step_modulus = int(self.step_count % self.active_policy_refresh_interval)

    def should_do_sortblock(self) -> bool:
        return self.do_sortblock

    def reset(self):
        self.initial_step = True
        self.curr_t = 0.0
        logging.info(f"Sortblock: resetting {len(self.all_blocks)} blocks")
        for block in self.all_blocks:
            clean_block(block)
        self.all_blocks = []
        self.blocks_per_type = {}
        self.step_count = 0
        self.activated_steps = [0]
        self.step_modulus = 0
        return self

    def clone(self):
        return SortblockHolder(predict_ratio=self.predict_ratio, policy_refresh_interval=self.policy_refresh_interval,
            start_percent=self.start_percent, end_percent=self.end_percent, subsample_factor=self.subsample_factor,
            verbose=self.verbose)


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
                io.Float.Input("predict_ratio", min=0.0, default=0.8, max=3.0, step=0.01, tooltip="The ratio of blocks to predict."),
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
    def execute(cls, model: io.Model.Type, predict_ratio: float, policy_refresh_interval: int, start_percent: float, end_percent: float, verbose: bool) -> io.NodeOutput:
        # TODO: check for specific flavors of supported models
        model = model.clone()
        model.model_options["transformer_options"]["sortblock"] = SortblockHolder(predict_ratio, policy_refresh_interval, start_percent, end_percent, subsample_factor=8, verbose=verbose)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, "sortblock", prepare_noise_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "sortblock", outer_sample_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "sortblock", model_forward_wrapper)
        return io.NodeOutput(model)


class SortblockExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SortblockNode,
        ]

def comfy_entrypoint():
    return SortblockExtension()
