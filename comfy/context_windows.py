from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import torch
import numpy as np
import collections
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import comfy.model_management
import comfy.patcher_extension
if TYPE_CHECKING:
    from comfy.model_base import BaseModel
    from comfy.model_patcher import ModelPatcher
    from comfy.controlnet import ControlBase


class ContextWindowABC(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def get_tensor(self, full: torch.Tensor) -> torch.Tensor:
        """
        Get torch.Tensor applicable to current window.
        """
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def add_window(self, full: torch.Tensor, to_add: torch.Tensor) -> torch.Tensor:
        """
        Apply torch.Tensor of window to the full tensor, in place. Returns reference to updated full tensor, not a copy.
        """
        raise NotImplementedError("Not implemented.")

class ContextHandlerABC(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def should_use_context(self, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]) -> bool:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor, window: ContextWindowABC, device=None) -> list:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def execute(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        raise NotImplementedError("Not implemented.")



class IndexListContextWindow(ContextWindowABC):
    def __init__(self, index_list: list[int], dim: int=0, total_frames: int=0):
        self.index_list = index_list
        self.context_length = len(index_list)
        self.dim = dim
        self.total_frames = total_frames
        self.center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)

    def get_tensor(self, full: torch.Tensor, device=None, dim=None, retain_index_list=[]) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        if dim == 0 and full.shape[dim] == 1:
            return full
        idx = tuple([slice(None)] * dim + [self.index_list])
        window = full[idx]
        if retain_index_list:
            idx = tuple([slice(None)] * dim + [retain_index_list])
            window[idx] = full[idx]
        return window.to(device)

    def add_window(self, full: torch.Tensor, to_add: torch.Tensor, dim=None) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        idx = tuple([slice(None)] * dim + [self.index_list])
        full[idx] += to_add
        return full

    def get_region_index(self, num_regions: int) -> int:
        region_idx = int(self.center_ratio * num_regions)
        return min(max(region_idx, 0), num_regions - 1)


class IndexListCallbacks:
    EVALUATE_CONTEXT_WINDOWS = "evaluate_context_windows"
    COMBINE_CONTEXT_WINDOW_RESULTS = "combine_context_window_results"
    EXECUTE_START = "execute_start"
    EXECUTE_CLEANUP = "execute_cleanup"
    RESIZE_COND_ITEM = "resize_cond_item"

    def init_callbacks(self):
        return {}


@dataclass
class ContextSchedule:
    name: str
    func: Callable

@dataclass
class ContextFuseMethod:
    name: str
    func: Callable

ContextResults = collections.namedtuple("ContextResults", ['window_idx', 'sub_conds_out', 'sub_conds', 'window'])
class IndexListContextHandler(ContextHandlerABC):
    def __init__(self, context_schedule: ContextSchedule, fuse_method: ContextFuseMethod, context_length: int=1, context_overlap: int=0, context_stride: int=1,
                 closed_loop: bool=False, dim:int=0, freenoise: bool=False, cond_retain_index_list: list[int]=[], split_conds_to_windows: bool=False):
        self.context_schedule = context_schedule
        self.fuse_method = fuse_method
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self.dim = dim
        self._step = 0
        self.freenoise = freenoise
        self.cond_retain_index_list = [int(x.strip()) for x in cond_retain_index_list.split(",")] if cond_retain_index_list else []
        self.split_conds_to_windows = split_conds_to_windows

        self.callbacks = {}

    def should_use_context(self, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]) -> bool:
        # for now, assume first dim is batch - should have stored on BaseModel in actual implementation
        if x_in.size(self.dim) > self.context_length:
            logging.info(f"Using context windows {self.context_length} with overlap {self.context_overlap} for {x_in.size(self.dim)} frames.")
            if self.cond_retain_index_list:
                logging.info(f"Retaining original cond for indexes: {self.cond_retain_index_list}")
            return True
        return False

    def prepare_control_objects(self, control: ControlBase, device=None) -> ControlBase:
        if control.previous_controlnet is not None:
            self.prepare_control_objects(control.previous_controlnet, device)
        return control

    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor, window: IndexListContextWindow, device=None) -> list:
        if cond_in is None:
            return None
        # reuse or resize cond items to match context requirements
        resized_cond = []
        # if multiple conds, split based on primary region
        if self.split_conds_to_windows and len(cond_in) > 1:
            region = window.get_region_index(len(cond_in))
            logging.info(f"Splitting conds to windows; using region {region} for window {window.index_list[0]}-{window.index_list[-1]} with center ratio {window.center_ratio:.3f}")
            cond_in = [cond_in[region]]
        # cond object is a list containing a dict - outer list is irrelevant, so just loop through it
        for actual_cond in cond_in:
            resized_actual_cond = actual_cond.copy()
            # now we are in the inner dict - "pooled_output" is a tensor, "control" is a ControlBase object, "model_conds" is dictionary
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    if isinstance(cond_item, torch.Tensor):
                        # check that tensor is the expected length - x.size(0)
                        if self.dim < cond_item.ndim and cond_item.size(self.dim) == x_in.size(self.dim):
                            # if so, it's subsetting time - tell controls the expected indeces so they can handle them
                            actual_cond_item = window.get_tensor(cond_item)
                            resized_actual_cond[key] = actual_cond_item.to(device)
                        else:
                            resized_actual_cond[key] = cond_item.to(device)
                    # look for control
                    elif key == "control":
                        resized_actual_cond[key] = self.prepare_control_objects(cond_item, device)
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        # when in dictionary, look for tensors and CONDCrossAttn [comfy/conds.py] (has cond attr that is a tensor)
                        for cond_key, cond_value in new_cond_item.items():
                            # Allow callbacks to handle custom conditioning items
                            handled = False
                            for callback in comfy.patcher_extension.get_all_callbacks(
                                IndexListCallbacks.RESIZE_COND_ITEM, self.callbacks
                            ):
                                result = callback(cond_key, cond_value, window, x_in, device, new_cond_item)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                                    break
                            if handled:
                                continue
                            if isinstance(cond_value, torch.Tensor):
                                if (self.dim < cond_value.ndim and cond_value(self.dim) == x_in.size(self.dim)) or \
                                   (cond_value.ndim < self.dim and cond_value.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = window.get_tensor(cond_value, device)
                            # Handle audio_embed (temporal dim is 1)
                            elif cond_key == "audio_embed" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                audio_cond = cond_value.cond
                                if audio_cond.ndim > 1 and audio_cond.size(1) == x_in.size(self.dim):
                                    new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(audio_cond, device, dim=1))
                            # if has cond that is a Tensor, check if needs to be subset
                            elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                if  (self.dim < cond_value.cond.ndim and cond_value.cond.size(self.dim) == x_in.size(self.dim)) or \
                                    (cond_value.cond.ndim < self.dim and cond_value.cond.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(cond_value.cond, device, retain_index_list=self.cond_retain_index_list))
                            elif cond_key == "num_video_frames": # for SVD
                                new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond)
                                new_cond_item[cond_key].cond = window.context_length
                        resized_actual_cond[key] = new_cond_item
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    del cond_item  # just in case to prevent VRAM issues
            resized_cond.append(resized_actual_cond)
        return resized_cond

    def set_step(self, timestep: torch.Tensor, model_options: dict[str]):
        mask = torch.isclose(model_options["transformer_options"]["sample_sigmas"], timestep[0], rtol=0.0001)
        matches = torch.nonzero(mask)
        if torch.numel(matches) == 0:
            raise Exception("No sample_sigmas matched current timestep; something went wrong.")
        self._step = int(matches[0].item())

    def get_context_windows(self, model: BaseModel, x_in: torch.Tensor, model_options: dict[str]) -> list[IndexListContextWindow]:
        full_length = x_in.size(self.dim) # TODO: choose dim based on model
        context_windows = self.context_schedule.func(full_length, self, model_options)
        context_windows = [IndexListContextWindow(window, dim=self.dim, total_frames=full_length) for window in context_windows]
        return context_windows

    def execute(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        self.set_step(timestep, model_options)
        context_windows = self.get_context_windows(model, x_in, model_options)
        enumerated_context_windows = list(enumerate(context_windows))

        conds_final = [torch.zeros_like(x_in) for _ in conds]
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            counts_final = [torch.ones(get_shape_for_dim(x_in, self.dim), device=x_in.device) for _ in conds]
        else:
            counts_final = [torch.zeros(get_shape_for_dim(x_in, self.dim), device=x_in.device) for _ in conds]
        biases_final = [([0.0] * x_in.shape[self.dim]) for _ in conds]

        for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_START, self.callbacks):
            callback(self, model, x_in, conds, timestep, model_options)

        for enum_window in enumerated_context_windows:
            results = self.evaluate_context_windows(calc_cond_batch, model, x_in, conds, timestep, [enum_window], model_options)
            for result in results:
                self.combine_context_window_results(x_in, result.sub_conds_out, result.sub_conds, result.window, result.window_idx, len(enumerated_context_windows), timestep,
                                            conds_final, counts_final, biases_final)
        try:
            # finalize conds
            if self.fuse_method.name == ContextFuseMethods.RELATIVE:
                # relative is already normalized, so return as is
                del counts_final
                return conds_final
            else:
                # normalize conds via division by context usage counts
                for i in range(len(conds_final)):
                    conds_final[i] /= counts_final[i]
                del counts_final
                return conds_final
        finally:
            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options)

    def evaluate_context_windows(self, calc_cond_batch: Callable, model: BaseModel, x_in: torch.Tensor, conds, timestep: torch.Tensor, enumerated_context_windows: list[tuple[int, IndexListContextWindow]],
                                model_options, device=None, first_device=None):
        results: list[ContextResults] = []
        for window_idx, window in enumerated_context_windows:
            # allow processing to end between context window executions for faster Cancel
            comfy.model_management.throw_exception_if_processing_interrupted()

            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options, window_idx, window, model_options, device, first_device)

            # update exposed params
            model_options["transformer_options"]["context_window"] = window
            # get subsections of x, timestep, conds
            sub_x = window.get_tensor(x_in, device)
            sub_timestep = window.get_tensor(timestep, device, dim=0)
            sub_conds = [self.get_resized_cond(cond, x_in, window, device) for cond in conds]

            sub_conds_out = calc_cond_batch(model, sub_conds, sub_x, sub_timestep, model_options)
            if device is not None:
                for i in range(len(sub_conds_out)):
                    sub_conds_out[i] = sub_conds_out[i].to(x_in.device)
            results.append(ContextResults(window_idx, sub_conds_out, sub_conds, window))
        return results


    def combine_context_window_results(self, x_in: torch.Tensor, sub_conds_out, sub_conds, window: IndexListContextWindow, window_idx: int, total_windows: int, timestep: torch.Tensor,
                                    conds_final: list[torch.Tensor], counts_final: list[torch.Tensor], biases_final: list[torch.Tensor]):
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            for pos, idx in enumerate(window.index_list):
                # bias is the influence of a specific index in relation to the whole context window
                bias = 1 - abs(idx - (window.index_list[0] + window.index_list[-1]) / 2) / ((window.index_list[-1] - window.index_list[0] + 1e-2) / 2)
                bias = max(1e-2, bias)
                # take weighted average relative to total bias of current idx
                for i in range(len(sub_conds_out)):
                    bias_total = biases_final[i][idx]
                    prev_weight = (bias_total / (bias_total + bias))
                    new_weight = (bias / (bias_total + bias))
                    # account for dims of tensors
                    idx_window = tuple([slice(None)] * self.dim + [idx])
                    pos_window = tuple([slice(None)] * self.dim + [pos])
                    # apply new values
                    conds_final[i][idx_window] = conds_final[i][idx_window] * prev_weight + sub_conds_out[i][pos_window] * new_weight
                    biases_final[i][idx] = bias_total + bias
        else:
            # add conds and counts based on weights of fuse method
            weights = get_context_weights(window.context_length, x_in.shape[self.dim], window.index_list, self, sigma=timestep)
            weights_tensor = match_weights_to_dim(weights, x_in, self.dim, device=x_in.device)
            for i in range(len(sub_conds_out)):
                window.add_window(conds_final[i], sub_conds_out[i] * weights_tensor)
                window.add_window(counts_final[i], weights_tensor)

        for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS, self.callbacks):
            callback(self, x_in, sub_conds_out, sub_conds, window, window_idx, total_windows, timestep, conds_final, counts_final, biases_final)


def _prepare_sampling_wrapper(executor, model, noise_shape: torch.Tensor, *args, **kwargs):
    # limit noise_shape length to context_length for more accurate vram use estimation
    model_options = kwargs.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in prepare_sampling_wrapper; this should never happen, something went wrong.")
    handler: IndexListContextHandler = model_options.get("context_handler", None)
    if handler is not None:
        noise_shape = list(noise_shape)
        noise_shape[handler.dim] = min(noise_shape[handler.dim], handler.context_length)
    return executor(model, noise_shape, *args, **kwargs)


def create_prepare_sampling_wrapper(model: ModelPatcher):
    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
        "ContextWindows_prepare_sampling",
        _prepare_sampling_wrapper
    )


def _sampler_sample_wrapper(executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
    model_options = extra_args.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in sampler_sample_wrapper; this should never happen, something went wrong.")
    handler: IndexListContextHandler = model_options.get("context_handler", None)
    if handler is None:
        raise Exception("context_handler not found in sampler_sample_wrapper; this should never happen, something went wrong.")
    if not handler.freenoise:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
    noise = apply_freenoise(noise, handler.dim, handler.context_length, handler.context_overlap, extra_args["seed"])

    return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)


def create_sampler_sample_wrapper(model: ModelPatcher):
    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        "ContextWindows_sampler_sample",
        _sampler_sample_wrapper
    )


def match_weights_to_dim(weights: list[float], x_in: torch.Tensor, dim: int, device=None) -> torch.Tensor:
    total_dims = len(x_in.shape)
    weights_tensor = torch.Tensor(weights).to(device=device)
    for _ in range(dim):
        weights_tensor = weights_tensor.unsqueeze(0)
    for _ in range(total_dims - dim - 1):
        weights_tensor = weights_tensor.unsqueeze(-1)
    return weights_tensor

def get_shape_for_dim(x_in: torch.Tensor, dim: int) -> list[int]:
    total_dims = len(x_in.shape)
    shape = []
    for _ in range(dim):
        shape.append(1)
    shape.append(x_in.shape[dim])
    for _ in range(total_dims - dim - 1):
        shape.append(1)
    return shape

class ContextSchedules:
    UNIFORM_LOOPED = "looped_uniform"
    UNIFORM_STANDARD = "standard_uniform"
    STATIC_STANDARD = "standard_static"
    BATCHED = "batched"


# from https://github.com/neggles/animatediff-cli/blob/main/src/animatediff/pipelines/context.py
def create_windows_uniform_looped(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames < handler.context_length:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(handler.context_stride, int(np.ceil(np.log2(num_frames / handler.context_length))) + 1)
    # obtain uniform windows as normal, looping and all
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(handler._step)))
        for j in range(
            int(ordered_halving(handler._step) * context_step) + pad,
            num_frames + pad + (0 if handler.closed_loop else -handler.context_overlap),
            (handler.context_length * context_step - handler.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + handler.context_length * context_step, context_step)])

    return windows

def create_windows_uniform_standard(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    # unlike looped, uniform_straight does NOT allow windows that loop back to the beginning;
    # instead, they get shifted to the corresponding end of the frames.
    # in the case that a window (shifted or not) is identical to the previous one, it gets skipped.
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(handler.context_stride, int(np.ceil(np.log2(num_frames / handler.context_length))) + 1)
    # first, obtain uniform windows as normal, looping and all
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(handler._step)))
        for j in range(
            int(ordered_halving(handler._step) * context_step) + pad,
            num_frames + pad + (-handler.context_overlap),
            (handler.context_length * context_step - handler.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + handler.context_length * context_step, context_step)])

    # now that windows are created, shift any windows that loop, and delete duplicate windows
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        # if window is rolls over itself, need to shift it
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]  # roll_val might not be 0 for windows of higher strides
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # check if next window (cyclical) is missing roll_val
            if roll_val not in windows[(win_i+1) % len(windows)]:
                # need to insert new window here - just insert window starting at roll_val
                windows.insert(win_i+1, list(range(roll_val, roll_val + handler.context_length)))
        # delete window if it's not unique
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    # reverse delete_idxs so that they will be deleted in an order that doesn't break idx correlation
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)

    return windows


def create_windows_static_standard(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = handler.context_length - handler.context_overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + handler.context_length
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + handler.context_length)))
            break
        windows.append(list(range(start_idx, start_idx + handler.context_length)))
    return windows


def create_windows_batched(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows;
    # no overlap, just cut up based on context_length;
    # last window size will be different if num_frames % opts.context_length != 0
    for start_idx in range(0, num_frames, handler.context_length):
        windows.append(list(range(start_idx, min(start_idx + handler.context_length, num_frames))))
    return windows


def create_windows_default(num_frames: int, handler: IndexListContextHandler):
    return [list(range(num_frames))]


CONTEXT_MAPPING = {
    ContextSchedules.UNIFORM_LOOPED: create_windows_uniform_looped,
    ContextSchedules.UNIFORM_STANDARD: create_windows_uniform_standard,
    ContextSchedules.STATIC_STANDARD: create_windows_static_standard,
    ContextSchedules.BATCHED: create_windows_batched,
}


def get_matching_context_schedule(context_schedule: str) -> ContextSchedule:
    func = CONTEXT_MAPPING.get(context_schedule, None)
    if func is None:
        raise ValueError(f"Unknown context_schedule '{context_schedule}'.")
    return ContextSchedule(context_schedule, func)


def get_context_weights(length: int, full_length: int, idxs: list[int], handler: IndexListContextHandler, sigma: torch.Tensor=None):
    return handler.fuse_method.func(length, sigma=sigma, handler=handler, full_length=full_length, idxs=idxs)


def create_weights_flat(length: int, **kwargs) -> list[float]:
    # weight is the same for all
    return [1.0] * length

def create_weights_pyramid(length: int, **kwargs) -> list[float]:
    # weight is based on the distance away from the edge of the context window;
    # based on weighted average concept in FreeNoise paper
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence

def create_weights_overlap_linear(length: int, full_length: int, idxs: list[int], handler: IndexListContextHandler, **kwargs):
    # based on code in Kijai's WanVideoWrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/dbb2523b37e4ccdf45127e5ae33e31362f755c8e/nodes.py#L1302
    # only expected overlap is given different weights
    weights_torch = torch.ones((length))
    # blend left-side on all except first window
    if min(idxs) > 0:
        ramp_up = torch.linspace(1e-37, 1, handler.context_overlap)
        weights_torch[:handler.context_overlap] = ramp_up
    # blend right-side on all except last window
    if max(idxs) < full_length-1:
        ramp_down = torch.linspace(1, 1e-37, handler.context_overlap)
        weights_torch[-handler.context_overlap:] = ramp_down
    return weights_torch

class ContextFuseMethods:
    FLAT = "flat"
    PYRAMID = "pyramid"
    RELATIVE = "relative"
    OVERLAP_LINEAR = "overlap-linear"

    LIST = [PYRAMID, FLAT, OVERLAP_LINEAR]
    LIST_STATIC = [PYRAMID, RELATIVE, FLAT, OVERLAP_LINEAR]


FUSE_MAPPING = {
    ContextFuseMethods.FLAT: create_weights_flat,
    ContextFuseMethods.PYRAMID: create_weights_pyramid,
    ContextFuseMethods.RELATIVE: create_weights_pyramid,
    ContextFuseMethods.OVERLAP_LINEAR: create_weights_overlap_linear,
}

def get_matching_fuse_method(fuse_method: str) -> ContextFuseMethod:
    func = FUSE_MAPPING.get(fuse_method, None)
    if func is None:
        raise ValueError(f"Unknown fuse_method '{fuse_method}'.")
    return ContextFuseMethod(fuse_method, func)

# Returns fraction that has denominator that is a power of 2
def ordered_halving(val):
    # get binary value, padded with 0s for 64 bits
    bin_str = f"{val:064b}"
    # flip binary value, padding included
    bin_flip = bin_str[::-1]
    # convert binary to int
    as_int = int(bin_flip, 2)
    # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
    # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
    return as_int / (1 << 64)


def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes


def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1


def shift_window_to_start(window: list[int], num_frames: int):
    start_val = window[0]
    for i in range(len(window)):
        # 1) subtract each element by start_val to move vals relative to the start of all frames
        # 2) add num_frames and take modulus to get adjusted vals
        window[i] = ((window[i] - start_val) + num_frames) % num_frames


def shift_window_to_end(window: list[int], num_frames: int):
    # 1) shift window to start
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        # 2) add end_delta to each val to slide windows to end
        window[i] = window[i] + end_delta


# https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/90fb1331201a4b29488089e4fbffc0d82cc6d0a9/animatediff/sample_settings.py#L465
def apply_freenoise(noise: torch.Tensor, dim: int, context_length: int, context_overlap: int, seed: int):
    logging.info("Context windows: Applying FreeNoise")
    generator = torch.Generator(device='cpu').manual_seed(seed)
    latent_video_length = noise.shape[dim]
    delta = context_length - context_overlap

    for start_idx in range(0, latent_video_length - context_length, delta):
        place_idx = start_idx + context_length

        actual_delta = min(delta, latent_video_length - place_idx)
        if actual_delta <= 0:
            break

        list_idx = torch.randperm(actual_delta, generator=generator, device='cpu') + start_idx

        source_slice = [slice(None)] * noise.ndim
        source_slice[dim] = list_idx
        target_slice = [slice(None)] * noise.ndim
        target_slice[dim] = slice(place_idx, place_idx + actual_delta)

        noise[tuple(target_slice)] = noise[tuple(source_slice)]

    return noise
