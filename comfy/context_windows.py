from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
import torch
import numpy as np
import collections
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import comfy.conds
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
    def __init__(self, index_list: list[int], dim: int=0, total_frames: int=0, modality_windows: dict=None):
        self.index_list = index_list
        self.context_length = len(index_list)
        self.dim = dim
        self.total_frames = total_frames
        self.center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
        self.modality_windows = modality_windows  # dict of {mod_idx: IndexListContextWindow}

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


def slice_cond(cond_value, window: IndexListContextWindow, x_in: torch.Tensor, device, temporal_dim: int, temporal_scale: int=1, temporal_offset: int=0, retain_index_list: list[int]=[]):
    if not (hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor)):
        return None
    cond_tensor = cond_value.cond
    if temporal_dim >= cond_tensor.ndim:
        return None

    cond_size = cond_tensor.size(temporal_dim)

    if temporal_scale == 1:
        expected_size = x_in.size(window.dim) - temporal_offset
        if cond_size != expected_size:
            return None

    if temporal_offset == 0 and temporal_scale == 1:
        sliced = window.get_tensor(cond_tensor, device, dim=temporal_dim, retain_index_list=retain_index_list)
        return cond_value._copy_with(sliced)

    # skip leading latent positions that have no corresponding conditioning (e.g. reference frames)
    if temporal_offset > 0:
        indices = [i - temporal_offset for i in window.index_list[temporal_offset:]]
        indices = [i for i in indices if 0 <= i]
    else:
        indices = list(window.index_list)

    if not indices:
        return None

    if temporal_scale > 1:
        scaled = []
        for i in indices:
            for k in range(temporal_scale):
                si = i * temporal_scale + k
                if si < cond_size:
                    scaled.append(si)
        indices = scaled
        if not indices:
            return None

    idx = tuple([slice(None)] * temporal_dim + [indices])
    sliced = cond_tensor[idx].to(device)
    return cond_value._copy_with(sliced)


def compute_guide_overlap(guide_entries: list[dict], window_index_list: list[int]):
    """Compute which concatenated guide frames overlap with a context window.

    Args:
        guide_entries: list of guide_attention_entry dicts
        window_index_list: the window's frame indices into the video portion

    Returns:
        suffix_indices: indices into the guide_frames tensor for frame selection
        overlap_info: list of (entry_idx, overlap_count) for guide_attention_entries adjustment
        kf_local_positions: window-local frame positions for keyframe_idxs regeneration
        total_overlap: total number of overlapping guide frames
    """
    window_set = set(window_index_list)
    window_list = list(window_index_list)
    suffix_indices = []
    overlap_info = []
    kf_local_positions = []
    suffix_base = 0

    for entry_idx, entry in enumerate(guide_entries):
        latent_start = entry.get("latent_start", None)
        if latent_start is None:
            raise ValueError("guide_attention_entry missing required 'latent_start'.")
        guide_len = entry["latent_shape"][0]
        entry_overlap = 0

        for local_offset in range(guide_len):
            video_pos = latent_start + local_offset
            if video_pos in window_set:
                suffix_indices.append(suffix_base + local_offset)
                kf_local_positions.append(window_list.index(video_pos))
                entry_overlap += 1

        if entry_overlap > 0:
            overlap_info.append((entry_idx, entry_overlap))
        suffix_base += guide_len

    return suffix_indices, overlap_info, kf_local_positions, len(suffix_indices)


def inject_guide_frames_into_window(video_slice: torch.Tensor, window: ContextWindowABC,
                                    window_data: 'WindowingContext', dim: int) -> tuple[torch.Tensor, int]:
    """Inject overlapping guide frames into a context window slice.

    Determines which guide frames overlap with this window's indices, concatenates
    them onto the video slice, and sets window attributes for downstream conditioning resize.

    Returns (augmented_slice, num_guide_frames_added).
    """
    guide_entries = window_data.aux_data["guide_entries"]
    guide_frames = window_data.guide_frames
    overlap = compute_guide_overlap(guide_entries, window.index_list)
    suffix_idx, overlap_info, kf_local_pos, guide_frame_count = overlap
    window.guide_frames_indices = suffix_idx
    window.guide_overlap_info = overlap_info
    window.guide_kf_local_positions = kf_local_pos
    
    # Derive per-overlap-entry latent_downscale_factor from guide entry latent_shape vs guide frame spatial dims.
    # guide_frames has full (post-dilation) spatial dims; entry["latent_shape"] has pre-dilation dims.
    guide_downscale_factors = []
    if guide_frame_count > 0:
        full_H = guide_frames.shape[3]
        for entry_idx, _ in overlap_info:
            entry_H = guide_entries[entry_idx]["latent_shape"][1]
            guide_downscale_factors.append(full_H // entry_H)
    window.guide_downscale_factors = guide_downscale_factors

    if guide_frame_count > 0:
        idx = tuple([slice(None)] * dim + [suffix_idx])
        sliced_guide = guide_frames[idx]
        return torch.cat([video_slice, sliced_guide], dim=dim), guide_frame_count
    return video_slice, 0


@dataclass
class WindowingContext:
    tensor: torch.Tensor
    guide_frames: torch.Tensor | None
    aux_data: Any
    latent_shapes: list | None

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

    @staticmethod
    def _get_latent_shapes(conds):
        for cond_list in conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                if 'latent_shapes' in model_conds:
                    return model_conds['latent_shapes'].cond
        return None

    @staticmethod
    def _unpack(combined_latent, latent_shapes):
        if latent_shapes is not None and len(latent_shapes) > 1:
            return comfy.utils.unpack_latents(combined_latent, latent_shapes)
        return [combined_latent]

    @staticmethod
    def _pack(latents):
        if len(latents) > 1:
            return comfy.utils.pack_latents(latents)
        return latents[0], [latents[0].shape]

    @staticmethod
    def _patch_latent_shapes(sub_conds, new_shapes):
        for cond_list in sub_conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                if 'latent_shapes' in model_conds:
                    model_conds['latent_shapes'] = comfy.conds.CONDConstant(new_shapes)

    def _build_window_data(self, x_in: torch.Tensor, conds: list[list[dict]]) -> WindowingContext:
        latent_shapes = self._get_latent_shapes(conds)
        is_multimodal = latent_shapes is not None and len(latent_shapes) > 1
        if is_multimodal:
            video_latent = comfy.utils.unpack_latents(x_in, latent_shapes)[0]
        else:
            video_latent = x_in

        guide_entries = None
        for cond_list in conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                entries = model_conds.get('guide_attention_entries')
                if entries is not None and hasattr(entries, 'cond') and entries.cond:
                    guide_entries = entries.cond
                    break
            if guide_entries is not None:
                break

        guide_frame_count = sum(e["latent_shape"][0] for e in guide_entries) if guide_entries is not None else 0
        primary_frame_count = video_latent.size(self.dim) - guide_frame_count
        primary_frames = video_latent.narrow(self.dim, 0, primary_frame_count)
        guide_frames = video_latent.narrow(self.dim, primary_frame_count, guide_frame_count) if guide_frame_count > 0 else None

        if guide_frame_count > 0:
            aux_data = {"guide_entries": guide_entries}
        else:
            aux_data = None

        return WindowingContext(
            tensor=primary_frames, guide_frames=guide_frames, aux_data=aux_data,
            latent_shapes=latent_shapes)

    def should_use_context(self, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]) -> bool:
        self._window_data = self._build_window_data(x_in, conds)
        video_frames = self._window_data.tensor.size(self.dim)
        guide_frames = self._window_data.guide_frames.size(self.dim) if self._window_data.guide_frames is not None else 0
        if video_frames > self.context_length:
            if guide_frames > 0:
                logging.info(f"Using context windows {self.context_length} with overlap {self.context_overlap} for {video_frames} video frames ({guide_frames} guide frames).")
            else:
                logging.info(f"Using context windows {self.context_length} with overlap {self.context_overlap} for {video_frames} frames.")
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
                            if not handled and self._model is not None:
                                result = self._model.resize_cond_for_context_window(
                                    cond_key, cond_value, window, x_in, device,
                                    retain_index_list=self.cond_retain_index_list)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                            if handled:
                                continue
                            if isinstance(cond_value, torch.Tensor):
                                if (self.dim < cond_value.ndim and cond_value.size(self.dim) == x_in.size(self.dim)) or \
                                   (cond_value.ndim < self.dim and cond_value.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = window.get_tensor(cond_value, device)
                            # Handle audio_embed (temporal dim is 1)
                            elif cond_key == "audio_embed" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                audio_cond = cond_value.cond
                                if audio_cond.ndim > 1 and audio_cond.size(1) == x_in.size(self.dim):
                                    new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(audio_cond, device, dim=1))
                            # Handle vace_context (temporal dim is 3)
                            elif cond_key == "vace_context" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                vace_cond = cond_value.cond
                                if vace_cond.ndim >= 4 and vace_cond.size(3) == x_in.size(self.dim):
                                    sliced_vace = window.get_tensor(vace_cond, device, dim=3, retain_index_list=self.cond_retain_index_list)
                                    new_cond_item[cond_key] = cond_value._copy_with(sliced_vace)
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
            return  # substep from multi-step sampler: keep self._step from the last full step
        self._step = int(matches[0].item())

    def get_context_windows(self, model: BaseModel, x_in: torch.Tensor, model_options: dict[str]) -> list[IndexListContextWindow]:
        full_length = x_in.size(self.dim) # TODO: choose dim based on model
        context_windows = self.context_schedule.func(full_length, self, model_options)
        context_windows = [IndexListContextWindow(window, dim=self.dim, total_frames=full_length) for window in context_windows]
        return context_windows

    def execute(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        self._model = model
        self.set_step(timestep, model_options)

        window_data = self._window_data
        is_multimodal = window_data.latent_shapes is not None and len(window_data.latent_shapes) > 1
        has_guide_frames = window_data.guide_frames is not None and window_data.guide_frames.size(self.dim) > 0

        # if multimodal or has concatenated guide frames on noise latent, use the extended execute path
        if is_multimodal or has_guide_frames:
            return self._execute_extended(calc_cond_batch, model, conds, x_in, timestep, model_options, window_data)

        # basic legacy execution path for single-modal video latent with no guide frames concatenated
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
            if self.fuse_method.name == ContextFuseMethods.RELATIVE:
                del counts_final
                return conds_final
            else:
                for i in range(len(conds_final)):
                    conds_final[i] /= counts_final[i]
                del counts_final
                return conds_final
        finally:
            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options)

    def _execute_extended(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor,
                            timestep: torch.Tensor, model_options: dict[str],
                            window_data: WindowingContext):
        """Extended execute path for multimodal models and models with guide frames appended to the noise latent."""

        latents = self._unpack(x_in, window_data.latent_shapes)
        is_multimodal = window_data.latent_shapes is not None and len(window_data.latent_shapes) > 1

        primary_frames = window_data.tensor
        num_guide_frames = window_data.guide_frames.size(self.dim) if window_data.guide_frames is not None else 0

        context_windows = self.get_context_windows(model, primary_frames, model_options)
        enumerated_context_windows = list(enumerate(context_windows))
        total_windows = len(enumerated_context_windows)

        # Accumulators sized to video portion for primary, full for other modalities
        accum_shape_refs = list(latents)
        if window_data.guide_frames is not None:
            accum_shape_refs[0] = primary_frames

        accum = [[torch.zeros_like(m) for _ in conds] for m in accum_shape_refs]
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            counts = [[torch.ones(get_shape_for_dim(m, self.dim), device=m.device) for _ in conds] for m in accum_shape_refs]
        else:
            counts = [[torch.zeros(get_shape_for_dim(m, self.dim), device=m.device) for _ in conds] for m in accum_shape_refs]
        biases = [[([0.0] * m.shape[self.dim]) for _ in conds] for m in accum_shape_refs]

        for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_START, self.callbacks):
            callback(self, model, x_in, conds, timestep, model_options)

        for window_idx, window in enumerated_context_windows:
            comfy.model_management.throw_exception_if_processing_interrupted()
            logging.info(f"Context window {window_idx + 1}/{total_windows}: frames {window.index_list[0]}-{window.index_list[-1]} of {primary_frames.shape[self.dim]}"
                         + (f" (+{num_guide_frames} guide frames)" if num_guide_frames > 0 else "")
                         + (f" [{len(latents)} modalities]" if is_multimodal else ""))

            # Per-modality window indices
            if is_multimodal:
                map_shapes = window_data.latent_shapes
                if primary_frames.size(self.dim) != latents[0].size(self.dim):
                    map_shapes = list(window_data.latent_shapes)
                    video_shape = list(window_data.latent_shapes[0])
                    video_shape[self.dim] = primary_frames.size(self.dim)
                    map_shapes[0] = torch.Size(video_shape)
                try:
                    per_modality_indices = model.map_context_window_to_modalities(
                        window.index_list, map_shapes, self.dim)
                except AttributeError:
                    raise NotImplementedError(
                        f"{type(model).__name__} must implement map_context_window_to_modalities for multimodal context windows.")
                modality_windows = {}
                for mod_idx in range(1, len(latents)):
                    modality_windows[mod_idx] = IndexListContextWindow(
                        per_modality_indices[mod_idx], dim=self.dim,
                        total_frames=latents[mod_idx].shape[self.dim])
                window = IndexListContextWindow(
                    window.index_list, dim=self.dim, total_frames=primary_frames.shape[self.dim],
                    modality_windows=modality_windows)

            # Build per-modality windows list
            per_modality_windows_list = [window]
            if is_multimodal:
                for mod_idx in range(1, len(latents)):
                    per_modality_windows_list.append(modality_windows[mod_idx])

            # Slice video, then inject overlapping guide frames if present
            sliced_video = per_modality_windows_list[0].get_tensor(primary_frames, retain_index_list=self.cond_retain_index_list)
            if window_data.aux_data is not None:
                sliced_primary, num_guide_frames = inject_guide_frames_into_window(sliced_video, window, window_data, self.dim)
            else:
                sliced_primary, num_guide_frames = sliced_video, 0
            sliced = [sliced_primary] + [per_modality_windows_list[mi].get_tensor(latents[mi]) for mi in range(1, len(latents))]

            sub_x, sub_shapes = self._pack(sliced)

            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options, window_idx, window, model_options, None, None)

            model_options["transformer_options"]["context_window"] = window
            sub_timestep = window.get_tensor(timestep, dim=0)
            sub_conds = [self.get_resized_cond(cond, primary_frames, window) for cond in conds]
            if is_multimodal:
                self._patch_latent_shapes(sub_conds, sub_shapes)

            sub_conds_out = calc_cond_batch(model, sub_conds, sub_x, sub_timestep, model_options)

            # Unpack output per modality
            out_per_modality = [self._unpack(sub_conds_out[i], sub_shapes) for i in range(len(sub_conds_out))]

            # Strip guide frames from primary output before accumulation
            if num_guide_frames > 0:
                window_len = len(window.index_list)
                for ci in range(len(sub_conds_out)):
                    out_per_modality[ci][0] = out_per_modality[ci][0].narrow(self.dim, 0, window_len)

            # Accumulate per modality
            for mod_idx in range(len(accum_shape_refs)):
                mw = per_modality_windows_list[mod_idx]
                sub_conds_out_per_modality = [out_per_modality[ci][mod_idx] for ci in range(len(sub_conds_out))]
                self.combine_context_window_results(
                    accum_shape_refs[mod_idx], sub_conds_out_per_modality, sub_conds, mw,
                    window_idx, total_windows, timestep,
                    accum[mod_idx], counts[mod_idx], biases[mod_idx])

        try:
            result = []
            for ci in range(len(conds)):
                finalized = []
                for mod_idx in range(len(accum_shape_refs)):
                    if self.fuse_method.name != ContextFuseMethods.RELATIVE:
                        accum[mod_idx][ci] /= counts[mod_idx][ci]
                    f = accum[mod_idx][ci]
                    if mod_idx == 0 and window_data.guide_frames is not None:
                        f = torch.cat([f, window_data.guide_frames], dim=self.dim)
                    finalized.append(f)
                packed, _ = self._pack(finalized)
                result.append(packed)
            return result
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
        # Guard: only clamp when dim is within bounds and the value is meaningful
        # (packed multimodal tensors have noise_shape=[B,1,flat] where flat is not frame count)
        if handler.dim < len(noise_shape) and noise_shape[handler.dim] > handler.context_length:
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

    # For packed multimodal tensors (e.g. LTXAV), noise is [B, 1, flat] and FreeNoise
    # must only shuffle the video portion. Unpack, apply to video, repack.
    latent_shapes = IndexListContextHandler._get_latent_shapes(
        [guider.conds.get('positive', guider.conds.get('negative', []))])

    if latent_shapes is not None and len(latent_shapes) > 1:
        modalities = comfy.utils.unpack_latents(noise, latent_shapes)
        video_total = latent_shapes[0][handler.dim]
        modalities[0] = apply_freenoise(modalities[0], handler.dim, handler.context_length, handler.context_overlap, extra_args["seed"])
        for i in range(1, len(modalities)):
            mod_total = latent_shapes[i][handler.dim]
            ratio = mod_total / video_total if video_total > 0 else 1
            mod_ctx_len = max(round(handler.context_length * ratio), 1)
            mod_ctx_overlap = max(round(handler.context_overlap * ratio), 0)
            modalities[i] = apply_freenoise(modalities[i], handler.dim, mod_ctx_len, mod_ctx_overlap, extra_args["seed"])
        noise, _ = comfy.utils.pack_latents(modalities)
    else:
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
