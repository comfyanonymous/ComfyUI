"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import collections
import copy
import inspect
import logging
import uuid
from typing import Optional

import torch
import torch.nn
from humanize import naturalsize

from . import model_management, lora
from . import utils
from .comfy_types import UnetWrapperFunction
from .float import stochastic_rounding
from .lora_types import PatchDict, PatchDictKey, PatchTuple, PatchWeightTuple, ModelPatchesDictValue
from .model_base import BaseModel
from .model_management_types import ModelManageable, MemoryMeasurements, ModelOptions

logger = logging.getLogger(__name__)


def string_to_seed(data):
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options


def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def set_model_options_pre_cfg_function(model_options, pre_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_pre_cfg_function"] = model_options.get("sampler_pre_cfg_function", []) + [pre_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights
    m.weight_function = None
    m.bias_function = None


class LowVramPatch:
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:  # intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return stochastic_rounding(lora.calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), weight.dtype, seed=string_to_seed(self.key))
        return lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)


def get_key_weight(model, key):
    set_func = None
    convert_func = None
    op_keys = key.rsplit('.', 1)
    if len(op_keys) < 2:
        weight = utils.get_attr(model, key)
    else:
        op = utils.get_attr(model, op_keys[0])
        try:
            set_func = getattr(op, "set_{}".format(op_keys[1]))
        except AttributeError:
            pass

        try:
            convert_func = getattr(op, "convert_{}".format(op_keys[1]))
        except AttributeError:
            pass

        weight = getattr(op, op_keys[1])
        if convert_func is not None:
            weight = utils.get_attr(model, key)

    return weight, set_func, convert_func


class ModelPatcher(ModelManageable):
    def __init__(self, model: BaseModel | torch.nn.Module, load_device: torch.device, offload_device: torch.device, size=0, weight_inplace_update=False, ckpt_name: Optional[str] = None):
        self.size = size
        self.model: BaseModel | torch.nn.Module = model
        self.patches: dict[PatchDictKey, ModelPatchesDictValue] = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self._model_options: ModelOptions = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        self._parent: ModelManageable | None = None
        self.patches_uuid: uuid.UUID = uuid.uuid4()
        self.ckpt_name = ckpt_name
        self._memory_measurements = MemoryMeasurements(self.model)

    @property
    def model_options(self) -> ModelOptions:
        return self._model_options

    @model_options.setter
    def model_options(self, value: ModelOptions):
        self._model_options = value

    @property
    def model_device(self) -> torch.device:
        return self._memory_measurements.device

    @model_device.setter
    def model_device(self, value: torch.device):
        self._memory_measurements.device = value

    @property
    def current_weight_patches_uuid(self) -> Optional[uuid.UUID]:
        return self._memory_measurements.current_weight_patches_uuid

    @current_weight_patches_uuid.setter
    def current_weight_patches_uuid(self, value):
        self._memory_measurements.current_weight_patches_uuid = value

    @property
    def parent(self) -> Optional["ModelPatcher"]:
        return self._parent

    def lowvram_patch_counter(self):
        return self._memory_measurements.lowvram_patch_counter

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = model_management.module_size(self.model)
        return self.size

    def loaded_size(self):
        return self._memory_measurements.model_loaded_weight_memory

    def clone(self):
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n._memory_measurements = self._memory_measurements
        n.ckpt_name = self.ckpt_name
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n._model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n._parent = self
        return n

    def is_clone(self, other):
        return hasattr(other, 'model') and self.model is other.model

    def clone_has_same_weights(self, clone: "ModelPatcher"):
        if not self.is_clone(clone):
            return False

        if len(self.patches) == 0 and len(clone.patches) == 0:
            return True

        if self.patches_uuid == clone.patches_uuid:
            if len(self.patches) != len(clone.patches):
                logger.warning("WARNING: something went wrong, same patch uuid but different length of patches.")
            else:
                return True

    def memory_required(self, input_shape) -> int:
        assert isinstance(self.model, BaseModel)
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"])  # Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(self.model_options, pre_cfg_function, disable_cfg1_optimization)

    def set_model_unet_function_wrapper(self, unet_wrapper_function: UnetWrapperFunction):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return utils.get_attr(self.model, name)

    @property
    def diffusion_model(self) -> torch.nn.Module | BaseModel:
        return self.get_model_object("diffusion_model")

    @diffusion_model.setter
    def diffusion_model(self, value: torch.nn.Module):
        self.add_object_patch("diffusion_model", value)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        # this pokes into the internals of diffusion model a little bit
        # todo: the base model isn't going to be aware that its diffusion model is patched this way
        if isinstance(self.model, BaseModel):
            diffusion_model = self.get_model_object("diffusion_model")
            return diffusion_model.dtype
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches: PatchDict, strength_patch=1.0, strength_model=1.0) -> list[PatchDictKey]:
        p: set[PatchDictKey] = set()
        model_sd = self.model.state_dict()
        k: PatchDictKey
        for k in patches:
            offset = None
            function = None
            if isinstance(k, str):
                key: str = k
            else:
                offset = k[1]
                key = k[0]
                if len(k) > 2:
                    function = k[2]

            if key in model_sd:
                p.add(k)
                current_patches = self.patches.get(key, [])
                current_patches.append(PatchTuple(strength_patch, patches[k], strength_model, offset, function))
                self.patches[key] = current_patches

        self.patches_uuid = uuid.uuid4()
        return list(p)

    def get_key_patches(self, filter_prefix=None):
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            bk: torch.nn.Module | None = self.backup.get(k, None)
            weight, set_func, convert_func = get_key_weight(self.model, k)
            if bk is not None:
                weight = bk.weight
            if convert_func is None:
                convert_func = lambda a, **kwargs: a

            if k in self.patches:
                p[k] = [PatchWeightTuple(weight, convert_func)] + self.patches[k]
            else:
                p[k] = [PatchWeightTuple(weight, convert_func)]
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return

        weight, set_func, convert_func = get_key_weight(self.model, key)
        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        if device_to is not None:
            temp_weight = model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = lora.calculate_weight(self.patches[key], temp_weight, key)
        if set_func is None:
            out_weight = stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            if inplace_update:
                utils.copy_to_param(self.model, key, out_weight)
            else:
                utils.set_attr_param(self.model, key, out_weight)
        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        loading = []
        for n, m in self.model.named_modules():
            if hasattr(m, "comfy_cast_weights") or hasattr(m, "weight"):
                loading.append((model_management.module_size(m), n, m))

        load_completely = []
        loading.sort(reverse=True)
        for x in loading:
            n = x[1]
            m = x[2]
            module_mem = x[0]

            lowvram_weight = False

            if not full_load and hasattr(m, "comfy_cast_weights"):
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True
                    lowvram_counter += 1
                    if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                        continue

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self.patches)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self.patches)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "comfy_cast_weights"):
                    if m.comfy_cast_weights:
                        wipe_lowvram_weight(m)

                if hasattr(m, "weight"):
                    mem_counter += module_mem
                    load_completely.append((module_mem, n, m))

        load_completely.sort(reverse=True)
        for x in load_completely:
            n = x[1]
            m = x[2]
            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights == True:
                    continue

            self.patch_weight_to_device(weight_key, device_to=device_to)
            self.patch_weight_to_device(bias_key, device_to=device_to)
            logger.debug("lowvram: loaded module regularly {} {}".format(n, m))
            m.comfy_patched_weights = True

        for x in load_completely:
            x[2].to(device_to)

        if lowvram_counter > 0:
            logger.debug("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
            self._memory_measurements.model_lowvram = True
        else:
            logger.debug("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
            self._memory_measurements.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

        self._memory_measurements.lowvram_patch_counter += patch_counter

        self.model_device = device_to
        self._memory_measurements.model_loaded_weight_memory = mem_counter
        self._memory_measurements.current_weight_patches_uuid = self.patches_uuid

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        for k in self.object_patches:
            old = utils.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if lowvram_model_memory == 0:
            full_load = True
        else:
            full_load = False

        if load_weights:
            self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            if self._memory_measurements.model_lowvram:
                for m in self.model.modules():
                    wipe_lowvram_weight(m)

                self._memory_measurements.model_lowvram = False
                self._memory_measurements.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    utils.copy_to_param(self.model, k, bk.weight)
                else:
                    utils.set_attr_param(self.model, k, bk.weight)

            self._memory_measurements.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.model_device = device_to
            self._memory_measurements.model_loaded_weight_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def partially_unload(self, device_to, memory_to_free=0):
        memory_freed = 0
        patch_counter = 0
        unload_list = []

        for n, m in self.model.named_modules():
            shift_lowvram = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = model_management.module_size(m)
                unload_list.append((module_mem, n, m))

        unload_list.sort()
        for unload in unload_list:
            if memory_to_free < memory_freed:
                break
            module_mem = unload[0]
            n = unload[1]
            m = unload[2]
            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True:
                for key in [weight_key, bias_key]:
                    bk: torch.nn.Module | None = self.backup.get(key, None)
                    if bk is not None:
                        if bk.inplace_update:
                            utils.copy_to_param(self.model, key, bk.weight)
                        else:
                            utils.set_attr_param(self.model, key, bk.weight)
                        self.backup.pop(key)

                m.to(device_to)
                if weight_key in self.patches:
                    m.weight_function = LowVramPatch(weight_key, self.patches)
                    patch_counter += 1
                if bias_key in self.patches:
                    m.bias_function = LowVramPatch(bias_key, self.patches)
                    patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
                m.comfy_patched_weights = False
                memory_freed += module_mem
                logger.debug("freed {}".format(n))

        self._memory_measurements.model_lowvram = True
        self._memory_measurements.lowvram_patch_counter += patch_counter
        self._memory_measurements.model_loaded_weight_memory -= memory_freed
        return memory_freed

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        unpatch_weights = self._memory_measurements.current_weight_patches_uuid is not None and (self._memory_measurements.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)
        # TODO: force_patch_weights should not unload + reload full model
        used = self._memory_measurements.model_loaded_weight_memory
        self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
        if unpatch_weights:
            extra_memory += (used - self._memory_measurements.model_loaded_weight_memory)

        self.patch_model(load_weights=False)
        full_load = False
        if not self._memory_measurements.model_lowvram and self._memory_measurements.model_loaded_weight_memory > 0:
            return 0
        if self._memory_measurements.model_loaded_weight_memory + extra_memory > self.model_size():
            full_load = True
        current_used = self._memory_measurements.model_loaded_weight_memory
        try:
            self.load(device_to, lowvram_model_memory=current_used + extra_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        except Exception as e:
            self.detach()
            raise e

        return self._memory_measurements.model_loaded_weight_memory - current_used

    def detach(self, unpatch_all=True):
        self.model_patches_to(self.offload_device)
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        return self.model

    def current_loaded_device(self):
        return self.model_device

    @property
    def current_device(self) -> torch.device:
        return self.current_loaded_device()

    def __str__(self):
        if hasattr(self.model, "operations"):
            operations_str = self.model.operations.__name__
        else:
            operations_str = None
        info_str = f"model_dtype={self.model_dtype()} device={self.model_device} size={naturalsize(self._memory_measurements.model_loaded_weight_memory, binary=True)} operations={operations_str}"
        if self.ckpt_name is not None:
            return f"<ModelPatcher for {self.ckpt_name} ({self.model.__class__.__name__} {info_str})>"
        else:
            return f"<ModelPatcher for {self.model.__class__.__name__} ({info_str})>"

    def calculate_weight(self, patches, weight, key, intermediate_dtype=torch.float32):
        print("WARNING the ModelPatcher.calculate_weight function is deprecated, please use: comfy.lora.calculate_weight instead")
        return lora.calculate_weight(patches, weight, key, intermediate_dtype=intermediate_dtype)

    def __del__(self):
        self.detach(unpatch_all=False)
