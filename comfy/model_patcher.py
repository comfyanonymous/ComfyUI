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

from __future__ import annotations

import collections
import copy
import inspect
import logging
import math
import uuid
from typing import Callable, Optional

import torch

import comfy.float
import comfy.hooks
import comfy.lora
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
from comfy.comfy_types import UnetWrapperFunction
from comfy.patcher_extension import CallbacksMP, PatcherInjection, WrappersMP


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

def create_model_options_clone(orig_model_options: dict):
    return comfy.patcher_extension.copy_nested_dicts(orig_model_options)

def create_hook_patches_clone(orig_hook_patches):
    new_hook_patches = {}
    for hook_ref in orig_hook_patches:
        new_hook_patches[hook_ref] = {}
        for k in orig_hook_patches[hook_ref]:
            new_hook_patches[hook_ref][k] = orig_hook_patches[hook_ref][k][:]
    return new_hook_patches

def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []

def move_weight_functions(m, device):
    if device is None:
        return 0

    memory = 0
    if hasattr(m, "weight_function"):
        for f in m.weight_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)

    if hasattr(m, "bias_function"):
        for f in m.bias_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)
    return memory

class LowVramPatch:
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches
    def __call__(self, weight):
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]: #intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return comfy.float.stochastic_rounding(comfy.lora.calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), weight.dtype, seed=string_to_seed(self.key))

        return comfy.lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)

def get_key_weight(model, key):
    set_func = None
    convert_func = None
    op_keys = key.rsplit('.', 1)
    if len(op_keys) < 2:
        weight = comfy.utils.get_attr(model, key)
    else:
        op = comfy.utils.get_attr(model, op_keys[0])
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
            weight = comfy.utils.get_attr(model, key)

    return weight, set_func, convert_func

class AutoPatcherEjector:
    def __init__(self, model: 'ModelPatcher', skip_and_inject_on_exit_only=False):
        self.model = model
        self.was_injected = False
        self.prev_skip_injection = False
        self.skip_and_inject_on_exit_only = skip_and_inject_on_exit_only

    def __enter__(self):
        self.was_injected = False
        self.prev_skip_injection = self.model.skip_injection
        if self.skip_and_inject_on_exit_only:
            self.model.skip_injection = True
        if self.model.is_injected:
            self.model.eject_model()
            self.was_injected = True

    def __exit__(self, *args):
        if self.skip_and_inject_on_exit_only:
            self.model.skip_injection = self.prev_skip_injection
            self.model.inject_model()
        if self.was_injected and not self.model.skip_injection:
            self.model.inject_model()
        self.model.skip_injection = self.prev_skip_injection

class MemoryCounter:
    def __init__(self, initial: int, minimum=0):
        self.value = initial
        self.minimum = minimum
        # TODO: add a safe limit besides 0

    def use(self, weight: torch.Tensor):
        weight_size = weight.nelement() * weight.element_size()
        if self.is_useable(weight_size):
            self.decrement(weight_size)
            return True
        return False

    def is_useable(self, used: int):
        return self.value - used > self.minimum

    def decrement(self, used: int):
        self.value -= used

class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        self.size = size
        self.model = model
        if not hasattr(self.model, 'device'):
            logging.debug("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device

        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.weight_wrapper_patches = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        self.force_cast_weights = False
        self.patches_uuid = uuid.uuid4()
        self.parent = None

        self.attachments: dict[str] = {}
        self.additional_models: dict[str, list[ModelPatcher]] = {}
        self.callbacks: dict[str, dict[str, list[Callable]]] = CallbacksMP.init_callbacks()
        self.wrappers: dict[str, dict[str, list[Callable]]] = WrappersMP.init_wrappers()

        self.is_injected = False
        self.skip_injection = False
        self.injections: dict[str, list[PatcherInjection]] = {}

        self.hook_patches: dict[comfy.hooks._HookRef] = {}
        self.hook_patches_backup: dict[comfy.hooks._HookRef] = None
        self.hook_backup: dict[str, tuple[torch.Tensor, torch.device]] = {}
        self.cached_hook_patches: dict[comfy.hooks.HookGroup, dict[str, torch.Tensor]] = {}
        self.current_hooks: Optional[comfy.hooks.HookGroup] = None
        self.forced_hooks: Optional[comfy.hooks.HookGroup] = None  # NOTE: only used for CLIP at this time
        self.is_clip = False
        self.hook_mode = comfy.hooks.EnumHookMode.MaxSpeed

        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0

        if not hasattr(self.model, 'lowvram_patch_counter'):
            self.model.lowvram_patch_counter = 0

        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False

        if not hasattr(self.model, 'current_weight_patches_uuid'):
            self.model.current_weight_patches_uuid = None

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = comfy.model_management.module_size(self.model)
        return self.size

    def loaded_size(self):
        return self.model.model_loaded_weight_memory

    def lowvram_patch_counter(self):
        return self.model.lowvram_patch_counter

    def clone(self):
        n = self.__class__(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.parent = self

        n.force_cast_weights = self.force_cast_weights

        # attachments
        n.attachments = {}
        for k in self.attachments:
            if hasattr(self.attachments[k], "on_model_patcher_clone"):
                n.attachments[k] = self.attachments[k].on_model_patcher_clone()
            else:
                n.attachments[k] = self.attachments[k]
        # additional models
        for k, c in self.additional_models.items():
            n.additional_models[k] = [x.clone() for x in c]
        # callbacks
        for k, c in self.callbacks.items():
            n.callbacks[k] = {}
            for k1, c1 in c.items():
                n.callbacks[k][k1] = c1.copy()
        # sample wrappers
        for k, w in self.wrappers.items():
            n.wrappers[k] = {}
            for k1, w1 in w.items():
                n.wrappers[k][k1] = w1.copy()
        # injection
        n.is_injected = self.is_injected
        n.skip_injection = self.skip_injection
        for k, i in self.injections.items():
            n.injections[k] = i.copy()
        # hooks
        n.hook_patches = create_hook_patches_clone(self.hook_patches)
        n.hook_patches_backup = create_hook_patches_clone(self.hook_patches_backup) if self.hook_patches_backup else self.hook_patches_backup
        for group in self.cached_hook_patches:
            n.cached_hook_patches[group] = {}
            for k in self.cached_hook_patches[group]:
                n.cached_hook_patches[group][k] = self.cached_hook_patches[group][k]
        n.hook_backup = self.hook_backup
        n.current_hooks = self.current_hooks.clone() if self.current_hooks else self.current_hooks
        n.forced_hooks = self.forced_hooks.clone() if self.forced_hooks else self.forced_hooks
        n.is_clip = self.is_clip
        n.hook_mode = self.hook_mode

        for callback in self.get_all_callbacks(CallbacksMP.ON_CLONE):
            callback(self, n)
        return n

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def clone_has_same_weights(self, clone: 'ModelPatcher'):
        if not self.is_clone(clone):
            return False

        if self.current_hooks != clone.current_hooks:
            return False
        if self.forced_hooks != clone.forced_hooks:
            return False
        if self.hook_patches.keys() != clone.hook_patches.keys():
            return False
        if self.attachments.keys() != clone.attachments.keys():
            return False
        if self.additional_models.keys() != clone.additional_models.keys():
            return False
        for key in self.callbacks:
            if len(self.callbacks[key]) != len(clone.callbacks[key]):
                return False
        for key in self.wrappers:
            if len(self.wrappers[key]) != len(clone.wrappers[key]):
                return False
        if self.injections.keys() != clone.injections.keys():
            return False

        if len(self.patches) == 0 and len(clone.patches) == 0:
            return True

        if self.patches_uuid == clone.patches_uuid:
            if len(self.patches) != len(clone.patches):
                logging.warning("WARNING: something went wrong, same patch uuid but different length of patches.")
            else:
                return True

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
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

    def set_model_emb_patch(self, patch):
        self.set_model_patch(patch, "emb_patch")

    def set_model_forward_timestep_embed_patch(self, patch):
        self.set_model_patch(patch, "forward_timestep_embed_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_compute_dtype(self, dtype):
        self.add_object_patch("manual_cast_dtype", dtype)
        if dtype is not None:
            self.force_cast_weights = True
        self.patches_uuid = uuid.uuid4() #TODO: optimize by preventing a full model reload for this

    def add_weight_wrapper(self, name, function):
        self.weight_wrapper_patches[name] = self.weight_wrapper_patches.get(name, []) + [function]
        self.patches_uuid = uuid.uuid4()

    def get_model_object(self, name: str) -> torch.nn.Module:
        """Retrieves a nested attribute from an object using dot notation considering
        object patches.

        Args:
            name (str): The attribute path using dot notation (e.g. "model.layer.weight")

        Returns:
            The value of the requested attribute

        Example:
            patcher = ModelPatcher()
            weight = patcher.get_model_object("layer1.conv.weight")
        """
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return comfy.utils.get_attr(self.model, name)

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
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        with self.use_ejected():
            p = set()
            model_sd = self.model.state_dict()
            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]

                if key in model_sd:
                    p.add(k)
                    current_patches = self.patches.get(key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
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
            bk = self.backup.get(k, None)
            hbk = self.hook_backup.get(k, None)
            weight, set_func, convert_func = get_key_weight(self.model, k)
            if bk is not None:
                weight = bk.weight
            if hbk is not None:
                weight = hbk[0]
            if convert_func is None:
                convert_func = lambda a, **kwargs: a

            if k in self.patches:
                p[k] = [(weight, convert_func)] + self.patches[k]
            else:
                p[k] = [(weight, convert_func)]
        return p

    def model_state_dict(self, filter_prefix=None):
        with self.use_ejected():
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
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key)
        if set_func is None:
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)
        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def _load_list(self):
        loading = []
        for n, m in self.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                loading.append((comfy.model_management.module_size(m), n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                lowvram_weight = False

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_weight = True
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed
                            continue

                cast_weight = self.force_cast_weights
                if lowvram_weight:
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                        else:
                            m.weight_function = [LowVramPatch(weight_key, self.patches)]
                            patch_counter += 1
                    if bias_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(bias_key)
                        else:
                            m.bias_function = [LowVramPatch(bias_key, self.patches)]
                            patch_counter += 1

                    cast_weight = True
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        wipe_lowvram_weight(m)

                    if full_load or mem_counter + module_mem < lowvram_model_memory:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if lowvram_counter > 0:
                logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
                logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        with self.use_ejected():
            for k in self.object_patches:
                old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
                if k not in self.object_patches_backup:
                    self.object_patches_backup[k] = old

            if lowvram_model_memory == 0:
                full_load = True
            else:
                full_load = False

            if load_weights:
                self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        self.inject_model()
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    comfy.utils.copy_to_param(self.model, k, bk.weight)
                else:
                    comfy.utils.set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.model.device = device_to
            self.model.model_loaded_weight_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def partially_unload(self, device_to, memory_to_free=0):
        with self.use_ejected():
            hooks_unpatched = False
            memory_freed = 0
            patch_counter = 0
            unload_list = self._load_list()
            unload_list.sort()
            for unload in unload_list:
                if memory_to_free < memory_freed:
                    break
                module_mem = unload[0]
                n = unload[1]
                m = unload[2]
                params = unload[3]

                lowvram_possible = hasattr(m, "comfy_cast_weights")
                if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True:
                    move_weight = True
                    for param in params:
                        key = "{}.{}".format(n, param)
                        bk = self.backup.get(key, None)
                        if bk is not None:
                            if not lowvram_possible:
                                move_weight = False
                                break

                            if not hooks_unpatched:
                                self.unpatch_hooks()
                                hooks_unpatched = True

                            if bk.inplace_update:
                                comfy.utils.copy_to_param(self.model, key, bk.weight)
                            else:
                                comfy.utils.set_attr_param(self.model, key, bk.weight)
                            self.backup.pop(key)

                    weight_key = "{}.weight".format(n)
                    bias_key = "{}.bias".format(n)
                    if move_weight:
                        cast_weight = self.force_cast_weights
                        m.to(device_to)
                        module_mem += move_weight_functions(m, device_to)
                        if lowvram_possible:
                            if weight_key in self.patches:
                                m.weight_function.append(LowVramPatch(weight_key, self.patches))
                                patch_counter += 1
                            if bias_key in self.patches:
                                m.bias_function.append(LowVramPatch(bias_key, self.patches))
                                patch_counter += 1
                            cast_weight = True

                        if cast_weight:
                            m.prev_comfy_cast_weights = m.comfy_cast_weights
                            m.comfy_cast_weights = True
                        m.comfy_patched_weights = False
                        memory_freed += module_mem
                        logging.debug("freed {}".format(n))

            self.model.model_lowvram = True
            self.model.lowvram_patch_counter += patch_counter
            self.model.model_loaded_weight_memory -= memory_freed
            return memory_freed

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        with self.use_ejected(skip_and_inject_on_exit_only=True):
            unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)
            # TODO: force_patch_weights should not unload + reload full model
            used = self.model.model_loaded_weight_memory
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
            if unpatch_weights:
                extra_memory += (used - self.model.model_loaded_weight_memory)

            self.patch_model(load_weights=False)
            full_load = False
            if self.model.model_lowvram == False and self.model.model_loaded_weight_memory > 0:
                self.apply_hooks(self.forced_hooks, force_apply=True)
                return 0
            if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
                full_load = True
            current_used = self.model.model_loaded_weight_memory
            try:
                self.load(device_to, lowvram_model_memory=current_used + extra_memory, force_patch_weights=force_patch_weights, full_load=full_load)
            except Exception as e:
                self.detach()
                raise e

            return self.model.model_loaded_weight_memory - current_used

    def detach(self, unpatch_all=True):
        self.eject_model()
        self.model_patches_to(self.offload_device)
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        for callback in self.get_all_callbacks(CallbacksMP.ON_DETACH):
            callback(self, unpatch_all)
        return self.model

    def current_loaded_device(self):
        return self.model.device

    def calculate_weight(self, patches, weight, key, intermediate_dtype=torch.float32):
        logging.warning("The ModelPatcher.calculate_weight function is deprecated, please use: comfy.lora.calculate_weight instead")
        return comfy.lora.calculate_weight(patches, weight, key, intermediate_dtype=intermediate_dtype)

    def cleanup(self):
        self.clean_hooks()
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = None
        for callback in self.get_all_callbacks(CallbacksMP.ON_CLEANUP):
            callback(self)

    def add_callback(self, call_type: str, callback: Callable):
        self.add_callback_with_key(call_type, None, callback)

    def add_callback_with_key(self, call_type: str, key: str, callback: Callable):
        c = self.callbacks.setdefault(call_type, {}).setdefault(key, [])
        c.append(callback)

    def remove_callbacks_with_key(self, call_type: str, key: str):
        c = self.callbacks.get(call_type, {})
        if key in c:
            c.pop(key)

    def get_callbacks(self, call_type: str, key: str):
        return self.callbacks.get(call_type, {}).get(key, [])

    def get_all_callbacks(self, call_type: str):
        c_list = []
        for c in self.callbacks.get(call_type, {}).values():
            c_list.extend(c)
        return c_list

    def add_wrapper(self, wrapper_type: str, wrapper: Callable):
        self.add_wrapper_with_key(wrapper_type, None, wrapper)

    def add_wrapper_with_key(self, wrapper_type: str, key: str, wrapper: Callable):
        w = self.wrappers.setdefault(wrapper_type, {}).setdefault(key, [])
        w.append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type: str, key: str):
        w = self.wrappers.get(wrapper_type, {})
        if key in w:
            w.pop(key)

    def get_wrappers(self, wrapper_type: str, key: str):
        return self.wrappers.get(wrapper_type, {}).get(key, [])

    def get_all_wrappers(self, wrapper_type: str):
        w_list = []
        for w in self.wrappers.get(wrapper_type, {}).values():
            w_list.extend(w)
        return w_list

    def set_attachments(self, key: str, attachment):
        self.attachments[key] = attachment

    def remove_attachments(self, key: str):
        if key in self.attachments:
            self.attachments.pop(key)

    def get_attachment(self, key: str):
        return self.attachments.get(key, None)

    def set_injections(self, key: str, injections: list[PatcherInjection]):
        self.injections[key] = injections

    def remove_injections(self, key: str):
        if key in self.injections:
            self.injections.pop(key)

    def get_injections(self, key: str):
        return self.injections.get(key, None)

    def set_additional_models(self, key: str, models: list['ModelPatcher']):
        self.additional_models[key] = models

    def remove_additional_models(self, key: str):
        if key in self.additional_models:
            self.additional_models.pop(key)

    def get_additional_models_with_key(self, key: str):
        return self.additional_models.get(key, [])

    def get_additional_models(self):
        all_models = []
        for models in self.additional_models.values():
            all_models.extend(models)
        return all_models

    def get_nested_additional_models(self):
        def _evaluate_sub_additional_models(prev_models: list[ModelPatcher], cache_set: set[ModelPatcher]):
            '''Make sure circular references do not cause infinite recursion.'''
            next_models = []
            for model in prev_models:
                candidates = model.get_additional_models()
                for c in candidates:
                    if c not in cache_set:
                        next_models.append(c)
                        cache_set.add(c)
            if len(next_models) == 0:
                return prev_models
            return prev_models + _evaluate_sub_additional_models(next_models, cache_set)

        all_models = self.get_additional_models()
        models_set = set(all_models)
        real_all_models = _evaluate_sub_additional_models(prev_models=all_models, cache_set=models_set)
        return real_all_models

    def use_ejected(self, skip_and_inject_on_exit_only=False):
        return AutoPatcherEjector(self, skip_and_inject_on_exit_only=skip_and_inject_on_exit_only)

    def inject_model(self):
        if self.is_injected or self.skip_injection:
            return
        for injections in self.injections.values():
            for inj in injections:
                inj.inject(self)
                self.is_injected = True
        if self.is_injected:
            for callback in self.get_all_callbacks(CallbacksMP.ON_INJECT_MODEL):
                callback(self)

    def eject_model(self):
        if not self.is_injected:
            return
        for injections in self.injections.values():
            for inj in injections:
                inj.eject(self)
        self.is_injected = False
        for callback in self.get_all_callbacks(CallbacksMP.ON_EJECT_MODEL):
            callback(self)

    def pre_run(self):
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = self
        for callback in self.get_all_callbacks(CallbacksMP.ON_PRE_RUN):
            callback(self)

    def prepare_state(self, timestep):
        for callback in self.get_all_callbacks(CallbacksMP.ON_PREPARE_STATE):
            callback(self, timestep)

    def restore_hook_patches(self):
        if self.hook_patches_backup is not None:
            self.hook_patches = self.hook_patches_backup
            self.hook_patches_backup = None

    def set_hook_mode(self, hook_mode: comfy.hooks.EnumHookMode):
        self.hook_mode = hook_mode

    def prepare_hook_patches_current_keyframe(self, t: torch.Tensor, hook_group: comfy.hooks.HookGroup, model_options: dict[str]):
        curr_t = t[0]
        reset_current_hooks = False
        transformer_options = model_options.get("transformer_options", {})
        for hook in hook_group.hooks:
            changed = hook.hook_keyframe.prepare_current_keyframe(curr_t=curr_t, transformer_options=transformer_options)
            # if keyframe changed, remove any cached HookGroups that contain hook with the same hook_ref;
            # this will cause the weights to be recalculated when sampling
            if changed:
                # reset current_hooks if contains hook that changed
                if self.current_hooks is not None:
                    for current_hook in self.current_hooks.hooks:
                        if current_hook == hook:
                            reset_current_hooks = True
                            break
                for cached_group in list(self.cached_hook_patches.keys()):
                    if cached_group.contains(hook):
                        self.cached_hook_patches.pop(cached_group)
        if reset_current_hooks:
            self.patch_hooks(None)

    def register_all_hook_patches(self, hooks: comfy.hooks.HookGroup, target_dict: dict[str], model_options: dict=None,
                                  registered: comfy.hooks.HookGroup = None):
        self.restore_hook_patches()
        if registered is None:
            registered = comfy.hooks.HookGroup()
        # handle WeightHooks
        weight_hooks_to_register: list[comfy.hooks.WeightHook] = []
        for hook in hooks.get_type(comfy.hooks.EnumHookType.Weight):
            if hook.hook_ref not in self.hook_patches:
                weight_hooks_to_register.append(hook)
            else:
                registered.add(hook)
        if len(weight_hooks_to_register) > 0:
            # clone hook_patches to become backup so that any non-dynamic hooks will return to their original state
            self.hook_patches_backup = create_hook_patches_clone(self.hook_patches)
            for hook in weight_hooks_to_register:
                hook.add_hook_patches(self, model_options, target_dict, registered)
        for callback in self.get_all_callbacks(CallbacksMP.ON_REGISTER_ALL_HOOK_PATCHES):
            callback(self, hooks, target_dict, model_options, registered)
        return registered

    def add_hook_patches(self, hook: comfy.hooks.WeightHook, patches, strength_patch=1.0, strength_model=1.0):
        with self.use_ejected():
            # NOTE: this mirrors behavior of add_patches func
            current_hook_patches: dict[str,list] = self.hook_patches.get(hook.hook_ref, {})
            p = set()
            model_sd = self.model.state_dict()
            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]

                if key in model_sd:
                    p.add(k)
                    current_patches: list[tuple] = current_hook_patches.get(key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                    current_hook_patches[key] = current_patches
            self.hook_patches[hook.hook_ref] = current_hook_patches
            # since should care about these patches too to determine if same model, reroll patches_uuid
            self.patches_uuid = uuid.uuid4()
            return list(p)

    def get_combined_hook_patches(self, hooks: comfy.hooks.HookGroup):
        # combined_patches will contain  weights of all relevant hooks, per key
        combined_patches = {}
        if hooks is not None:
            for hook in hooks.hooks:
                hook_patches: dict = self.hook_patches.get(hook.hook_ref, {})
                for key in hook_patches.keys():
                    current_patches: list[tuple] = combined_patches.get(key, [])
                    if math.isclose(hook.strength, 1.0):
                        current_patches.extend(hook_patches[key])
                    else:
                        # patches are stored as tuples: (strength_patch, (tuple_with_weights,), strength_model)
                        for patch in hook_patches[key]:
                            new_patch = list(patch)
                            new_patch[0] *= hook.strength
                            current_patches.append(tuple(new_patch))
                    combined_patches[key] = current_patches
        return combined_patches

    def apply_hooks(self, hooks: comfy.hooks.HookGroup, transformer_options: dict=None, force_apply=False):
        # TODO: return transformer_options dict with any additions from hooks
        if self.current_hooks == hooks and (not force_apply or (not self.is_clip and hooks is None)):
            return comfy.hooks.create_transformer_options_from_hooks(self, hooks, transformer_options)
        self.patch_hooks(hooks=hooks)
        for callback in self.get_all_callbacks(CallbacksMP.ON_APPLY_HOOKS):
            callback(self, hooks)
        return comfy.hooks.create_transformer_options_from_hooks(self, hooks, transformer_options)

    def patch_hooks(self, hooks: comfy.hooks.HookGroup):
        with self.use_ejected():
            if hooks is not None:
                model_sd_keys = list(self.model_state_dict().keys())
                memory_counter = None
                if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
                    # TODO: minimum_counter should have a minimum that conforms to loaded model requirements
                    memory_counter = MemoryCounter(initial=comfy.model_management.get_free_memory(self.load_device),
                                                minimum=comfy.model_management.minimum_inference_memory()*2)
                # if have cached weights for hooks, use it
                cached_weights = self.cached_hook_patches.get(hooks, None)
                if cached_weights is not None:
                    model_sd_keys_set = set(model_sd_keys)
                    for key in cached_weights:
                        if key not in model_sd_keys:
                            logging.warning(f"Cached hook could not patch. Key does not exist in model: {key}")
                            continue
                        self.patch_cached_hook_weights(cached_weights=cached_weights, key=key, memory_counter=memory_counter)
                        model_sd_keys_set.remove(key)
                    self.unpatch_hooks(model_sd_keys_set)
                else:
                    self.unpatch_hooks()
                    relevant_patches = self.get_combined_hook_patches(hooks=hooks)
                    original_weights = None
                    if len(relevant_patches) > 0:
                        original_weights = self.get_key_patches()
                    for key in relevant_patches:
                        if key not in model_sd_keys:
                            logging.warning(f"Cached hook would not patch. Key does not exist in model: {key}")
                            continue
                        self.patch_hook_weight_to_device(hooks=hooks, combined_patches=relevant_patches, key=key, original_weights=original_weights,
                                                            memory_counter=memory_counter)
            else:
                self.unpatch_hooks()
            self.current_hooks = hooks

    def patch_cached_hook_weights(self, cached_weights: dict, key: str, memory_counter: MemoryCounter):
        if key not in self.hook_backup:
            weight: torch.Tensor = comfy.utils.get_attr(self.model, key)
            target_device = self.offload_device
            if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
                used = memory_counter.use(weight)
                if used:
                    target_device = weight.device
            self.hook_backup[key] = (weight.to(device=target_device, copy=True), weight.device)
        comfy.utils.copy_to_param(self.model, key, cached_weights[key][0].to(device=cached_weights[key][1]))

    def clear_cached_hook_weights(self):
        self.cached_hook_patches.clear()
        self.patch_hooks(None)

    def patch_hook_weight_to_device(self, hooks: comfy.hooks.HookGroup, combined_patches: dict, key: str, original_weights: dict, memory_counter: MemoryCounter):
        if key not in combined_patches:
            return

        weight, set_func, convert_func = get_key_weight(self.model, key)
        weight: torch.Tensor
        if key not in self.hook_backup:
            target_device = self.offload_device
            if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
                used = memory_counter.use(weight)
                if used:
                    target_device = weight.device
            self.hook_backup[key] = (weight.to(device=target_device, copy=True), weight.device)
        # TODO: properly handle LowVramPatch, if it ends up an issue
        temp_weight = comfy.model_management.cast_to_device(weight, weight.device, torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy.lora.calculate_weight(combined_patches[key],
                                                 temp_weight,
                                                 key, original_weights=original_weights)
        del original_weights[key]
        if set_func is None:
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            set_func(out_weight, inplace_update=True, seed=string_to_seed(key))
        if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
            # TODO: disable caching if not enough system RAM to do so
            target_device = self.offload_device
            used = memory_counter.use(weight)
            if used:
                target_device = weight.device
            self.cached_hook_patches.setdefault(hooks, {})
            self.cached_hook_patches[hooks][key] = (out_weight.to(device=target_device, copy=False), weight.device)
        del temp_weight
        del out_weight
        del weight

    def unpatch_hooks(self, whitelist_keys_set: set[str]=None) -> None:
        with self.use_ejected():
            if len(self.hook_backup) == 0:
                self.current_hooks = None
                return
            keys = list(self.hook_backup.keys())
            if whitelist_keys_set:
                for k in keys:
                    if k in whitelist_keys_set:
                        comfy.utils.copy_to_param(self.model, k, self.hook_backup[k][0].to(device=self.hook_backup[k][1]))
                        self.hook_backup.pop(k)
            else:
                for k in keys:
                    comfy.utils.copy_to_param(self.model, k, self.hook_backup[k][0].to(device=self.hook_backup[k][1]))

                self.hook_backup.clear()
                self.current_hooks = None

    def clean_hooks(self):
        self.unpatch_hooks()
        self.clear_cached_hook_weights()

    def __del__(self):
        self.detach(unpatch_all=False)

