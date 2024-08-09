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

import torch
import copy
import inspect
import logging
import uuid
import collections

import comfy.utils
import comfy.model_management
from comfy.types import UnetWrapperFunction


def weight_decompose(dora_scale, weight, lora_diff, alpha, strength):
    dora_scale = comfy.model_management.cast_to_device(dora_scale, weight.device, torch.float32)
    lora_diff *= alpha
    weight_calc = weight + lora_diff.type(weight.dtype)
    weight_norm = (
        weight_calc.transpose(0, 1)
        .reshape(weight_calc.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
        .transpose(0, 1)
    )

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight


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
    def __init__(self, key, model_patcher):
        self.key = key
        self.model_patcher = model_patcher
    def __call__(self, weight):
        return self.model_patcher.calculate_weight(self.model_patcher.patches[self.key], weight, self.key)


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        self.size = size
        self.model = model
        if not hasattr(self.model, 'device'):
            logging.info("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device

        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        self.patches_uuid = uuid.uuid4()

        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0

        if not hasattr(self.model, 'lowvram_patch_counter'):
            self.model.lowvram_patch_counter = 0

        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False

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
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def clone_has_same_weights(self, clone):
        if not self.is_clone(clone):
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

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
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
        comfy.model_management.unload_model_clones(self)
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
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

        weight = comfy.utils.get_attr(self.model, key)

        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def patch_model(self, device_to=None, patch_weights=True):
        for k in self.object_patches:
            old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning("could not patch. key doesn't exist in model: {}".format(key))
                    continue

                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.model.device = device_to
                self.model.model_loaded_weight_memory = self.model_size()

        return self.model

    def lowvram_load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False):
        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = comfy.model_management.module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True
                    lowvram_counter += 1
                    if m.comfy_cast_weights:
                        continue

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "comfy_cast_weights"):
                    if m.comfy_cast_weights:
                        wipe_lowvram_weight(m)

                if hasattr(m, "weight"):
                    mem_counter += comfy.model_management.module_size(m)
                    param = list(m.parameters())
                    if len(param) > 0:
                        weight = param[0]
                        if weight.device == device_to:
                            continue

                    self.patch_weight_to_device(weight_key) #TODO: speed this up without OOM
                    self.patch_weight_to_device(bias_key)
                    m.to(device_to)
                    logging.debug("lowvram: loaded module regularly {} {}".format(n, m))

        if lowvram_counter > 0:
            logging.info("loaded in lowvram mode {}".format(lowvram_model_memory / (1024 * 1024)))
            self.model.model_lowvram = True
        else:
            logging.info("loaded completely {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024)))
            self.model.model_lowvram = False
        self.model.lowvram_patch_counter += patch_counter
        self.model.device = device_to
        self.model.model_loaded_weight_memory = mem_counter


    def patch_model_lowvram(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False):
        self.patch_model(device_to, patch_weights=False)
        self.lowvram_load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights)
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            strength = p[0]
            v = p[1]
            strength_model = p[2]
            offset = p[3]
            function = p[4]
            if function is None:
                function = lambda a: a

            old_weight = None
            if offset is not None:
                old_weight = weight
                weight = weight.narrow(offset[0], offset[1], offset[2])

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key), )

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if strength != 0.0:
                    if w1.shape != weight.shape:
                        logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                    else:
                        weight += function(strength * comfy.model_management.cast_to_device(w1, weight.device, weight.dtype))
            elif patch_type == "lora": #lora/locon
                mat1 = comfy.model_management.cast_to_device(v[0], weight.device, torch.float32)
                mat2 = comfy.model_management.cast_to_device(v[1], weight.device, torch.float32)
                dora_scale = v[4]
                if v[2] is not None:
                    alpha = v[2] / mat2.shape[0]
                else:
                    alpha = 1.0

                if v[3] is not None:
                    #locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = comfy.model_management.cast_to_device(v[3], weight.device, torch.float32)
                    final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                try:
                    lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dora_scale = v[8]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(comfy.model_management.cast_to_device(w1_a, weight.device, torch.float32),
                                  comfy.model_management.cast_to_device(w1_b, weight.device, torch.float32))
                else:
                    w1 = comfy.model_management.cast_to_device(w1, weight.device, torch.float32)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(comfy.model_management.cast_to_device(w2_a, weight.device, torch.float32),
                                      comfy.model_management.cast_to_device(w2_b, weight.device, torch.float32))
                    else:
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                          comfy.model_management.cast_to_device(t2, weight.device, torch.float32),
                                          comfy.model_management.cast_to_device(w2_b, weight.device, torch.float32),
                                          comfy.model_management.cast_to_device(w2_a, weight.device, torch.float32))
                else:
                    w2 = comfy.model_management.cast_to_device(w2, weight.device, torch.float32)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha = v[2] / dim
                else:
                    alpha = 1.0

                try:
                    lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha = v[2] / w1b.shape[0]
                else:
                    alpha = 1.0

                w2a = v[3]
                w2b = v[4]
                dora_scale = v[7]
                if v[5] is not None: #cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      comfy.model_management.cast_to_device(t1, weight.device, torch.float32),
                                      comfy.model_management.cast_to_device(w1b, weight.device, torch.float32),
                                      comfy.model_management.cast_to_device(w1a, weight.device, torch.float32))

                    m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      comfy.model_management.cast_to_device(t2, weight.device, torch.float32),
                                      comfy.model_management.cast_to_device(w2b, weight.device, torch.float32),
                                      comfy.model_management.cast_to_device(w2a, weight.device, torch.float32))
                else:
                    m1 = torch.mm(comfy.model_management.cast_to_device(w1a, weight.device, torch.float32),
                                  comfy.model_management.cast_to_device(w1b, weight.device, torch.float32))
                    m2 = torch.mm(comfy.model_management.cast_to_device(w2a, weight.device, torch.float32),
                                  comfy.model_management.cast_to_device(w2b, weight.device, torch.float32))

                try:
                    lora_diff = (m1 * m2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha = v[4] / v[0].shape[0]
                else:
                    alpha = 1.0

                dora_scale = v[5]

                a1 = comfy.model_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, torch.float32)
                a2 = comfy.model_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, torch.float32)
                b1 = comfy.model_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, torch.float32)
                b2 = comfy.model_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, torch.float32)

                try:
                    lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            else:
                logging.warning("patch type not recognized {} {}".format(patch_type, key))

            if old_weight is not None:
                weight = old_weight

        return weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            if self.model.model_lowvram:
                for m in self.model.modules():
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

            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.model.device = device_to
            self.model.model_loaded_weight_memory = 0

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def partially_unload(self, device_to, memory_to_free=0):
        memory_freed = 0
        patch_counter = 0

        for n, m in list(self.model.named_modules())[::-1]:
            if memory_to_free < memory_freed:
                break

            shift_lowvram = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = comfy.model_management.module_size(m)
                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)


                if m.weight is not None and m.weight.device != device_to:
                    for key in [weight_key, bias_key]:
                        bk = self.backup.get(key, None)
                        if bk is not None:
                            if bk.inplace_update:
                                comfy.utils.copy_to_param(self.model, key, bk.weight)
                            else:
                                comfy.utils.set_attr_param(self.model, key, bk.weight)
                            self.backup.pop(key)

                    m.to(device_to)
                    if weight_key in self.patches:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                    if bias_key in self.patches:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True
                    memory_freed += module_mem
                    logging.debug("freed {}".format(n))

        self.model.model_lowvram = True
        self.model.lowvram_patch_counter += patch_counter
        self.model.model_loaded_weight_memory -= memory_freed
        return memory_freed

    def partially_load(self, device_to, extra_memory=0):
        if self.model.model_lowvram == False:
            return 0
        if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
            pass #TODO: Full load
        current_used = self.model.model_loaded_weight_memory
        self.lowvram_load(device_to, lowvram_model_memory=current_used + extra_memory)
        return self.model.model_loaded_weight_memory - current_used

    def current_loaded_device(self):
        return self.model.device
