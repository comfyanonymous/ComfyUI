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
from enum import Enum
import math
import os
import logging
import comfy.utils
import comfy.model_management
import comfy.model_detection
import comfy.model_patcher
import comfy.ops
import comfy.latent_formats

import comfy.cldm.cldm
import comfy.t2i_adapter.adapter
import comfy.ldm.cascade.controlnet
import comfy.cldm.mmdit
import comfy.ldm.hydit.controlnet
import comfy.ldm.flux.controlnet


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    #print(current_batch_size, target_batch_size)
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)

class StrengthType(Enum):
    CONSTANT = 1
    LINEAR_UP = 2

class ControlBase:
    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.latent_format = None
        self.vae = None
        self.global_average_pooling = False
        self.timestep_range = None
        self.compression_ratio = 8
        self.upscale_algorithm = 'nearest-exact'
        self.extra_args = {}

        if device is None:
            device = comfy.model_management.get_torch_device()
        self.device = device
        self.previous_controlnet = None
        self.extra_conds = []
        self.strength_type = StrengthType.CONSTANT

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0), vae=None):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        if self.latent_format is not None:
            self.vae = vae
        return self

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (percent_to_timestep_function(self.timestep_percent_range[0]), percent_to_timestep_function(self.timestep_percent_range[1]))
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None
        self.timestep_range = None

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range
        c.global_average_pooling = self.global_average_pooling
        c.compression_ratio = self.compression_ratio
        c.upscale_algorithm = self.upscale_algorithm
        c.latent_format = self.latent_format
        c.extra_args = self.extra_args.copy()
        c.vae = self.vae
        c.extra_conds = self.extra_conds.copy()
        c.strength_type = self.strength_type

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0

    def control_merge(self, control, control_prev, output_dtype):
        out = {'input':[], 'middle':[], 'output': []}

        for key in control:
            control_output = control[key]
            applied_to = set()
            for i in range(len(control_output)):
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                    if x not in applied_to: #memory saving strategy, allow shared tensors and only apply strength to shared tensors once
                        applied_to.add(x)
                        if self.strength_type == StrengthType.CONSTANT:
                            x *= self.strength
                        elif self.strength_type == StrengthType.LINEAR_UP:
                            x *= (self.strength ** float(len(control_output) - i))

                    if output_dtype is not None and x.dtype != output_dtype:
                        x = x.to(output_dtype)

                out[key].append(x)

        if control_prev is not None:
            for x in ['input', 'middle', 'output']:
                o = out[x]
                for i in range(len(control_prev[x])):
                    prev_val = control_prev[x][i]
                    if i >= len(o):
                        o.append(prev_val)
                    elif prev_val is not None:
                        if o[i] is None:
                            o[i] = prev_val
                        else:
                            if o[i].shape[0] < prev_val.shape[0]:
                                o[i] = prev_val + o[i]
                            else:
                                o[i] = prev_val + o[i] #TODO: change back to inplace add if shared tensors stop being an issue
        return out

    def set_extra_arg(self, argument, value=None):
        self.extra_args[argument] = value


class ControlNet(ControlBase):
    def __init__(self, control_model=None, global_average_pooling=False, compression_ratio=8, latent_format=None, device=None, load_device=None, manual_cast_dtype=None, extra_conds=["y"], strength_type=StrengthType.CONSTANT):
        super().__init__(device)
        self.control_model = control_model
        self.load_device = load_device
        if control_model is not None:
            self.control_model_wrapped = comfy.model_patcher.ModelPatcher(self.control_model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device())

        self.compression_ratio = compression_ratio
        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None
        self.manual_cast_dtype = manual_cast_dtype
        self.latent_format = latent_format
        self.extra_conds += extra_conds
        self.strength_type = strength_type

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        if self.cond_hint is None or x_noisy.shape[2] * self.compression_ratio != self.cond_hint.shape[2] or x_noisy.shape[3] * self.compression_ratio != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            compression_ratio = self.compression_ratio
            if self.vae is not None:
                compression_ratio *= self.vae.downscale_ratio
            self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * compression_ratio, x_noisy.shape[2] * compression_ratio, self.upscale_algorithm, "center")
            if self.vae is not None:
                loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                self.cond_hint = self.vae.encode(self.cond_hint.movedim(1, -1))
                comfy.model_management.load_models_gpu(loaded_models)
            if self.latent_format is not None:
                self.cond_hint = self.latent_format.process_in(self.cond_hint)
            self.cond_hint = self.cond_hint.to(device=self.device, dtype=dtype)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        context = cond.get('crossattn_controlnet', cond['c_crossattn'])
        extra = self.extra_args.copy()
        for c in self.extra_conds:
            temp = cond.get(c, None)
            if temp is not None:
                extra[c] = temp.to(dtype)

        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.to(dtype), context=context.to(dtype), **extra)
        return self.control_merge(control, control_prev, output_dtype=None)

    def copy(self):
        c = ControlNet(None, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.model_sampling

    def cleanup(self):
        self.model_sampling_current = None
        super().cleanup()

class ControlLoraOps:
    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        def __init__(self, in_features: int, out_features: int, bias: bool = True,
                    device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

        def forward(self, input):
            weight, bias = comfy.ops.cast_bias_weight(self, input)
            if self.up is not None:
                return torch.nn.functional.linear(input, weight + (torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), bias)
            else:
                return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = False
            self.output_padding = 0
            self.groups = groups
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None


        def forward(self, input):
            weight, bias = comfy.ops.cast_bias_weight(self, input)
            if self.up is not None:
                return torch.nn.functional.conv2d(input, weight + (torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                return torch.nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class ControlLora(ControlNet):
    def __init__(self, control_weights, global_average_pooling=False, device=None):
        ControlBase.__init__(self, device)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling
        self.extra_conds += ["y"]

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = self.control_weights["input_hint_block.0.weight"].shape[1]
        self.manual_cast_dtype = model.manual_cast_dtype
        dtype = model.get_dtype()
        if self.manual_cast_dtype is None:
            class control_lora_ops(ControlLoraOps, comfy.ops.disable_weight_init):
                pass
        else:
            class control_lora_ops(ControlLoraOps, comfy.ops.manual_cast):
                pass
            dtype = self.manual_cast_dtype

        controlnet_config["operations"] = control_lora_ops
        controlnet_config["dtype"] = dtype
        self.control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)
        self.control_model.to(comfy.model_management.get_torch_device())
        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()
        cm = self.control_model.state_dict()

        for k in sd:
            weight = sd[k]
            try:
                comfy.utils.set_attr_param(self.control_model, k, weight)
            except:
                pass

        for k in self.control_weights:
            if k not in {"lora_controlnet"}:
                comfy.utils.set_attr_param(self.control_model, k, self.control_weights[k].to(dtype).to(comfy.model_management.get_torch_device()))

    def copy(self):
        c = ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def cleanup(self):
        del self.control_model
        self.control_model = None
        super().cleanup()

    def get_models(self):
        out = ControlBase.get_models(self)
        return out

    def inference_memory_requirements(self, dtype):
        return comfy.utils.calculate_parameters(self.control_weights) * comfy.model_management.dtype_size(dtype) + ControlBase.inference_memory_requirements(self, dtype)

def controlnet_config(sd):
    model_config = comfy.model_detection.model_config_from_unet(sd, "", True)

    supported_inference_dtypes = model_config.supported_inference_dtypes

    controlnet_config = model_config.unet_config
    unet_dtype = comfy.model_management.unet_dtype(supported_dtypes=supported_inference_dtypes)
    load_device = comfy.model_management.get_torch_device()
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        operations = comfy.ops.manual_cast
    else:
        operations = comfy.ops.disable_weight_init

    offload_device = comfy.model_management.unet_offload_device()
    return model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device

def controlnet_load_state_dict(control_model, sd):
    missing, unexpected = control_model.load_state_dict(sd, strict=False)

    if len(missing) > 0:
        logging.warning("missing controlnet keys: {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("unexpected controlnet keys: {}".format(unexpected))
    return control_model

def load_controlnet_mmdit(sd):
    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(new_sd)
    num_blocks = comfy.model_detection.count_blocks(new_sd, 'joint_blocks.{}.')
    for k in sd:
        new_sd[k] = sd[k]

    control_model = comfy.cldm.mmdit.ControlNet(num_blocks=num_blocks, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.SD3()
    latent_format.shift_factor = 0 #SD3 controlnet weirdness
    control = ControlNet(control_model, compression_ratio=1, latent_format=latent_format, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control


def load_controlnet_hunyuandit(controlnet_data):
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(controlnet_data)

    control_model = comfy.ldm.hydit.controlnet.HunYuanControlNet(operations=operations, device=offload_device, dtype=unet_dtype)
    control_model = controlnet_load_state_dict(control_model, controlnet_data)

    latent_format = comfy.latent_formats.SDXL()
    extra_conds = ['text_embedding_mask', 'encoder_hidden_states_t5', 'text_embedding_mask_t5', 'image_meta_size', 'style', 'cos_cis_img', 'sin_cis_img']
    control = ControlNet(control_model, compression_ratio=1, latent_format=latent_format, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds, strength_type=StrengthType.CONSTANT)
    return control

def load_controlnet_flux_xlabs(sd):
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(sd)
    control_model = comfy.ldm.flux.controlnet.ControlNetFlux(operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, sd)
    extra_conds = ['y', 'guidance']
    control = ControlNet(control_model, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control

def load_controlnet_flux_instantx(sd):
    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(new_sd)
    for k in sd:
        new_sd[k] = sd[k]

    num_union_modes = 0
    union_cnet = "controlnet_mode_embedder.weight"
    if union_cnet in new_sd:
        num_union_modes = new_sd[union_cnet].shape[0]

    control_model = comfy.ldm.flux.controlnet.ControlNetFlux(latent_input=True, num_union_modes=num_union_modes, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.Flux()
    extra_conds = ['y', 'guidance']
    control = ControlNet(control_model, compression_ratio=1, latent_format=latent_format, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control

def load_controlnet(ckpt_path, model=None):
    controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    if 'after_proj_list.18.bias' in controlnet_data.keys(): #Hunyuan DiT
        return load_controlnet_hunyuandit(controlnet_data)

    if "lora_controlnet" in controlnet_data:
        return ControlLora(controlnet_data)

    controlnet_config = None
    supported_inference_dtypes = None

    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        controlnet_config = comfy.model_detection.unet_config_from_diffusers_unet(controlnet_data)
        diffusers_keys = comfy.utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        if "control_add_embedding.linear_1.bias" in controlnet_data: #Union Controlnet
            controlnet_config["union_controlnet_num_control_type"] = controlnet_data["task_embedding"].shape[0]
            for k in list(controlnet_data.keys()):
                new_k = k.replace('.attn.in_proj_', '.attn.in_proj.')
                new_sd[new_k] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            logging.warning("leftover keys: {}".format(leftover_keys))
        controlnet_data = new_sd
    elif "controlnet_blocks.0.weight" in controlnet_data: #SD3 diffusers format
        if "double_blocks.0.img_attn.norm.key_norm.scale" in controlnet_data:
            return load_controlnet_flux_xlabs(controlnet_data)
        elif "pos_embed_input.proj.weight" in controlnet_data:
            return load_controlnet_mmdit(controlnet_data)
        elif "controlnet_x_embedder.weight" in controlnet_data:
            return load_controlnet_flux_instantx(controlnet_data)

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        net = load_t2i_adapter(controlnet_data)
        if net is None:
            logging.error("error checkpoint does not contain controlnet or t2i adapter data {}".format(ckpt_path))
        return net

    if controlnet_config is None:
        model_config = comfy.model_detection.model_config_from_unet(controlnet_data, prefix, True)
        supported_inference_dtypes = model_config.supported_inference_dtypes
        controlnet_config = model_config.unet_config

    load_device = comfy.model_management.get_torch_device()
    if supported_inference_dtypes is None:
        unet_dtype = comfy.model_management.unet_dtype()
    else:
        unet_dtype = comfy.model_management.unet_dtype(supported_dtypes=supported_inference_dtypes)

    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = comfy.ops.manual_cast
    controlnet_config["dtype"] = unet_dtype
    controlnet_config["device"] = comfy.model_management.unet_offload_device()
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                comfy.model_management.load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                logging.warning("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)

    if len(missing) > 0:
        logging.warning("missing controlnet keys: {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("unexpected controlnet keys: {}".format(unexpected))

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    control = ControlNet(control_model, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control

class T2IAdapter(ControlBase):
    def __init__(self, t2i_model, channels_in, compression_ratio, upscale_algorithm, device=None):
        super().__init__(device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None
        self.compression_ratio = compression_ratio
        self.upscale_algorithm = upscale_algorithm

    def scale_image_to(self, width, height):
        unshuffle_amount = self.t2i_model.unshuffle_amount
        width = math.ceil(width / unshuffle_amount) * unshuffle_amount
        height = math.ceil(height / unshuffle_amount) * unshuffle_amount
        return width, height

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        if self.cond_hint is None or x_noisy.shape[2] * self.compression_ratio != self.cond_hint.shape[2] or x_noisy.shape[3] * self.compression_ratio != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            width, height = self.scale_image_to(x_noisy.shape[3] * self.compression_ratio, x_noisy.shape[2] * self.compression_ratio)
            self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, width, height, self.upscale_algorithm, "center").float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        if self.control_input is None:
            self.t2i_model.to(x_noisy.dtype)
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint.to(x_noisy.dtype))
            self.t2i_model.cpu()

        control_input = {}
        for k in self.control_input:
            control_input[k] = list(map(lambda a: None if a is None else a.clone(), self.control_input[k]))

        return self.control_merge(control_input, control_prev, x_noisy.dtype)

    def copy(self):
        c = T2IAdapter(self.t2i_model, self.channels_in, self.compression_ratio, self.upscale_algorithm)
        self.copy_to(c)
        return c

def load_t2i_adapter(t2i_data):
    compression_ratio = 8
    upscale_algorithm = 'nearest-exact'

    if 'adapter' in t2i_data:
        t2i_data = t2i_data['adapter']
    if 'adapter.body.0.resnets.0.block1.weight' in t2i_data: #diffusers format
        prefix_replace = {}
        for i in range(4):
            for j in range(2):
                prefix_replace["adapter.body.{}.resnets.{}.".format(i, j)] = "body.{}.".format(i * 2 + j)
            prefix_replace["adapter.body.{}.".format(i, j)] = "body.{}.".format(i * 2)
        prefix_replace["adapter."] = ""
        t2i_data = comfy.utils.state_dict_prefix_replace(t2i_data, prefix_replace)
    keys = t2i_data.keys()

    if "body.0.in_conv.weight" in keys:
        cin = t2i_data['body.0.in_conv.weight'].shape[1]
        model_ad = comfy.t2i_adapter.adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif 'conv_in.weight' in keys:
        cin = t2i_data['conv_in.weight'].shape[1]
        channel = t2i_data['conv_in.weight'].shape[0]
        ksize = t2i_data['body.0.block2.weight'].shape[2]
        use_conv = False
        down_opts = list(filter(lambda a: a.endswith("down_opt.op.weight"), keys))
        if len(down_opts) > 0:
            use_conv = True
        xl = False
        if cin == 256 or cin == 768:
            xl = True
        model_ad = comfy.t2i_adapter.adapter.Adapter(cin=cin, channels=[channel, channel*2, channel*4, channel*4][:4], nums_rb=2, ksize=ksize, sk=True, use_conv=use_conv, xl=xl)
    elif "backbone.0.0.weight" in keys:
        model_ad = comfy.ldm.cascade.controlnet.ControlNet(c_in=t2i_data['backbone.0.0.weight'].shape[1], proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63])
        compression_ratio = 32
        upscale_algorithm = 'bilinear'
    elif "backbone.10.blocks.0.weight" in keys:
        model_ad = comfy.ldm.cascade.controlnet.ControlNet(c_in=t2i_data['backbone.0.weight'].shape[1], bottleneck_mode="large", proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63])
        compression_ratio = 1
        upscale_algorithm = 'nearest-exact'
    else:
        return None

    missing, unexpected = model_ad.load_state_dict(t2i_data)
    if len(missing) > 0:
        logging.warning("t2i missing {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("t2i unexpected {}".format(unexpected))

    return T2IAdapter(model_ad, model_ad.input_channels, compression_ratio, upscale_algorithm)
