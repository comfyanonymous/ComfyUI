from __future__ import annotations

import logging
import os
from enum import Enum

import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io

CLAMP_QUANTILE = 0.99


def extract_lora(diff, rank):
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    U, S, Vh = torch.linalg.svd(diff.float())
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
    if conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return (U, Vh)


class LORAType(Enum):
    STANDARD = 0
    FULL_DIFF = 1


LORA_TYPES = {
    "standard": LORAType.STANDARD,
    "full_diff": LORAType.FULL_DIFF,
}


def calc_lora_model(model_diff, rank, prefix_model, prefix_lora, output_sd, lora_type, bias_diff=False):
    comfy.model_management.load_models_gpu([model_diff], force_patch_weights=True)
    sd = model_diff.model_state_dict(filter_prefix=prefix_model)

    for k in sd:
        if k.endswith(".weight"):
            weight_diff = sd[k]
            if lora_type == LORAType.STANDARD:
                if weight_diff.ndim < 2:
                    if bias_diff:
                        output_sd["{}{}.diff".format(prefix_lora, k[len(prefix_model):-7])] = weight_diff.contiguous().half().cpu()
                    continue
                try:
                    out = extract_lora(weight_diff, rank)
                    output_sd["{}{}.lora_up.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[0].contiguous().half().cpu()
                    output_sd["{}{}.lora_down.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[1].contiguous().half().cpu()
                except Exception:
                    logging.warning("Could not generate lora weights for key {}, is the weight difference a zero?".format(k))
            elif lora_type == LORAType.FULL_DIFF:
                output_sd["{}{}.diff".format(prefix_lora, k[len(prefix_model):-7])] = weight_diff.contiguous().half().cpu()

        elif bias_diff and k.endswith(".bias"):
            output_sd["{}{}.diff_b".format(prefix_lora, k[len(prefix_model):-5])] = sd[k].contiguous().half().cpu()
    return output_sd


class LoraSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoraSave_V3",
            display_name="Extract and Save Lora _V3",
            category="_for_testing",
            is_output_node=True,
            inputs=[
                io.String.Input("filename_prefix", default="loras/ComfyUI_extracted_lora"),
                io.Int.Input("rank", default=8, min=1, max=4096, step=1),
                io.Combo.Input("lora_type", options=list(LORA_TYPES.keys())),
                io.Boolean.Input("bias_diff", default=True),
                io.Model.Input(
                    id="model_diff", optional=True, tooltip="The ModelSubtract output to be converted to a lora."
                ),
                io.Clip.Input(
                    id="text_encoder_diff", optional=True, tooltip="The CLIPSubtract output to be converted to a lora."
                ),
            ],
            outputs=[],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, filename_prefix, rank, lora_type, bias_diff, model_diff=None, text_encoder_diff=None):
        if model_diff is None and text_encoder_diff is None:
            return io.NodeOutput()

        lora_type = LORA_TYPES.get(lora_type)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )

        output_sd = {}
        if model_diff is not None:
            output_sd = calc_lora_model(
                model_diff, rank, "diffusion_model.", "diffusion_model.", output_sd, lora_type, bias_diff=bias_diff
            )
        if text_encoder_diff is not None:
            output_sd = calc_lora_model(
                text_encoder_diff.patcher, rank, "", "text_encoders.", output_sd, lora_type, bias_diff=bias_diff
            )

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        return io.NodeOutput()


NODES_LIST = [
    LoraSave,
]
