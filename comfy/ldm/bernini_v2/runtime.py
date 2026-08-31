# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Comfy-managed runtime objects for the Bernini v2 semantic planner."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.utils
from comfy.model_patcher import CoreModelPatcher
from comfy.text_encoders.llama import Llama2_, Qwen25_7BVLI_Config
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer

from .planner_model import DiffLossFM, MLPConnector
from .sharded import component_checkpoint, load_checkpoint_state_dict


class PlannerAux(nn.Module):
    """Connector, VIT diffusion decoder, and learned mask-token table."""

    def __init__(
        self,
        *,
        num_mask_tokens: int = 4096,
        hidden_size: int = 3584,
        decoder_width: int = 4096,
        decoder_depth: int = 16,
        device=None,
        dtype=None,
        operations,
    ):
        super().__init__()
        self.connector = MLPConnector(
            in_dim=hidden_size,
            out_dim_for_gen=4096,
            out_dim_for_vit=hidden_size,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.vit_decoder = DiffLossFM(
            target_channels=hidden_size,
            z_channels=hidden_size,
            depth=decoder_depth,
            width=decoder_width,
            shift=2.0,
            extra_one_step=True,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.mask_tokens = nn.Parameter(
            torch.empty(1, num_mask_tokens, hidden_size, device=device, dtype=dtype),
            requires_grad=False,
        )


@dataclass
class BerniniV2PlannerRuntime:
    """A lightweight handle tracked by Comfy through its three model patchers."""

    language_model: nn.Module
    vision_model: nn.Module
    aux: PlannerAux
    language_patcher: object
    vision_patcher: object
    aux_patcher: object
    tokenizer: object
    dtype: torch.dtype
    load_device: torch.device

    def get_models(self) -> list[object]:
        return [self.language_patcher, self.vision_patcher, self.aux_patcher]

    def load_vision(self) -> None:
        comfy.model_management.load_models_gpu([self.vision_patcher])

    def load_planner(self) -> None:
        comfy.model_management.load_models_gpu(
            [self.language_patcher, self.aux_patcher]
        )


def _load_assign(
    module: nn.Module, state_dict: dict[str, torch.Tensor], label: str
) -> None:
    result = module.load_state_dict(state_dict, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{label} checkpoint mismatch: missing={result.missing_keys[:12]}, unexpected={result.unexpected_keys[:12]}"
        )


def _component_state(root: Path, component: str) -> dict[str, torch.Tensor]:
    state_dict = load_checkpoint_state_dict(component_checkpoint(root, component))
    if "scaled_fp8" in state_dict:
        state_dict, _ = comfy.utils.convert_old_quants(state_dict)
    return state_dict


def _uses_native_quant(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.endswith(".comfy_quant") for key in state_dict)


def _module_init_device(
    state_dict: dict[str, torch.Tensor], offload_device: torch.device
) -> torch.device | str:
    """Quant hooks must materialize on storage, while assign-loading BF16 can start on meta."""

    return offload_device if _uses_native_quant(state_dict) else "meta"


def _qwen_config(config_path: Path) -> dict[str, object]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in dataclasses.fields(Qwen25_7BVLI_Config)}
    return {key: value for key, value in payload.items() if key in allowed}


def load_planner_runtime(
    root: str | Path, *, dtype: torch.dtype = torch.bfloat16
) -> BerniniV2PlannerRuntime:
    """Load native Qwen/VIT planner weights from a streamed repack directory."""

    root = Path(root).resolve()
    config_path = root / "mllm" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} is missing; download the complete Bernini v2 native model package"
        )

    config = _qwen_config(config_path)
    config_obj = Qwen25_7BVLI_Config(**config)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    rope_dims = raw_config.get("rope_scaling", {}).get("mrope_section")
    if rope_dims is not None and list(rope_dims) != list(config_obj.rope_dims):
        raise ValueError(
            f"checkpoint mRoPE sections {rope_dims} do not match ComfyUI Qwen config {config_obj.rope_dims}"
        )
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.text_encoder_offload_device()
    mllm_state = _component_state(root, "mllm")
    mllm_quantized = _uses_native_quant(mllm_state)
    operations = (
        comfy.ops.mixed_precision_ops({}, dtype)
        if mllm_quantized
        else comfy.ops.manual_cast
    )
    init_device = _module_init_device(mllm_state, offload_device)
    language_model = Llama2_(
        config_obj, device=init_device, dtype=dtype, ops=operations
    )
    vision_model = Qwen2VLVisionTransformer(
        hidden_size=1280,
        output_hidden_size=config_obj.hidden_size,
        intermediate_size=3420,
        num_heads=16,
        num_layers=32,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=112,
        device=init_device,
        dtype=dtype,
        ops=operations,
    )
    language_state = {
        key.removeprefix("model."): value
        for key, value in mllm_state.items()
        if key.startswith("model.")
    }
    vision_state = {
        key.removeprefix("visual."): value
        for key, value in mllm_state.items()
        if key.startswith("visual.")
    }
    _load_assign(language_model, language_state, "Qwen language model")
    _load_assign(vision_model, vision_state, "Qwen vision model")
    del language_state, vision_state, mllm_state

    aux_state = {}
    connector_state = _component_state(root, "connector")
    aux_state.update(
        {f"connector.{key}": value for key, value in connector_state.items()}
    )
    decoder_state = _component_state(root, "vit_decoder")
    aux_state.update(
        {f"vit_decoder.{key}": value for key, value in decoder_state.items()}
    )
    aux_state.update(_component_state(root, "mask_tokens"))
    aux_quantized = _uses_native_quant(aux_state)
    aux_operations = (
        comfy.ops.mixed_precision_ops({}, dtype)
        if aux_quantized
        else comfy.ops.manual_cast
    )
    aux_init_device = _module_init_device(aux_state, offload_device)
    aux = PlannerAux(device=aux_init_device, dtype=dtype, operations=aux_operations)
    _load_assign(aux, aux_state, "Bernini planner auxiliary")
    del connector_state, decoder_state, aux_state

    language_patcher = CoreModelPatcher(
        language_model, load_device=load_device, offload_device=offload_device
    )
    vision_patcher = CoreModelPatcher(
        vision_model, load_device=load_device, offload_device=offload_device
    )
    aux_patcher = CoreModelPatcher(
        aux, load_device=load_device, offload_device=offload_device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        root / "mllm", local_files_only=True, use_fast=True
    )
    return BerniniV2PlannerRuntime(
        language_model=language_model,
        vision_model=vision_model,
        aux=aux,
        language_patcher=language_patcher,
        vision_patcher=vision_patcher,
        aux_patcher=aux_patcher,
        tokenizer=tokenizer,
        dtype=dtype,
        load_device=load_device,
    )


def load_wan_t5(root: str | Path, *, dtype: torch.dtype = torch.bfloat16):
    """Load the official UMT5-XXL weights as a standard Comfy ``CLIP``."""

    root = Path(root).resolve()
    state_dict = _component_state(root, "t5_text_encoder")
    tokenizer_path = root / "t5_tokenizer" / "spiece.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    state_dict["spiece_model"] = torch.frombuffer(
        bytearray(tokenizer_path.read_bytes()), dtype=torch.uint8
    )
    return comfy.sd.load_text_encoder_state_dicts(
        [state_dict],
        clip_type=comfy.sd.CLIPType.WAN,
        model_options={"dtype": dtype},
        disable_dynamic=False,
    )
