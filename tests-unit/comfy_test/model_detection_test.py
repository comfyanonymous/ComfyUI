import torch
import pytest
from unittest.mock import patch

from comfy.model_detection import detect_unet_config, model_config_from_unet_config
import comfy.supported_models


def _make_longcat_diffusers_sd():
    """Minimal Diffusers-format state dict that triggers the LongCat-Image detection path."""
    sd = {}
    H = 3072  # hidden_size (matches real LongCat-Image)
    C_IN = 16
    C_CTX = 3584  # context_in_dim that distinguishes LongCat from standard Flux (4096)

    sd["x_embedder.weight"] = torch.empty(H, C_IN * 4)
    sd["x_embedder.bias"] = torch.empty(H)
    sd["context_embedder.weight"] = torch.empty(H, C_CTX)
    sd["context_embedder.bias"] = torch.empty(H)

    sd["time_embed.timestep_embedder.linear_1.weight"] = torch.empty(H, 256)
    sd["time_embed.timestep_embedder.linear_1.bias"] = torch.empty(H)
    sd["time_embed.timestep_embedder.linear_2.weight"] = torch.empty(H, H)
    sd["time_embed.timestep_embedder.linear_2.bias"] = torch.empty(H)

    sd["norm_out.linear.weight"] = torch.empty(2 * H, H)
    sd["norm_out.linear.bias"] = torch.empty(2 * H)
    sd["proj_out.weight"] = torch.empty(C_IN * 4, H)
    sd["proj_out.bias"] = torch.empty(C_IN * 4)

    # Need enough transformer_blocks and single_transformer_blocks for count_blocks
    # and for the required_keys check (single_transformer_blocks.10.*)
    for i in range(19):
        sd[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.empty(H, H)
        sd[f"transformer_blocks.{i}.norm1.linear.weight"] = torch.empty(H)
    for i in range(38):
        sd[f"single_transformer_blocks.{i}.attn.to_q.weight"] = torch.empty(H, H)
        sd[f"single_transformer_blocks.{i}.norm.linear.weight"] = torch.empty(H)

    return sd


def _make_flux_schnell_comfyui_sd():
    """Minimal ComfyUI-format state dict that triggers the standard Flux detection path."""
    sd = {}
    H = 3072
    C_IN = 16

    sd["img_in.weight"] = torch.empty(H, C_IN * 4)
    sd["img_in.bias"] = torch.empty(H)
    sd["txt_in.weight"] = torch.empty(H, 4096)
    sd["txt_in.bias"] = torch.empty(H)

    sd["double_blocks.0.img_attn.norm.key_norm.weight"] = torch.empty(128)
    sd["double_blocks.0.img_attn.qkv.weight"] = torch.empty(3 * H, H)
    sd["double_blocks.0.img_mod.lin.weight"] = torch.empty(H, H)

    for i in range(19):
        sd[f"double_blocks.{i}.img_attn.norm.key_norm.weight"] = torch.empty(128)
    for i in range(38):
        sd[f"single_blocks.{i}.modulation.lin.weight"] = torch.empty(H, H)

    return sd


class TestModelDetectionSpecificity:
    """Verify that model_config_from_unet_config picks the most specific match."""

    def test_longcat_wins_regardless_of_list_order(self):
        """Specificity logic must pick LongCatImage even when FluxSchnell appears first."""
        sd = _make_longcat_diffusers_sd()
        unet_config = detect_unet_config(sd, "")
        original_models = comfy.supported_models.models

        longcat_cls = comfy.supported_models.LongCatImage
        schnell_cls = comfy.supported_models.FluxSchnell

        # Order A: FluxSchnell before LongCatImage
        order_a = [schnell_cls, longcat_cls]
        # Order B: LongCatImage before FluxSchnell
        order_b = [longcat_cls, schnell_cls]

        for label, order in [("schnell-first", order_a), ("longcat-first", order_b)]:
            with patch.object(comfy.supported_models, "models", order):
                result = model_config_from_unet_config(unet_config, sd)
                assert result is not None, f"No match with order {label}"
                assert type(result).__name__ == "LongCatImage", (
                    f"Expected LongCatImage with order {label}, got {type(result).__name__}"
                )

    def test_longcat_diffusers_detected_as_longcat(self):
        sd = _make_longcat_diffusers_sd()
        unet_config = detect_unet_config(sd, "")
        assert unet_config is not None
        assert unet_config["image_model"] == "flux"
        assert unet_config["context_in_dim"] == 3584
        assert unet_config["txt_ids_dims"] == [1, 2]

        model_config = model_config_from_unet_config(unet_config, sd)
        assert model_config is not None
        assert type(model_config).__name__ == "LongCatImage"

    def test_longcat_process_unet_state_dict_converts_keys(self):
        sd = _make_longcat_diffusers_sd()
        unet_config = detect_unet_config(sd, "")
        model_config = model_config_from_unet_config(unet_config, sd)

        converted = model_config.process_unet_state_dict(dict(sd))
        assert "img_in.weight" in converted
        assert "img_in.bias" in converted
        assert "txt_in.weight" in converted
        assert "x_embedder.weight" not in converted
        assert "context_embedder.weight" not in converted

    def test_flux_schnell_comfyui_detected_as_flux_schnell(self):
        sd = _make_flux_schnell_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        assert unet_config is not None
        assert unet_config["image_model"] == "flux"
        assert unet_config["context_in_dim"] == 4096
        assert unet_config["txt_ids_dims"] == []

        model_config = model_config_from_unet_config(unet_config, sd)
        assert model_config is not None
        assert type(model_config).__name__ == "FluxSchnell"
