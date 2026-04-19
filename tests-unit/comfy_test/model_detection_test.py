import torch

from comfy.model_detection import detect_unet_config, model_config_from_unet_config
import comfy.supported_models


def _make_longcat_comfyui_sd():
    """Minimal ComfyUI-format state dict for pre-converted LongCat-Image weights."""
    sd = {}
    H = 32  # Reduce hidden state dimension to reduce memory usage
    C_IN = 16
    C_CTX = 3584

    sd["img_in.weight"] = torch.empty(H, C_IN * 4)
    sd["img_in.bias"] = torch.empty(H)
    sd["txt_in.weight"] = torch.empty(H, C_CTX)
    sd["txt_in.bias"] = torch.empty(H)

    sd["time_in.in_layer.weight"] = torch.empty(H, 256)
    sd["time_in.in_layer.bias"] = torch.empty(H)
    sd["time_in.out_layer.weight"] = torch.empty(H, H)
    sd["time_in.out_layer.bias"] = torch.empty(H)

    sd["final_layer.adaLN_modulation.1.weight"] = torch.empty(2 * H, H)
    sd["final_layer.adaLN_modulation.1.bias"] = torch.empty(2 * H)
    sd["final_layer.linear.weight"] = torch.empty(C_IN * 4, H)
    sd["final_layer.linear.bias"] = torch.empty(C_IN * 4)

    for i in range(19):
        sd[f"double_blocks.{i}.img_attn.norm.key_norm.weight"] = torch.empty(128)
        sd[f"double_blocks.{i}.img_attn.qkv.weight"] = torch.empty(3 * H, H)
        sd[f"double_blocks.{i}.img_mod.lin.weight"] = torch.empty(H, H)
    for i in range(38):
        sd[f"single_blocks.{i}.modulation.lin.weight"] = torch.empty(H, H)

    return sd


def _make_flux_schnell_comfyui_sd():
    """Minimal ComfyUI-format state dict for standard Flux Schnell."""
    sd = {}
    H = 32  # Reduce hidden state dimension to reduce memory usage
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


class TestModelDetection:
    """Verify that first-match model detection selects the correct model
    based on list ordering and unet_config specificity."""

    def test_longcat_before_schnell_in_models_list(self):
        """LongCatImage must appear before FluxSchnell in the models list."""
        models = comfy.supported_models.models
        longcat_idx = next(i for i, m in enumerate(models) if m.__name__ == "LongCatImage")
        schnell_idx = next(i for i, m in enumerate(models) if m.__name__ == "FluxSchnell")
        assert longcat_idx < schnell_idx, (
            f"LongCatImage (index {longcat_idx}) must come before "
            f"FluxSchnell (index {schnell_idx}) in the models list"
        )

    def test_longcat_comfyui_detected_as_longcat(self):
        sd = _make_longcat_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        assert unet_config is not None
        assert unet_config["image_model"] == "flux"
        assert unet_config["context_in_dim"] == 3584
        assert unet_config["vec_in_dim"] is None
        assert unet_config["guidance_embed"] is False
        assert unet_config["txt_ids_dims"] == [1, 2]

        model_config = model_config_from_unet_config(unet_config, sd)
        assert model_config is not None
        assert type(model_config).__name__ == "LongCatImage"

    def test_longcat_comfyui_keys_pass_through_unchanged(self):
        """Pre-converted weights should not be transformed by process_unet_state_dict."""
        sd = _make_longcat_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        model_config = model_config_from_unet_config(unet_config, sd)

        processed = model_config.process_unet_state_dict(dict(sd))
        assert "img_in.weight" in processed
        assert "txt_in.weight" in processed
        assert "time_in.in_layer.weight" in processed
        assert "final_layer.linear.weight" in processed

    def test_nucleus_diffusers_expert_weights_stay_packed_for_grouped_mm(self):
        model_config = comfy.supported_models.NucleusImage({"image_model": "nucleus_image"})
        gate_up = torch.arange(2 * 3 * 4, dtype=torch.bfloat16).reshape(2, 3, 4)
        down = torch.arange(2 * 5 * 3, dtype=torch.bfloat16).reshape(2, 5, 3)
        sd = {
            "img_in.weight": torch.empty(2048, 64),
            "transformer_blocks.3.img_mlp.experts.gate_up_proj": gate_up,
            "transformer_blocks.3.img_mlp.experts.down_proj": down,
        }

        processed = model_config.process_unet_state_dict(dict(sd))

        assert processed["transformer_blocks.3.img_mlp.experts.gate_up_proj"] is gate_up
        assert processed["transformer_blocks.3.img_mlp.experts.down_proj"] is down

    def test_nucleus_swiglu_experts_loads_packed_weights(self):
        from comfy.ldm.nucleus.model import SwiGLUExperts

        experts = SwiGLUExperts(
            hidden_size=2,
            moe_intermediate_dim=1,
            num_experts=2,
            use_grouped_mm=False,
            operations=torch.nn,
        )
        gate_up = torch.tensor(
            [
                [[1.0, 0.5], [0.0, 1.0]],
                [[0.0, -1.0], [1.0, 0.25]],
            ]
        )
        down = torch.tensor(
            [
                [[2.0, -1.0]],
                [[-0.5, 1.5]],
            ]
        )

        experts.load_state_dict({"gate_up_proj": gate_up, "down_proj": down})
        x = torch.tensor([[2.0, 3.0], [1.0, -2.0], [4.0, 0.5]])
        num_tokens_per_expert = torch.tensor([2, 1], dtype=torch.long)

        out = experts(x, num_tokens_per_expert)
        expected_parts = []
        offset = 0
        for expert_idx, count in enumerate(num_tokens_per_expert.tolist()):
            x_expert = x[offset : offset + count]
            offset += count
            gate, up = (x_expert @ gate_up[expert_idx]).chunk(2, dim=-1)
            expected_parts.append((torch.nn.functional.silu(gate) * up) @ down[expert_idx])
        expected = torch.cat(expected_parts, dim=0)

        assert torch.allclose(out, expected)
        assert hasattr(experts, "comfy_cast_weights")
        assert experts.comfy_cast_weights is True
        assert hasattr(experts, "weight")
        assert hasattr(experts, "bias")
        assert not hasattr(experts, "gate_up_proj")
        assert not hasattr(experts, "down_proj")
        assert torch.equal(experts.state_dict()["weight"], gate_up)
        assert torch.equal(experts.state_dict()["bias"], down)

    def test_nucleus_swiglu_experts_loads_packed_quantized_weights(self):
        import json

        from comfy.ldm.nucleus.model import SwiGLUExperts
        from comfy.quant_ops import QuantizedTensor

        experts = SwiGLUExperts(
            hidden_size=2,
            moe_intermediate_dim=1,
            num_experts=2,
            use_grouped_mm=False,
            operations=torch.nn,
            dtype=torch.bfloat16,
        )
        gate_up = QuantizedTensor.from_float(
            torch.tensor(
                [
                    [[1.0, 0.5], [0.0, 1.0]],
                    [[0.0, -1.0], [1.0, 0.25]],
                ],
                dtype=torch.bfloat16,
            ),
            "TensorCoreFP8E4M3Layout",
            scale="recalculate",
        ).state_dict("gate_up_proj")
        down = QuantizedTensor.from_float(
            torch.tensor(
                [
                    [[2.0, -1.0]],
                    [[-0.5, 1.5]],
                ],
                dtype=torch.bfloat16,
            ),
            "TensorCoreFP8E4M3Layout",
            scale="recalculate",
        ).state_dict("down_proj")
        state_dict = {
            **gate_up,
            **down,
            "comfy_quant": torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")), dtype=torch.uint8),
        }

        experts.load_state_dict(state_dict)

        assert isinstance(experts.weight, QuantizedTensor)
        assert isinstance(experts.bias, QuantizedTensor)
        assert experts.weight.shape == (2, 2, 2)
        assert experts.bias.shape == (2, 1, 2)
        assert experts.weight.dtype == torch.bfloat16
        assert experts.bias.dtype == torch.bfloat16

    def test_nucleus_split_expert_weights_still_load_for_quantized_files(self):
        from comfy.ldm.nucleus.model import SwiGLUExperts

        experts = SwiGLUExperts(
            hidden_size=2,
            moe_intermediate_dim=1,
            num_experts=2,
            use_grouped_mm=True,
            operations=torch.nn,
        )
        split_state = {
            "gate_up_projs.0.weight": torch.tensor([[1.0, 0.0], [0.5, 1.0]]),
            "gate_up_projs.1.weight": torch.tensor([[0.0, 1.0], [-1.0, 0.25]]),
            "down_projs.0.weight": torch.tensor([[2.0], [-1.0]]),
            "down_projs.1.weight": torch.tensor([[-0.5], [1.5]]),
        }

        experts.load_state_dict(split_state)
        x = torch.tensor([[2.0, 3.0], [1.0, -2.0], [4.0, 0.5]])
        out = experts(x, torch.tensor([2, 1], dtype=torch.long))

        assert out.shape == x.shape
        assert not hasattr(experts, "comfy_cast_weights")
        assert not hasattr(experts, "gate_up_proj")
        assert not hasattr(experts, "weight")
        assert torch.equal(
            experts.gate_up_projs[0].weight,
            split_state["gate_up_projs.0.weight"],
        )

    def test_nucleus_dense_swiglu_uses_diffusers_chunk_order(self):
        from comfy.ldm.nucleus.model import FeedForward

        ff = FeedForward(dim=2, dim_out=1, inner_dim=2, operations=torch.nn)
        with torch.no_grad():
            ff.net[0].proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.5, 0.0],
                        [0.0, -0.5],
                    ]
                )
            )
            ff.net[2].weight.copy_(torch.tensor([[1.0, 1.0]]))

        x = torch.tensor([[[2.0, 4.0]]])
        expected = 2.0 * torch.nn.functional.silu(torch.tensor(1.0)) + 4.0 * torch.nn.functional.silu(torch.tensor(-2.0))

        assert torch.allclose(ff(x), expected.reshape(1, 1, 1))

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
