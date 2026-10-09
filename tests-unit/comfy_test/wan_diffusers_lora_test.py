"""WAN LoRAs saved in Diffusers/PEFT naming map onto ComfyUI's native WAN weights."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.wan.model  # noqa: E402
import comfy.lora  # noqa: E402
import comfy.model_base  # noqa: E402
import comfy.supported_models  # noqa: E402

# Diffusers name -> native WAN name, as in diffusers' _convert_non_diffusers_wan_lora_to_diffusers.
BLOCK_NAMES = {
    "attn1.to_q": "self_attn.q", "attn1.to_k": "self_attn.k", "attn1.to_v": "self_attn.v", "attn1.to_out.0": "self_attn.o",
    "attn2.to_q": "cross_attn.q", "attn2.to_k": "cross_attn.k", "attn2.to_v": "cross_attn.v", "attn2.to_out.0": "cross_attn.o",
    "attn2.add_k_proj": "cross_attn.k_img", "attn2.add_v_proj": "cross_attn.v_img",
    "ffn.net.0.proj": "ffn.0", "ffn.net.2": "ffn.2",
}
OTHER_NAMES = {
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1",
    "condition_embedder.image_embedder.ff.net.0.proj": "img_emb.proj.1",
    "condition_embedder.image_embedder.ff.net.2": "img_emb.proj.3",
    "proj_out": "head.head",
}
NUM_LAYERS = 2
UNET_CONFIG = {"image_model": "wan2.1", "model_type": "i2v", "dim": 16, "ffn_dim": 32, "num_heads": 2,
               "num_layers": NUM_LAYERS, "text_dim": 8, "freq_dim": 8, "in_dim": 4, "out_dim": 4}


def expected_targets():
    targets = {name: native for name, native in OTHER_NAMES.items()}
    for i in range(NUM_LAYERS):
        for name, native in BLOCK_NAMES.items():
            targets["blocks.{}.{}".format(i, name)] = "blocks.{}.{}".format(i, native)
    return targets


def small_wan():
    return comfy.supported_models.WAN21_T2V(dict(UNET_CONFIG)).get_model({})


def diffusers_lora(model, prefix):
    sd = model.state_dict()
    lora = {}
    for name, native in expected_targets().items():
        out_dim, in_dim = sd["diffusion_model.{}.weight".format(native)].shape
        lora["{}{}.lora_A.weight".format(prefix, name)] = torch.ones(1, in_dim)
        lora["{}{}.lora_B.weight".format(prefix, name)] = torch.ones(out_dim, 1)
    return lora


@pytest.mark.parametrize("prefix", ["transformer.", ""])
def test_wan_diffusers_lora_patches_every_target(prefix):
    model = small_wan()
    assert isinstance(model, comfy.model_base.WAN21)
    lora = diffusers_lora(model, prefix)
    lora["{}blocks.0.attn1.to_q.alpha".format(prefix)] = torch.tensor(0.5)

    patches = comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(model, {}))

    expected = {"diffusion_model.{}.weight".format(native) for native in expected_targets().values()}
    assert set(patches) == expected

    target = "diffusion_model.blocks.0.self_attn.q.weight"
    weight = torch.zeros_like(model.state_dict()[target])
    updated = comfy.lora.calculate_weight([(1.0, patches[target], 1.0, None, None)], weight, target)
    assert torch.equal(updated, torch.full_like(weight, 0.5))  # alpha / rank = 0.5 / 1


def test_diffusers_wan_names_stay_wan_only():
    config = comfy.supported_models.WAN21_T2V(dict(UNET_CONFIG))
    other = comfy.model_base.BaseModel(config, unet_model=comfy.ldm.wan.model.WanModel)
    key_map = comfy.lora.model_lora_keys_unet(other, {})
    assert not any(k in key_map for k in expected_targets())
    assert not any("transformer.{}".format(k) in key_map for k in expected_targets())
