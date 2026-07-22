# Detection + loading for the Hunyuan3D 2.1 paint (hunyuan3d-paintpbr-v2-1) UNet.
# Follows core conventions: config is *detected* from the state_dict, dtype is chosen
# via comfy.model_management, weights are cast + loaded strict, and the model is
# wrapped in a comfy ModelPatcher for offload-aware execution.

import logging

import torch

import comfy.model_management
import comfy.ops
from comfy.model_patcher import ModelPatcher

from .unet import UNet2p5DConditionModel

# Fixed SD-2.1 head dim of the released paint checkpoint (config.json:
# attention_head_dim [5, 10, 20, 20] over block_out_channels [320, 640, 1280, 1280]).
_HEAD_DIM = 64


def detect_paint_config(state_dict, prefix=""):
    """Return a UNet2p5DConditionModel config dict if ``state_dict`` looks like the
    Hunyuan3D 2.1 paint UNet, else ``None``. Everything derivable is read from tensor
    shapes; only the (weight-invisible) attention head dim and group count are assumed.

    Raises ValueError (never KeyError) when the checkpoint matches the paint family
    but is missing keys the config derivation needs (truncated / renamed weights).
    """
    ck = lambda k: f"{prefix}{k}"  # noqa: E731
    conv_in = ck("unet.conv_in.weight")
    if conv_in not in state_dict:
        return None
    if ck("unet_dual.conv_in.weight") not in state_dict:
        return None
    if ck("unet.learned_text_clip_albedo") not in state_dict:
        return None

    def need(key):
        full = ck(key)
        if full not in state_dict:
            raise ValueError(
                f"checkpoint looks like a Hunyuan3D 2.1 paint UNet but is missing "
                f"'{full}'; the file appears truncated or its keys were renamed "
                f"(expected the hunyuan3d-paintpbr-v2-1 layout)")
        return state_dict[full]

    in_channels = state_dict[conv_in].shape[1]
    ref_in_channels = state_dict[ck("unet_dual.conv_in.weight")].shape[1]
    out_channels = need("unet.conv_out.weight").shape[0]

    # block_out_channels from each down block's resnet output width
    block_out_channels = []
    i = 0
    while ck(f"unet.down_blocks.{i}.resnets.0.conv2.weight") in state_dict:
        block_out_channels.append(state_dict[ck(f"unet.down_blocks.{i}.resnets.0.conv2.weight")].shape[0])
        i += 1

    # layers_per_block = number of resnets in the first down block
    layers_per_block = 0
    while ck(f"unet.down_blocks.0.resnets.{layers_per_block}.conv2.weight") in state_dict:
        layers_per_block += 1

    # transformer_layers_per_block
    transformer_layers = 0
    while ck(f"unet.down_blocks.0.attentions.0.transformer_blocks.{transformer_layers}.transformer.norm1.weight") in state_dict:
        transformer_layers += 1

    if not block_out_channels or layers_per_block == 0 or transformer_layers == 0:
        raise ValueError(
            "checkpoint looks like a Hunyuan3D 2.1 paint UNet but its down_blocks "
            "layout is incomplete; the file appears truncated (expected the "
            "hunyuan3d-paintpbr-v2-1 layout)")

    cross_attention_dim = need(
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.transformer.attn2.to_k.weight").shape[1]

    num_attention_heads = [max(1, c // _HEAD_DIM) for c in block_out_channels]

    # PBR settings from learned_text_clip_{token}
    pbr_setting = ["albedo"]
    if ck("unet.learned_text_clip_mr") in state_dict:
        pbr_setting.append("mr")
    pbr_token_channels = state_dict[ck("unet.learned_text_clip_albedo")].shape[0]

    use_dino = ck("unet.image_proj_model_dino.proj.weight") in state_dict
    dino_embeddings_dim = 1536
    if use_dino:
        dino_embeddings_dim = state_dict[ck("unet.image_proj_model_dino.proj.weight")].shape[1]

    return {
        "in_channels": in_channels,
        "ref_in_channels": ref_in_channels,
        "out_channels": out_channels,
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": layers_per_block,
        "cross_attention_dim": cross_attention_dim,
        "num_attention_heads": tuple(num_attention_heads),
        "transformer_layers_per_block": transformer_layers,
        "norm_num_groups": 32,
        "pbr_setting": tuple(pbr_setting),
        "pbr_token_channels": pbr_token_channels,
        "dino_embeddings_dim": dino_embeddings_dim,
        "use_dino": use_dino,
    }


def _describe_checkpoint_family(state_dict):
    """Best-effort guess at what an unrecognised checkpoint actually is, so loader
    errors say "this looks like X" instead of a bare key dump."""
    keys = list(state_dict.keys())

    def has(prefix):
        return any(k.startswith(prefix) for k in keys)

    if has("unet.conv_in") and not has("unet_dual."):
        return ("a single-stream 'unet.*' checkpoint without the paint model's "
                "dual-stream reference UNet ('unet_dual.*')")
    if has("model.diffusion_model."):
        return ("a ComfyUI/LDM diffusion checkpoint ('model.diffusion_model.*') - "
                "load it with the standard checkpoint/diffusion-model loader")
    if has("double_blocks.") or has("single_blocks.") or has("joint_blocks."):
        return "a DiT-style diffusion model (double/single/joint blocks)"
    if has("down_blocks.") and has("conv_in."):
        return ("a plain diffusers UNet (unprefixed 'down_blocks.*') without the "
                "paint model's 'unet.*'/'unet_dual.*' dual-stream layout")
    if has("decoder.") and has("encoder."):
        return "a VAE (encoder/decoder) checkpoint"
    sample = ", ".join(sorted(keys)[:3]) if keys else "no keys at all"
    return f"an unrecognised checkpoint (first keys: {sample})"


def load_paint_unet(state_dict, model_options={}):
    """Build a UNet2p5DConditionModel from a paint state_dict and wrap it in a
    ModelPatcher. Returns ``(patcher, config)``.

    Raises ValueError with a family diagnosis when the checkpoint is not a
    hunyuan3d-paintpbr-v2-1 paint UNet, or when it matches the family but is
    missing weights (truncated file)."""
    config = detect_paint_config(state_dict)
    if config is None:
        raise ValueError(
            f"this looks like {_describe_checkpoint_family(state_dict)}; expected a "
            f"Hunyuan3D 2.1 paint UNet (hunyuan3d-paintpbr-v2-1) with "
            f"'unet.*'/'unet_dual.*' keys and learned_text_clip_* embeddings")

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    parameters = sum(v.numel() for v in state_dict.values())

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    unet_dtype = model_options.get("dtype", None)
    if unet_dtype is None:
        unet_dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=supported_dtypes)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device, supported_dtypes)
    operations = model_options.get("custom_operations", None)
    if operations is None:
        operations = comfy.ops.pick_operations(unet_dtype, manual_cast_dtype)

    model = UNet2p5DConditionModel(
        dtype=unet_dtype, device=offload_device, operations=operations, **_model_kwargs(config))
    model.eval()

    cast_sd = {k: v.to(unet_dtype) for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cast_sd, strict=False)
    if missing:
        raise ValueError(
            f"Hunyuan3D 2.1 paint checkpoint is missing {len(missing)} weights "
            f"(e.g. {missing[:3]}); the file appears truncated or from a different "
            f"model revision than hunyuan3d-paintpbr-v2-1")
    if unexpected:
        logging.warning("Hunyuan3D paint: %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])

    model.manual_cast_dtype = manual_cast_dtype
    patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    return patcher, config


def _model_kwargs(config):
    return dict(
        in_channels=config["in_channels"], ref_in_channels=config["ref_in_channels"],
        out_channels=config["out_channels"], block_out_channels=config["block_out_channels"],
        layers_per_block=config["layers_per_block"], cross_attention_dim=config["cross_attention_dim"],
        num_attention_heads=config["num_attention_heads"],
        transformer_layers_per_block=config["transformer_layers_per_block"],
        norm_num_groups=config["norm_num_groups"], pbr_setting=config["pbr_setting"],
        pbr_token_channels=config["pbr_token_channels"], dino_embeddings_dim=config["dino_embeddings_dim"],
        use_dino=config["use_dino"])
