"""Native-side parity harness: tiny seeded model + hooked forward + golden IO.

Imports comfy, so it runs in the ComfyUI environment (unit tests, golden
regeneration, author-side reference compare) - never in the reference venv.
"""

import zlib

import torch

import comfy.ops
from comfy.ldm.hunyuan3d.paint.unet import UNet2p5DConditionModel

from . import bundle_format

# 2-block micro-config: smallest UNet2p5D that still exercises every attention
# mechanism (material, reference, multiview + PoseRoPE, DINO) and both a
# cross-attn and a plain down/up block.
TINY_CONFIG = dict(
    in_channels=12, ref_in_channels=4, out_channels=4,
    block_out_channels=(32, 64), layers_per_block=1, cross_attention_dim=32,
    num_attention_heads=(1, 2), transformer_layers_per_block=1, norm_num_groups=32,
    pbr_setting=("albedo", "mr"), pbr_token_channels=4, dino_embeddings_dim=16,
    use_dino=True,
)
TINY_WEIGHT_SEED = 1234
TINY_INPUT_SEED = 7
TINY_INPUT_ARGS = dict(seed=TINY_INPUT_SEED, batch=1, n_pbr=2, views=2, channels=4,
                       height=16, tokens=4, cross_dim=32, dino_tokens=5, dino_dim=16,
                       timestep=500)
# fp32 CPU across platforms/torch builds; the model is 2 blocks deep so accumulated
# conv/SDPA reassociation differences stay well under this.
TINY_ATOL = 2e-4
TINY_RTOL = 1e-3


def build_tiny_model(seed=TINY_WEIGHT_SEED, config=None):
    """Deterministic random-init tiny UNet2p5D. Each parameter is seeded from
    (seed, crc32(param name)) so the init is stable under parameter-registration
    reordering; only adding/removing/renaming parameters changes it."""
    model = UNet2p5DConditionModel(dtype=torch.float32, device="cpu",
                                   operations=comfy.ops.disable_weight_init,
                                   **(config or TINY_CONFIG))
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            g = torch.Generator().manual_seed(
                (int(seed) * 0x9E3779B1 + zlib.crc32(name.encode())) % (2 ** 63))
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05)
    model.eval()
    return model


def run_model(model, tensors, capture_blocks=False):
    """One denoise forward from bundle input tensors.

    Returns (noise_pred, activations) where activations is a dict of
    ``act/<module path>`` -> float32 tensor (empty unless capture_blocks).
    """
    acts = {}
    handles = []
    if capture_blocks:
        names = bundle_format.block_names(len(model.unet.down_blocks),
                                          len(model.unet.up_blocks))

        def make_hook(name):
            def hook(_module, _args, output):
                out = output[0] if isinstance(output, tuple) else output
                acts[f"act/{name}"] = out.detach().float()
            return hook

        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            out = model(
                tensors["input/sample"],
                tensors["input/timestep"],
                tensors["input/encoder_hidden_states"],
                dino_hidden_states=tensors.get("input/dino_hidden_states"),
                ref_latents=tensors["input/ref_latents"],
                embeds_normal=tensors["input/embeds_normal"],
                embeds_position=tensors["input/embeds_position"],
                position_maps=tensors["input/position_maps"],
                cache={},
            )
    finally:
        for h in handles:
            h.remove()
    return out, acts


def make_tiny_golden(path):
    """Regenerate the committed tiny golden bundle (inputs + expected output)."""
    model = build_tiny_model()
    tensors = bundle_format.make_parity_inputs(**TINY_INPUT_ARGS)
    out, _ = run_model(model, tensors)
    tensors["output/noise_pred"] = out
    bundle_format.save_bundle(path, tensors, {
        "source": "native-tiny",
        "weight_seed": str(TINY_WEIGHT_SEED),
        "input_args": TINY_INPUT_ARGS,
        "config": TINY_CONFIG,
        "tolerance": f"atol={TINY_ATOL},rtol={TINY_RTOL} (fp32 CPU)",
        "torch_version": torch.__version__,
    })
    return path
