import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.lora  # noqa: E402
import comfy.sd  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402

DIM, RANK, ALPHA = 16, 4, 2.0


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.qkv = torch.nn.Linear(DIM, 3 * DIM)
        self.diffusion_model.o = torch.nn.Linear(DIM, DIM)


def _key_map(model, key_map):
    # what model_lora_keys_unet maps for checkpoints with a fused qkv weight (MOSS-Image, Flux3, Flux diffusers LoRAs)
    for i, proj in enumerate("qkv"):
        key_map[f"diffusion_model.{proj}"] = ("diffusion_model.qkv.weight", (0, i * DIM, DIM))
    key_map["diffusion_model.o"] = "diffusion_model.o.weight"
    return key_map


def _lora():
    lora = {}
    for proj in "qkvo":
        lora[f"diffusion_model.{proj}.lora_down.weight"] = torch.randn(RANK, DIM)
        lora[f"diffusion_model.{proj}.lora_up.weight"] = torch.randn(DIM, RANK)
        lora[f"diffusion_model.{proj}.alpha"] = torch.tensor(ALPHA)
    return lora


def _delta(lora, proj, strength):
    return strength * ALPHA / RANK * lora[f"diffusion_model.{proj}.lora_up.weight"] @ lora[f"diffusion_model.{proj}.lora_down.weight"]


def test_bypass_lora_applies_adapters_on_fused_weight_slices(monkeypatch):
    torch.manual_seed(0)
    model = _Model()
    lora = _lora()
    strength = 0.7
    monkeypatch.setattr(comfy.lora, "model_lora_keys_unet", _key_map)
    qkv, o = model.diffusion_model.qkv, model.diffusion_model.o
    x = torch.randn(2, 5, DIM)
    base_qkv, base_o = qkv(x), o(x)
    expected_qkv = torch.nn.functional.linear(x, qkv.weight + torch.cat([_delta(lora, p, strength) for p in "qkv"]), qkv.bias)
    expected_o = torch.nn.functional.linear(x, o.weight + _delta(lora, "o", strength), o.bias)

    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    bypassed = comfy.sd.load_bypass_lora_for_models(patcher, None, lora, strength, 0)[0]
    for _ in range(2):  # inject/eject twice: the chained hooks on qkv must unwind cleanly
        bypassed.inject_model()
        try:
            torch.testing.assert_close(qkv(x), expected_qkv)
            torch.testing.assert_close(o(x), expected_o)
        finally:
            bypassed.eject_model()
        assert torch.equal(qkv(x), base_qkv)
        assert torch.equal(o(x), base_o)
