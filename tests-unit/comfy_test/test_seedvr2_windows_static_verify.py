from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_seedvr2_windows_static_contract_tokens():
    nodes = _read("comfy_extras/nodes_seedvr.py")
    sd = _read("comfy/sd.py")
    vae = _read("comfy/ldm/seedvr/vae.py")

    required = [
        "SeedVR2Resize",
        "SeedVR2ResizeAdvanced",
        "SeedVR2PostProcessing",
        'io.Image.Input("decoded")',
        'io.Image.Input("original_image")',
        'io.Int.Input("upscaled_shorter_edge", min=2, force_input=True)',
        'io.Combo.Input("color_correction_method", options=["lab", "wavelet", "adain", "none"], default="lab")',
        "def _format_seedvr2_encoded_samples",
        "def decode(self, z, seedvr2_tiling=None)",
    ]
    for needle in required:
        if needle not in nodes + sd + vae:
            pytest.fail(f"missing required static token: {needle}")

    forbidden = ["original_image_video", "img_dims", "tiled_args"]
    for needle in forbidden:
        if needle in nodes + sd + vae:
            pytest.fail(f"forbidden hidden-state token remains: {needle}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
