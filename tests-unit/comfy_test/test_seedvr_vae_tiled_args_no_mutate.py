import re
from pathlib import Path


def test_seedvr_vae_decode_uses_explicit_tiling_options_not_object_state():
    path = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"
    src = path.read_text(encoding="utf-8")
    assert not re.search(r"(?:self\.)?tiled_args\b", src), (
        "VideoAutoencoderKLWrapper.decode must not read or mutate tiled_args "
        f"object state. Source path: {path}"
    )
