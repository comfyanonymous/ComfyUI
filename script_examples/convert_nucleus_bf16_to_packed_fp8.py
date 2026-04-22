import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comfy.quant_ops import QuantizedTensor


LAYOUT = "TensorCoreFP8E4M3Layout"
QUANT_CONFIG = {"format": "float8_e4m3fn"}


def is_packed_expert(key, tensor):
    return (
        key.endswith(".img_mlp.experts.gate_up_proj")
        or key.endswith(".img_mlp.experts.down_proj")
    ) and tensor.ndim == 3


def is_split_expert(key):
    return ".img_mlp.experts.gate_up_projs." in key or ".img_mlp.experts.down_projs." in key


def is_linear_weight(key, tensor):
    return key.endswith(".weight") and tensor.ndim >= 2


def module_key_for_weight(key):
    return key[:-len(".weight")]


def module_key_for_packed_expert(key):
    return key.rsplit(".", 1)[0]


def quant_config_tensor():
    return torch.tensor(list(json.dumps(QUANT_CONFIG).encode("utf-8")), dtype=torch.uint8)


def quantize_tensor(key, tensor):
    quantized = QuantizedTensor.from_float(tensor, LAYOUT, scale="recalculate")
    return quantized.state_dict(key)


def convert(input_path, output_path, overwrite=False):
    input_path = Path(input_path)
    output_path = Path(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if not input_path.exists():
        raise FileNotFoundError(f"input checkpoint does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output checkpoint already exists: {output_path}")
    if tmp_path.exists():
        tmp_path.unlink()

    start = time.time()
    out = {}
    packed_modules = set()
    stats = {
        "normal_quantized": 0,
        "packed_quantized": 0,
        "kept": 0,
        "split_expert_keys": 0,
    }

    with safe_open(input_path, framework="pt", device="cpu") as src:
        keys = list(src.keys())
        total = len(keys)
        print(f"source={input_path} keys={total}", flush=True)

        for index, key in enumerate(keys, 1):
            tensor = src.get_tensor(key)

            if is_split_expert(key):
                stats["split_expert_keys"] += 1

            if is_packed_expert(key, tensor):
                out.update(quantize_tensor(key, tensor))
                packed_modules.add(module_key_for_packed_expert(key))
                stats["packed_quantized"] += 1
            elif is_linear_weight(key, tensor):
                out.update(quantize_tensor(key, tensor))
                out[f"{module_key_for_weight(key)}.comfy_quant"] = quant_config_tensor()
                stats["normal_quantized"] += 1
            else:
                out[key] = tensor
                stats["kept"] += 1

            del tensor
            if index % 25 == 0 or index == total:
                elapsed = time.time() - start
                print(
                    "processed "
                    f"{index}/{total} "
                    f"normal_quantized={stats['normal_quantized']} "
                    f"packed_quantized={stats['packed_quantized']} "
                    f"kept={stats['kept']} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                gc.collect()

    if stats["packed_quantized"] == 0:
        raise RuntimeError("no packed Nucleus expert tensors were found in the input checkpoint")
    if stats["split_expert_keys"] > 0:
        raise RuntimeError(
            "input checkpoint already contains split Nucleus expert tensors; "
            "use the BF16/BF16-packed source checkpoint instead"
        )

    for module_key in packed_modules:
        out[f"{module_key}.comfy_quant"] = quant_config_tensor()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"saving {len(out)} tensors to {tmp_path}", flush=True)
    save_file(out, str(tmp_path), metadata={"format": "pt"})
    os.replace(tmp_path, output_path)

    stats["packed_modules"] = len(packed_modules)
    stats["seconds"] = round(time.time() - start, 1)
    stats["output_gib"] = round(output_path.stat().st_size / (1024 ** 3), 2)
    print(f"wrote {output_path}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a BF16 Nucleus-Image diffusion checkpoint to scaled FP8 "
            "while preserving packed MoE expert tensors for grouped-mm."
        )
    )
    parser.add_argument("input", help="BF16 Nucleus-Image diffusion checkpoint")
    parser.add_argument("output", help="output FP8 safetensors path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    convert(args.input, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
