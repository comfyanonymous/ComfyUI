"""MiniMax H3 pruned AdaLN LoRA helpers.

These helpers are model-specific and live in the MiniMax module instead of
``comfy/sd.py``.  ``sd.py`` only calls them from ``load_lora_for_models``.
"""

from __future__ import annotations

import logging

import torch


def set_patch_value(patch):
    if not isinstance(patch, tuple) or len(patch) != 2:
        return None
    if patch[0] != "set" or not isinstance(patch[1], tuple) or len(patch[1]) != 1:
        return None
    return patch[1][0]


def existing_set(model, key):
    for p in model.patches.get(key, []):
        value = set_patch_value(p[1])
        if value is not None:
            return value
    return None


def has_pruned_adaln(lora):
    return (
        "adaln_t_table" in lora
        or "diffusion_model.adaln_t_table" in lora
        or "adaln_t_table.set_weight" in lora
        or "diffusion_model.adaln_t_table.set_weight" in lora
        or any(
            ".adaln_proj.linear.weight" in k
            or ".adaln_proj.linear.bias" in k
            or ".adaln_proj.linear.weight.set_weight" in k
            or ".adaln_proj.linear.bias.set_weight" in k
            for k in lora
        )
    )


def has_legacy_dora(lora):
    """Detect the node package's legacy ``diff_b`` DoRA convention.

    ComfyUI's official ``diff_b`` handling is a 1-D bias diff.  A 2-D
    ``diff_b``, or a ``diff_b`` paired with ``lora_A`` on the same base, means
    the LoRA was produced with the legacy LyCORIS-style DoRA convention used by
    the MiniMax node package.  Loading it through the official LoraLoader
    would silently treat it as a bias diff.
    """
    for k in lora:
        if k.endswith(".lora_A.weight"):
            base = k[: -len(".lora_A.weight")]
        elif k.endswith(".lora_down.weight"):
            base = k[: -len(".lora_down.weight")]
        else:
            continue
        if base + ".diff_b" in lora:
            return True
    return False


def adaln_adapter_compatible(patch, target_shape):
    if isinstance(patch, tuple):
        if len(patch) == 2 and patch[0] == "diff":
            data = patch[1]
            if isinstance(data, tuple) and data:
                return data[0].shape == target_shape
        return True
    weights = patch.weights
    if not weights or len(weights) < 2:
        return False
    return (
        weights[0].shape[0] == target_shape[0]
        and weights[1].shape[-1] == target_shape[-1]
    )


def apply_merged_adaln_patches(patcher, merged):
    for key, patch in merged:
        existing = patcher.patches.get(key, [])
        preserved = [(1.0, patch, 1.0, None, None)]
        preserved.extend([
            p for p in existing
            if set_patch_value(p[1]) is None
        ])
        patcher.patches[key] = preserved


def merge_adaln_patches(model, loaded, model_sd, strength_model):
    """Merge stacked complete pruned AdaLN set patches at the loader boundary.

    Standard ``set`` patches replace the target weight, so loading two complete
    pruned LoRAs through separate LoraLoader nodes would otherwise overwrite the
    first LoRA's AdaLN projection.  This helper converts the incoming set patch
    into a delta relative to the original model weight and combines it with the
    existing full replacement:

        combined = current_full + strength * (new_full - base)

    The table itself is kept from the existing/base model.  If the incoming
    LoRA carries a different table, the merge still runs but is approximate.
    """
    current_table = None
    new_table = None
    for key in model_sd:
        if key.endswith("adaln_t_table"):
            current_table = existing_set(model, key)
            if current_table is None:
                current_table = model_sd[key]
            break
    for key in loaded:
        if key.endswith("adaln_t_table"):
            new_table = set_patch_value(loaded[key])
            break

    table_warned = False
    if current_table is not None and new_table is not None:
        if current_table.shape != new_table.shape:
            table_warned = True
        else:
            target_device = current_table.device
            new_table = new_table.to(target_device)
            if (current_table.float() - new_table.float()).abs().max().item() > 1e-3:
                table_warned = True
    if table_warned:
        logging.warning(
            "MiniMax H3 pruned LoRA merge: adaln_t_table differs between "
            "stacked LoRAs. Projection deltas are still merged, but results "
            "are approximate; use a shared fixed table or generate a combined "
            "complete pruned LoRA."
        )

    merged = []
    for key in list(loaded):
        if not (
            key.endswith("adaln_t_table")
            or key.endswith(".adaln_proj.linear.weight")
            or key.endswith(".adaln_proj.linear.bias")
        ):
            continue
        new_full = set_patch_value(loaded[key])
        if new_full is None:
            continue

        if key.endswith("adaln_t_table"):
            base = model_sd.get(key)
            if base is None:
                merged.append((key, ("set", (new_full,))))
            loaded.pop(key)
            continue

        base = model_sd.get(key)
        if base is None:
            continue
        current_full = existing_set(model, key)
        if current_full is None:
            current_full = base
        if current_full.shape != new_full.shape or current_full.shape != base.shape:
            loaded.pop(key)
            logging.warning(
                "MiniMax H3 pruned LoRA merge skipped for %s: shape mismatch "
                "current=%s new=%s base=%s",
                key,
                tuple(current_full.shape),
                tuple(new_full.shape),
                tuple(base.shape),
            )
            continue
        target_device = base.device
        current_full = current_full.to(target_device)
        new_full = new_full.to(target_device)
        combined = (
            current_full.float()
            + strength_model * (new_full.float() - base.float())
        ).to(base.dtype)
        merged.append((key, ("set", (combined,))))
        loaded.pop(key)

    return merged
