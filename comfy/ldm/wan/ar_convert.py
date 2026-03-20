"""
State dict conversion for Causal Forcing checkpoints.

Handles three checkpoint layouts:
  1. Training checkpoint with top-level generator_ema / generator keys
  2. Already-flattened state dict with model.* prefixes
  3. Already-converted ComfyUI state dict (bare model keys)

Strips prefixes so the result matches the standard Wan 2.1 / CausalWanModel key layout
(e.g. blocks.0.self_attn.q.weight, head.modulation, etc.)
"""

import logging

log = logging.getLogger(__name__)

PREFIXES_TO_STRIP = ["model._fsdp_wrapped_module.", "model."]

_MODEL_KEY_PREFIXES = (
    "blocks.", "head.", "patch_embedding.", "text_embedding.",
    "time_embedding.", "time_projection.", "img_emb.", "rope_embedder.",
)


def extract_state_dict(state_dict: dict, use_ema: bool = True) -> dict:
    """
    Extract and clean a Causal Forcing state dict from a training checkpoint.

    Returns a state dict with keys matching the CausalWanModel / WanModel layout.
    """
    # Case 3: already converted -- keys are bare model keys
    if "head.modulation" in state_dict and "blocks.0.self_attn.q.weight" in state_dict:
        return state_dict

    # Case 1: training checkpoint with wrapper key
    raw_sd = None
    order = ["generator_ema", "generator"] if use_ema else ["generator", "generator_ema"]
    for wrapper_key in order:
        if wrapper_key in state_dict:
            raw_sd = state_dict[wrapper_key]
            log.info("Causal Forcing: extracted '%s' with %d keys", wrapper_key, len(raw_sd))
            break

    # Case 2: flat dict with model.* prefixes
    if raw_sd is None:
        if any(k.startswith("model.") for k in state_dict):
            raw_sd = state_dict
        else:
            raise KeyError(
                f"Cannot detect Causal Forcing checkpoint layout. "
                f"Top-level keys: {list(state_dict.keys())[:20]}"
            )

    out_sd = {}
    for k, v in raw_sd.items():
        new_k = k
        for prefix in PREFIXES_TO_STRIP:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        else:
            if not new_k.startswith(_MODEL_KEY_PREFIXES):
                log.debug("Causal Forcing: skipping non-model key: %s", k)
                continue
        out_sd[new_k] = v

    if "head.modulation" not in out_sd:
        raise ValueError("Conversion failed: 'head.modulation' not found in output keys")

    return out_sd
