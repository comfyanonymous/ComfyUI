# TCFG: Tangential Damping Classifier-free Guidance - (arXiv: https://arxiv.org/abs/2503.18137)

import torch

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict


def score_tangential_damping(cond_score: torch.Tensor, uncond_score: torch.Tensor) -> torch.Tensor:
    """Drop tangential components from uncond score to align with cond score."""
    # (B, 1, ...)
    batch_num = cond_score.shape[0]
    cond_score_flat = cond_score.reshape(batch_num, 1, -1).float()
    uncond_score_flat = uncond_score.reshape(batch_num, 1, -1).float()

    # Score matrix A (B, 2, ...)
    score_matrix = torch.cat((uncond_score_flat, cond_score_flat), dim=1)
    try:
        _, _, Vh = torch.linalg.svd(score_matrix, full_matrices=False)
    except RuntimeError:
        # Fallback to CPU
        _, _, Vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)

    # Drop the tangential components
    v1 = Vh[:, 0:1, :].to(uncond_score_flat.device)  # (B, 1, ...)
    uncond_score_td = (uncond_score_flat @ v1.transpose(-2, -1)) * v1
    return uncond_score_td.reshape_as(uncond_score).to(uncond_score.dtype)


class TCFG(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"

    CATEGORY = "advanced/guidance"
    DESCRIPTION = "TCFG – Tangential Damping CFG (2503.18137)\n\nRefine the uncond (negative) to align with the cond (positive) for improving quality."

    def patch(self, model):
        m = model.clone()

        def tangential_damping_cfg(args):
            #  Assume [cond, uncond, ...]
            x = args["input"]
            conds_out = args["conds_out"]
            if len(conds_out) <= 1 or None in args["conds"][:2]:
                # Skip when either cond or uncond is None
                return conds_out
            cond_pred = conds_out[0]
            uncond_pred = conds_out[1]
            uncond_td = score_tangential_damping(x - cond_pred, x - uncond_pred)
            uncond_pred_td = x - uncond_td
            return [cond_pred, uncond_pred_td] + conds_out[2:]

        m.set_model_sampler_pre_cfg_function(tangential_damping_cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "TCFG": TCFG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TCFG": "Tangential Damping CFG",
}
