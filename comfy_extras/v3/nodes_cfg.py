from __future__ import annotations

import torch

from comfy_api.v3 import io


def optimized_scale(positive, negative):
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star.reshape([positive.shape[0]] + [1] * (positive.ndim - 1))


class CFGNorm(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls) -> io.SchemaV3:
        return io.SchemaV3(
            node_id="CFGNorm_V3",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=1.0, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output("patched_model", display_name="patched_model")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, strength) -> io.NodeOutput:
        m = model.clone()

        def cfg_norm(args):
            cond_p = args['cond_denoised']
            pred_text_ = args["denoised"]

            norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
            norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
            scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
            return pred_text_ * scale * strength

        m.set_model_sampler_post_cfg_function(cfg_norm)
        return io.NodeOutput(m)


class CFGZeroStar(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls) -> io.SchemaV3:
        return io.SchemaV3(
            node_id="CFGZeroStar_V3",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[io.Model.Output("patched_model", display_name="patched_model")],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()

        def cfg_zero_star(args):
            guidance_scale = args['cond_scale']
            x = args['input']
            cond_p = args['cond_denoised']
            uncond_p = args['uncond_denoised']
            out = args["denoised"]
            alpha = optimized_scale(x - cond_p, x - uncond_p)

            return out + uncond_p * (alpha - 1.0) + guidance_scale * uncond_p * (1.0 - alpha)

        m.set_model_sampler_post_cfg_function(cfg_zero_star)
        return io.NodeOutput(m)


NODES_LIST = [
    CFGNorm,
    CFGZeroStar,
]
