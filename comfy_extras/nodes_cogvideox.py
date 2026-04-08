import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override

class SparkVSRConditioning(io.ComfyNode):
    """Conditioning node for SparkVSR video super-resolution.

    Encodes LQ video and optional HR reference frames through the VAE,
    builds the concat conditioning for the CogVideoX I2V model.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SparkVSRConditioning",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("lq_video"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("length", default=49, min=1, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("batch_size", default=1, min=1, max=64),
                io.Image.Input("ref_frames", optional=True),
                io.Combo.Input("ref_mode", options=["auto", "manual"], default="auto", optional=True),
                io.String.Input("ref_indices", default="", optional=True),
                io.Float.Input("ref_guidance_scale", default=1.0, min=0.0, max=10.0, step=0.1, optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, lq_video, width, height, length,
                batch_size, ref_frames=None, ref_mode="auto", ref_indices="",
                ref_guidance_scale=1.0) -> io.NodeOutput:

        temporal_compression = 4
        latent_t = ((length - 1) // temporal_compression) + 1
        latent_h = height // 8
        latent_w = width // 8

        # Base latent (noise will be added by KSampler)
        latent = torch.zeros(
            [batch_size, 16, latent_t, latent_h, latent_w],
            device=comfy.model_management.intermediate_device()
        )

        # Encode LQ video → this becomes the base latent (KSampler adds noise to this)
        lq = lq_video[:length]
        lq = comfy.utils.common_upscale(lq.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        lq_latent = vae.encode(lq[:, :, :, :3])

        # Ensure temporal dim matches
        if lq_latent.shape[2] > latent_t:
            lq_latent = lq_latent[:, :, :latent_t]
        elif lq_latent.shape[2] < latent_t:
            pad = latent_t - lq_latent.shape[2]
            lq_latent = torch.cat([lq_latent, lq_latent[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)

        # Build reference latent (16ch) — goes as concat_latent_image
        # concat_cond in model_base will concatenate this with the noise (16ch) → 32ch total
        ref_latent = torch.zeros_like(lq_latent)

        if ref_frames is not None:
            num_video_frames = lq_video.shape[0]

            # Determine reference indices
            if ref_mode == "manual" and ref_indices.strip():
                indices = [int(x.strip()) for x in ref_indices.split(",") if x.strip()]
            else:
                indices = _select_indices(num_video_frames)

            # Encode each reference frame and place at its temporal position.
            # SparkVSR places refs at specific latent indices, rest stays zeros.
            for ref_idx in indices:
                if ref_idx >= ref_frames.shape[0]:
                    continue

                frame = ref_frames[ref_idx:ref_idx + 1]
                frame = comfy.utils.common_upscale(frame.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                frame_latent = vae.encode(frame[:, :, :, :3])

                target_lat_idx = ref_idx // temporal_compression
                if target_lat_idx < latent_t:
                    ref_latent[:, :, target_lat_idx] = frame_latent[:, :, 0]

        # Set ref latent as concat conditioning (16ch, model_base.concat_cond adds it to noise)
        if ref_guidance_scale != 1.0 and ref_frames is not None:
            # CFG: positive gets real refs, negative gets zero refs
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": ref_latent})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": torch.zeros_like(ref_latent)})
        else:
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": ref_latent})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": ref_latent})

        # LQ latent is the base — KSampler will noise it and denoise
        out_latent = {"samples": lq_latent}
        return io.NodeOutput(positive, negative, out_latent)


def _select_indices(num_frames, max_refs=None):
    """Auto-select reference frame indices (first, evenly spaced, last)."""
    if max_refs is None:
        max_refs = (num_frames - 1) // 4 + 1
    max_refs = min(max_refs, 3)

    if num_frames <= 1:
        return [0]
    if max_refs == 1:
        return [0]
    if max_refs == 2:
        return [0, num_frames - 1]

    mid = num_frames // 2
    return [0, mid, num_frames - 1]


class CogVideoXExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SparkVSRConditioning,
        ]


async def comfy_entrypoint() -> CogVideoXExtension:
    return CogVideoXExtension()
