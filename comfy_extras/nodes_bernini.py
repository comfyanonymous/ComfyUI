import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io


def _resize_long_edge(image, max_size, stride=16):
    """Resize (preserve aspect) so the long edge <= max_size, snapped to `stride`."""
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return comfy.utils.common_upscale(image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled").movedim(1, -1)


class BerniniConditioning(io.ComfyNode):
    """Bernini in-context conditioning for a Wan2.2-A14B model.

    Attaches the VAE-encoded source video / reference images to the conditioning
    an ordered list of clean latents (source video first, then each reference image),
    which the Wan model appends as extra tokens with per-stream source_id rope.

    The task is inferred from which inputs are connected:
      (nothing)                 -> t2v
      source_video              -> v2v
      source_video + ref images -> rv2v
      ref images only           -> r2v   (each kept at native aspect)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniConditioning",
            display_name="Bernini Conditioning",
            category="conditioning/video_models",
            description="Conditioning node for Bernini in-context video/image conditioning. Attach source video and/or reference images to the positive/negative conditioning, "
                        "which the Wan model will append as extra tokens with per-stream source_id rope.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("source_video", optional=True, tooltip="Source video to edit/restyle (original task v2v or rv2v). Resized to width/height and trimmed to length."),
                io.Image.Input("reference_images", optional=True, tooltip="Reference image(s) injected as in-context tokens (task r2v or rv2v). Each is kept at its native aspect ratio, long edge capped at ref_max_size."),
                io.Int.Input("ref_max_size", default=848, min=16, max=8192, step=16, optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                source_video=None, reference_images=None, ref_max_size=848) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())

        # Ordered list of condition streams: source video (source_id 1) first,
        # then each reference image (source_id 2, 3, ...), the model assigns the source_id from list order.
        context = []
        if source_video is not None:
            vid = comfy.utils.common_upscale(source_video[:length, :, :, :3].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            context.append(vae.encode(vid[:, :, :, :3]))

        if reference_images is not None:
            for i in range(reference_images.shape[0]):
                img = _resize_long_edge(reference_images[i:i + 1], ref_max_size)  # native aspect per ref
                context.append(vae.encode(img[:, :, :, :3]))

        if context:
            positive = node_helpers.conditioning_set_values(positive, {"context_latents": context})
            negative = node_helpers.conditioning_set_values(negative, {"context_latents": context})

        return io.NodeOutput(positive, negative, {"samples": latent})


class BerniniExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BerniniConditioning,
        ]


async def comfy_entrypoint() -> BerniniExtension:
    return BerniniExtension()
