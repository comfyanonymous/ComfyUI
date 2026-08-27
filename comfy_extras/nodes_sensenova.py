from typing_extensions import override

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import node_helpers
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io


class EmptySenseNovaLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptySenseNovaLatentImage",
            display_name="Empty SenseNova Pixel Latent",
            category="model/latent/sensenova",
            description="Empty RGB pixel-space latent for SenseNova U1.5. Dimensions are rounded to 32-pixel patches by the model.",
            inputs=[
                io.Int.Input(id="width", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="height", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="batch_size", default=1, min=1, max=16),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> io.NodeOutput:
        samples = torch.zeros(
            (batch_size, 3, height, width),
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": samples})


def _prefix_cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(
        original_model_options
    )
    cache = {}
    guider.model_options.setdefault("transformer_options", {})[
        "sensenova_prefix_cache"
    ] = cache
    try:
        return executor(*args, **kwargs)
    finally:
        cache.clear()
        guider.model_options = original_model_options


class SenseNovaSamplingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/sensenova",
            description="Set the SenseNova flow shift and cache the text/reference prefix during each sampling run.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift: float) -> io.NodeOutput:
        patched = model.clone()
        model_sampling = SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "sensenova_prefix_cache",
            _prefix_cache_sample_wrapper,
        )
        return io.NodeOutput(patched)


class SenseNovaReferenceImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaReferenceImages",
            display_name="SenseNova Reference Images",
            category="model/conditioning/sensenova",
            description="Attach 1-10 source or reference images for SenseNova instruction editing.",
            inputs=[
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="negative"),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{index}" for index in range(1, 11)],
                        min=1,
                    ),
                    tooltip="Reference images in Image-1 through Image-10 order.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="image_condition"),
            ],
        )

    @classmethod
    def execute(cls, *, positive, negative, images: io.Autogrow.Type) -> io.NodeOutput:
        references = [
            images[f"image_{index}"]
            for index in range(1, 11)
            if f"image_{index}" in images
        ]
        for image in references:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
                raise ValueError(
                    "Each SenseNova reference input requires one IMAGE with at least three channels"
                )
        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "sensenova_reference_images": references,
                "sensenova_reference_mode": "condition",
            },
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "sensenova_reference_images": references,
                "sensenova_reference_mode": "image_only",
            },
            append=True,
        )
        return io.NodeOutput(positive, negative)


class SenseNovaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptySenseNovaLatentImage,
            SenseNovaSamplingOptions,
            SenseNovaReferenceImages,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    return SenseNovaExtension()
