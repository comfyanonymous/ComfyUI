import torch
from typing_extensions import override

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from comfy.ldm.sensenova.interleave import (
    SenseNovaInterleaveSession,
    _live_conditioning,
    _prefix_arguments,
    build_interleave_result,
    interleave_result_to_markdown,
)
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io, ui


InterleaveResultIO = io.Custom("SENSENOVA_INTERLEAVE_RESULT")
WEB_DIRECTORY = "./web"


def interleave_output_samples(result, latent_samples):
    if not result.images:
        return latent_samples
    return torch.cat(result.images).to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )


def run_interleave(
    model,
    clip,
    positive,
    negative,
    noise_seed,
    cfg,
    sampler,
    sigmas,
    latent,
    max_text_tokens,
    max_images=1,
):
    latent = latent.copy()
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    if latent_samples.shape[0] != 1:
        raise ValueError("SenseNova interleave requires a single latent image.")
    latent["samples"] = latent_samples

    positive_data = positive[0][1]
    negative_data = negative[0][1]
    if not positive_data.get("sensenova_interleave") or not negative_data.get(
        "sensenova_interleave"
    ):
        raise ValueError(
            "SenseNova interleave requires positive and negative conditioning "
            "encoded with mode=interleave."
        )

    comfy.model_management.load_models_gpu([model])
    device = model.load_device
    transformer_options = model.model_options.get("transformer_options", {}).copy()
    model.pre_run()
    try:
        diffusion_model = model.model.diffusion_model
        session = SenseNovaInterleaveSession(
            diffusion_model,
            positive_prefix=_prefix_arguments(
                positive_data, device, diffusion_model.dtype, image_only=False
            ),
            negative_prefix=_prefix_arguments(
                negative_data, device, diffusion_model.dtype, image_only=True
            ),
            decode_tokens=lambda values: clip.tokenizer.sensenova_u15.tokenizer.decode(
                values, skip_special_tokens=True
            ),
            transformer_options=transformer_options,
        )
        image_index = 0

        def sample_image(positive_prefix, negative_prefix):
            nonlocal image_index
            noise = comfy.sample.prepare_noise(
                latent_samples, noise_seed, [image_index]
            )
            callback = latent_preview.prepare_callback(
                model, sigmas.shape[-1] - 1, {}
            )
            samples = comfy.sample.sample_custom(
                model,
                noise,
                cfg,
                sampler,
                sigmas,
                _live_conditioning(positive_prefix),
                _live_conditioning(negative_prefix),
                latent_samples,
                noise_mask=latent.get("noise_mask"),
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=noise_seed,
            )
            image_index += 1
            comfy.model_management.load_models_gpu([model])
            model.pre_run()
            return samples.to(device=device, dtype=diffusion_model.dtype)

        progress = comfy.utils.ProgressBar(max_text_tokens)
        result = session.generate(
            sample_image,
            max_text_tokens=max_text_tokens,
            max_images=max_images,
            progress=progress.update_absolute,
            interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
        )
    finally:
        model.cleanup()

    latent.pop("downscale_ratio_spacial", None)
    latent.pop("downscale_ratio_temporal", None)
    latent["samples"] = interleave_output_samples(result, latent_samples)
    return latent, result.text, build_interleave_result(result)


class SenseNovaSamplingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/sensenova",
            description="Set the SenseNova flow shift.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift: float) -> io.NodeOutput:
        patched = model.clone()
        model_sampling = SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(patched)


class SenseNovaTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaTextEncode",
            display_name="SenseNova 1.x Text Encode",
            category="model/conditioning/sensenova",
            description="Encode a SenseNova prompt with optional image-generation reasoning.",
            inputs=[
                io.Clip.Input(id="clip"),
                io.String.Input(id="text", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    id="mode",
                    options=["image", "interleave"],
                    default="image",
                ),
                io.Boolean.Input(id="thinking", default=False),
                io.Int.Input(
                    id="max_think_tokens",
                    default=1024,
                    min=1,
                    advanced=True,
                ),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(
        cls,
        *,
        clip,
        text: str,
        thinking: bool,
        max_think_tokens: int,
        mode: str = "image",
    ) -> io.NodeOutput:
        tokenize_options = {"thinking": thinking}
        if mode == "interleave":
            tokenize_options["mode"] = mode
        tokens = clip.tokenize(text, **tokenize_options)
        metadata = {
            "sensenova_thinking": thinking,
            "sensenova_max_think_tokens": max_think_tokens,
        }
        if mode == "interleave":
            metadata["sensenova_interleave"] = True
        conditioning = clip.encode_from_tokens_scheduled(
            tokens,
            add_dict=metadata,
        )
        return io.NodeOutput(conditioning)


class SenseNovaInterleave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaInterleave",
            display_name="SenseNova 1.x Interleave",
            category="model/sampling/sensenova",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01
                ),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Int.Input(
                    "max_text_tokens", default=1024, min=1, max=8192, advanced=True
                ),
                io.Int.Input("max_images", default=4, min=1, max=10, advanced=True),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
                io.String.Output(display_name="text"),
                InterleaveResultIO.Output(display_name="interleave_result"),
            ],
        )

    @classmethod
    def execute(
        cls,
        *,
        model,
        clip,
        positive,
        negative,
        noise_seed,
        cfg,
        sampler,
        sigmas,
        latent_image,
        max_text_tokens,
        max_images,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *run_interleave(
                model,
                clip,
                positive,
                negative,
                noise_seed,
                cfg,
                sampler,
                sigmas,
                latent_image,
                max_text_tokens,
                max_images,
            )
        )


def _save_preview_images(images):
    if images is None or images.shape[0] == 0:
        return []
    return [dict(value) for value in ui.PreviewImage(images).as_dict()["images"]]


class SenseNovaInterleavePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaInterleavePreview",
            display_name="SenseNova Interleave Preview",
            category="model/sampling/sensenova",
            is_output_node=True,
            inputs=[
                InterleaveResultIO.Input("interleave_result"),
                io.Boolean.Input("include_think", default=False),
                io.Image.Input("images", optional=True),
            ],
            outputs=[io.String.Output(display_name="markdown")],
        )

    @classmethod
    def execute(cls, *, interleave_result, include_think, images=None) -> io.NodeOutput:
        markdown = interleave_result_to_markdown(
            interleave_result, include_think=include_think
        )
        saved_images = _save_preview_images(images)
        parts_payload = []
        for part in interleave_result.get("parts", []):
            part_type = part.get("type")
            if part_type == "think" and not include_think:
                continue
            if part_type in ("text", "think"):
                text = str(part.get("text", "")).strip()
                if text:
                    parts_payload.append({"type": part_type, "text": text})
            elif part_type == "image":
                index = int(part.get("index", 0))
                image = saved_images[index] if index < len(saved_images) else None
                if image is None:
                    parts_payload.append(
                        {"type": "image", "index": index, "missing": True}
                    )
                else:
                    parts_payload.append(
                        {
                            "type": "image",
                            "index": index,
                            "filename": image.get("filename", ""),
                            "subfolder": image.get("subfolder", ""),
                            "image_type": image.get("type", "temp"),
                        }
                    )
        return io.NodeOutput(
            markdown,
            ui={"text": [markdown], "parts": parts_payload},
        )

class SenseNovaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaTextEncode,
            SenseNovaSamplingOptions,
            SenseNovaInterleave,
            SenseNovaInterleavePreview,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    return SenseNovaExtension()
