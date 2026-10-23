import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
from comfy.ldm.sensenova.conditioning import (
    IMAGE_CONTEXT_ID,
    IMAGE_END_ID,
    IMAGE_START_ID,
    THINK_SUFFIX_IDS,
)
from comfy.ldm.sensenova.interleave import (
    append_interleave_image,
    append_interleave_tokens,
    expand_interleave_metadata,
    generate_text_event,
    prefix_arguments,
    split_interleave_text,
)
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io, ui
from comfy_execution.graph_utils import ExecutionBlocker


class SenseNovaSamplingOptions(io.ComfyNode):
    """Configure SenseNova flow sampling parameters on a model patcher."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/sensenova",
            description="Set the SenseNova flow shift.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, min=0.01, step=0.01),
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
    """Encode a SenseNova image or interleave prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaTextEncode",
            display_name="SenseNova Text Encode",
            category="model/conditioning/sensenova",
            description=(
                "Encode a SenseNova prompt. Use SenseNova Generate before the "
                "KSampler when image-generation thinking is required."
            ),
            inputs=[
                io.Clip.Input(id="clip"),
                io.String.Input(id="text", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    id="mode",
                    options=["image", "interleave"],
                    default="image",
                ),
                io.Boolean.Input(
                    id="thinking",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Set true for T2I/Edit before SenseNova Generate; it also "
                        "controls interleave thinking."
                    ),
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
        mode: str = "image",
    ) -> io.NodeOutput:
        tokenize_options = {"thinking": thinking}
        if mode == "interleave":
            tokenize_options["mode"] = mode
        tokens = clip.tokenize(text, **tokenize_options)
        metadata = {"sensenova_thinking": thinking}
        if mode == "interleave":
            metadata["sensenova_interleave"] = True
        conditioning = clip.encode_from_tokens_scheduled(
            tokens,
            add_dict=metadata,
        )
        return io.NodeOutput(conditioning)


class SenseNovaGenerate(io.ComfyNode):
    """Generate SenseNova image reasoning before standard image sampling."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaGenerate",
            display_name="SenseNova Generate",
            category="model/sampling/sensenova",
            description=(
                "Run SenseNova thinking when enabled; otherwise pass conditioning "
                "through unchanged. In interleave mode, stop at the next image "
                "request or EOS so an external KSampler can produce the image."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input(
                    "samples",
                    optional=True,
                    tooltip="Previous KSampler pixel latent, required after an image request.",
                ),
                io.Int.Input(
                    "max_think_tokens",
                    default=1024,
                    min=1,
                    max=8192,
                    advanced=True,
                    tooltip="Maximum generated tokens for thinking or one interleave text stage.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(
                    display_name="thinking",
                    tooltip="Thinking text for image mode, or the current text stage for interleave mode.",
                ),
                io.Conditioning.Output(display_name="negative"),
                io.Boolean.Output(display_name="image_requested"),
            ],
        )

    @classmethod
    def execute(
        cls,
        *,
        model,
        clip,
        conditioning,
        max_think_tokens,
        negative=None,
        samples=None,
    ) -> io.NodeOutput:
        if not conditioning:
            return io.NodeOutput(
                conditioning, "", negative or [], False, ui=ui.PreviewText("")
            )
        metadata = conditioning[0][1]
        if metadata.get("sensenova_interleave"):
            return cls._execute_interleave(
                model=model,
                clip=clip,
                positive=conditioning,
                negative=negative,
                samples=samples,
                max_text_tokens=max_think_tokens,
            )
        if not metadata.get("sensenova_thinking", False):
            return io.NodeOutput(
                conditioning, "", negative or [], True, ui=ui.PreviewText("")
            )
        if len(conditioning) != 1:
            raise ValueError(
                "SenseNova Generate requires one positive conditioning entry "
                "when thinking is enabled."
            )
        if metadata.get("text_input_ids") is None:
            raise ValueError("SenseNova Generate requires SenseNova text conditioning.")

        diffusion_model = model.model.diffusion_model
        device = model.load_device
        dtype = model.model.get_dtype_inference()
        transformer_options = model.model_options.get(
            "transformer_options", {}
        ).copy()
        comfy.model_management.load_models_gpu([model])
        model.pre_run()
        try:
            prefix_args = prefix_arguments(
                metadata, device, dtype, image_only=False
            )
            progress = comfy.utils.ProgressBar(max_think_tokens)
            token_ids = diffusion_model.generate_thinking_tokens(
                *prefix_args,
                max_think_tokens=max_think_tokens,
                transformer_options=transformer_options,
                progress=progress.update_absolute,
                interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
            )
        finally:
            model.cleanup()

        thinking = _format_thinking_text(
            clip.decode(token_ids, skip_special_tokens=True)
        )
        if not thinking:
            thinking = "SenseNova thinking completed without visible text."

        input_ids = metadata["text_input_ids"]
        suffix = input_ids.new_tensor([*token_ids, *THINK_SUFFIX_IDS]).unsqueeze(0)
        updated_metadata = {
            **metadata,
            "text_input_ids": torch.cat((input_ids, suffix), dim=1),
            "sensenova_thinking": False,
        }
        updated_conditioning = [[conditioning[0][0], updated_metadata]]

        return io.NodeOutput(
            updated_conditioning,
            thinking,
            negative or [],
            True,
            ui=ui.PreviewText(thinking),
        )

    @classmethod
    def _execute_interleave(
        cls,
        *,
        model,
        clip,
        positive,
        negative,
        samples,
        max_text_tokens,
    ) -> io.NodeOutput:
        """Generate one interleave text stage for external KSampler orchestration."""

        if negative is None or len(positive) != 1 or len(negative) != 1:
            raise ValueError(
                "SenseNova interleave requires one positive and one negative conditioning entry."
            )
        if positive[0][1].get("sensenova_interleave_finished"):
            return io.NodeOutput(
                positive, "", negative, False, ui=ui.PreviewText("")
            )
        positive_metadata = expand_interleave_metadata(positive[0][1], image_only=False)
        negative_metadata = expand_interleave_metadata(negative[0][1], image_only=True)
        pending_image = bool(
            positive_metadata.get("sensenova_interleave_pending_image")
        )
        if pending_image != bool(
            negative_metadata.get("sensenova_interleave_pending_image")
        ):
            raise ValueError("SenseNova interleave conditioning histories are out of sync.")
        latent_samples = samples.get("samples") if isinstance(samples, dict) else None
        if pending_image:
            if latent_samples is None:
                raise ValueError(
                    "SenseNova interleave is waiting for samples from the preceding KSampler."
                )
            positive_metadata = append_interleave_image(
                positive_metadata, latent_samples
            )
            negative_metadata = append_interleave_image(
                negative_metadata, latent_samples
            )
        elif latent_samples is not None:
            raise ValueError(
                "SenseNova interleave received samples before requesting an image."
            )

        diffusion_model = model.model.diffusion_model
        device = model.load_device
        transformer_options = model.model_options.get(
            "transformer_options", {}
        ).copy()
        comfy.model_management.load_models_gpu([model])
        model.pre_run()
        try:
            progress = comfy.utils.ProgressBar(max_text_tokens)
            event = generate_text_event(
                diffusion_model,
                prefix_arguments(
                    positive_metadata,
                    device,
                    model.model.get_dtype_inference(),
                    image_only=False,
                ),
                max_text_tokens=max_text_tokens,
                transformer_options=transformer_options,
                progress=progress.update_absolute,
                interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
            )
        finally:
            model.cleanup()

        positive_metadata = append_interleave_tokens(
            positive_metadata, event.token_ids, event.image_requested
        )
        positive_metadata["sensenova_interleave_finished"] = event.stop_reason == "eos"
        if event.image_requested:
            negative_metadata = append_interleave_tokens(
                negative_metadata, [IMAGE_START_ID], True
            )
        text_token_ids = [value for value in event.token_ids if value != IMAGE_START_ID]
        text = (
            clip.decode(text_token_ids, skip_special_tokens=True).strip()
            if text_token_ids
            else ""
        )
        updated_positive = [[positive[0][0], positive_metadata]]
        updated_negative = [[negative[0][0], negative_metadata]]
        return io.NodeOutput(
            updated_positive,
            text,
            updated_negative,
            event.image_requested,
            ui=ui.PreviewText(text),
        )


def _format_thinking_text(text: str) -> str:
    """Return only the visible body of a SenseNova thinking result."""

    text = text.strip()
    if text.startswith("<think>"):
        text = text[len("<think>") :].lstrip()
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    return text.strip()


class SenseNovaDecode(io.ComfyNode):
    """Decode the generated portion of a SenseNova interleave history."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaDecode",
            display_name="SenseNova Decode",
            category="model/sampling/sensenova",
            description=(
                "Decode the final conditioning from SenseNova Generate into text, "
                "an image list, and thinking. With loops, use End Loop with "
                "accumulate disabled. Input reference images are excluded. "
                "Image consumers are skipped when no images were generated."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Last sampled image, if the loop ended on an image request.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.Image.Output(display_name="images", is_output_list=True),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        conditioning,
        image=None,
    ) -> io.NodeOutput:
        if len(conditioning) != 1 or "sensenova_interleave_start" not in conditioning[0][1]:
            raise ValueError(
                "SenseNova Decode requires interleave conditioning from SenseNova Generate."
            )
        metadata = conditioning[0][1]
        token_ids = metadata["text_input_ids"][0, metadata["sensenova_interleave_start"]:].tolist()
        text_parts = []
        text_tokens = []
        for token_id in token_ids:
            if token_id == IMAGE_START_ID:
                text_parts.append(clip.decode(text_tokens, skip_special_tokens=True))
                text_tokens = []
                text_parts.append("<image>")
            elif token_id not in (IMAGE_CONTEXT_ID, IMAGE_END_ID):
                text_tokens.append(token_id)
        text_parts.append(clip.decode(text_tokens, skip_special_tokens=True))

        text, thinking = split_interleave_text("".join(text_parts))
        images = [
            reference.clamp(0.0, 1.0)
            for reference in metadata["reference_latents"][metadata["sensenova_interleave_input_images"]:]
        ]
        if metadata["sensenova_interleave_pending_image"] and image is not None:
            images.extend(value.clone() for value in image.split(1))
        return io.NodeOutput(text, images or [ExecutionBlocker(None)], thinking)


class SenseNovaExtension(ComfyExtension):
    """Register the native SenseNova node collection."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaTextEncode,
            SenseNovaGenerate,
            SenseNovaSamplingOptions,
            SenseNovaDecode,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    """Create the native SenseNova node extension."""

    return SenseNovaExtension()
