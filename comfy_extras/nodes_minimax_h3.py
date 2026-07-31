"""MiniMax H3 nodes: AV latent creation and task conditioning (t2va / fl2va / ref2va).

The H3 packed-DiT consumes, via conditioning:
- Qwen3-VL-32B hidden states with per-token modality tags (from the minimax CLIP)
- keyframe / reference condition latents, re-injected every step (never denoised)

Latents are NestedTensor pairs (video [B,24,T,H/16,W/16], audio [B,32,2,T40]);
sampling runs on the flat pack with any stock sampler (the model handles the
audio stream's shifted schedule internally).
"""

import math
import types

import torch
import torchaudio

import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.utils
from comfy.ldm.minimax import model as minimax_model
import node_helpers
from comfy_api.latest import ComfyExtension, io, ui

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
MODULATION_MODEL_KEY = "minimax_h3_modulation"


def _modulation_timesteps(modulation_model, model_call_sigmas, transformer_options):
    modulation = modulation_model.model.diffusion_model
    timesteps = modulation_model.get_model_object("model_sampling").timestep(model_call_sigmas).flatten()
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", modulation.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", modulation.sigma_shift_audio))
    values = {minimax_model.VISUAL_COND_TIMESTEP, minimax_model.AUDIO_COND_TIMESTEP}
    for timestep in timesteps:
        t_v, t_a, _, _ = minimax_model.step_timesteps(timestep / 1000.0, shift_v, shift_a)
        values.update((t_v, t_a))
    return torch.tensor(sorted(values), dtype=torch.float32, device=model_call_sigmas.device)


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, C]
    samples = image.movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


class EmptyMiniMaxH3LatentAV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMiniMaxH3LatentAV",
            display_name="Empty MiniMax H3 AV Latent",
            category="latent/video",
            description="Joint video+audio latent for MiniMax H3. Duration snaps to the model's 17k+5 frame grid at 24 fps.",
            inputs=[
                io.Int.Input("width", default=1344, min=32, max=2048, step=32),
                io.Int.Input("height", default=768, min=32, max=2048, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, length) -> io.NodeOutput:
        latent, _ = _empty_av_latent(width, height, length)
        return io.NodeOutput(latent)


class MiniMaxH3TextToVideo(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TextToVideo",
            display_name="MiniMax H3 Text/Keyframe to Video",
            category="conditioning/video_models",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=2048, step=32),
                io.Int.Input("height", default=768, min=32, max=2048, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        images = []
        keyframes = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })
        return io.NodeOutput(cond, latent)


class MiniMaxH3ReferenceToVideo(io.ComfyNode):
    """ref2va: prompt + reference images / video / audio -> conditioning + AV latent.

    References enter the presentation in fixed order: images, then video, then
    standalone audio (matching their <Picture i>/<Video k>/<Audio j> labels).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideo",
            display_name="MiniMax H3 Reference to Video",
            category="conditioning/video_models",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=2048, step=32),
                io.Int.Input("height", default=768, min=32, max=2048, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image (encoded at 2048 short edge)"),
                        prefix="ref_image_", min=0, max=8)),
                io.Image.Input("ref_video_frames", optional=True, tooltip="Reference video frames at 24 fps"),
                io.Audio.Input("ref_video_audio", optional=True, tooltip="Audio track of the reference video"),
                io.Audio.Input("ref_audio", optional=True, tooltip="Standalone reference audio"),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]  # [B, C, L]
        sr = audio["sample_rate"]
        if sr != 32000:
            waveform = torchaudio.functional.resample(waveform, sr, 32000)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        z = audio_vae.first_stage_model.encode(waveform[:1, :2].float())  # [1, 32, 2, T]
        return z, z.shape[-1]

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_images=None, ref_video_frames=None,
                ref_video_audio=None, ref_audio=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # for the DiT payload, same order

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            scale = REF_IMAGE_SHORT_EDGE / min(w, h)
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        if ref_video_frames is not None:
            vh, vw = ref_video_frames.shape[1], ref_video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            frames = _resize(ref_video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            while n % 17 != 5 and n > 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if ref_video_audio is not None:
                audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, ref_video_audio)
            # Qwen sees the video at 2 fps with timestamps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        if ref_audio is not None:
            audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, ref_audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)


class MiniMaxH3SigmaShift(io.ComfyNode):
    """Set the video/audio flow shifts coherently.

    The video shift drives the sampler's sigma schedule; both values are also
    handed to the DiT, which inverts the video schedule to the shared base grid
    and derives the audio schedule from it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SigmaShift",
            description="Set the video/audio flow shifts.",
            display_name="MiniMax H3 Sigma Shift",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, shift_video, shift_audio) -> io.NodeOutput:
        m = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        original = m.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        m.add_object_patch("model_sampling", model_sampling)

        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to["minimax_h3_sigma_shift_video"] = shift_video
        to["minimax_h3_sigma_shift_audio"] = shift_audio
        return io.NodeOutput(m)


class MiniMaxH3Modulation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Modulation",
            display_name="MiniMax H3 Modulation",
            category="advanced/model",
            description="Precompute the sampling schedule's modulation tensors and attach them to the split transformer.",
            inputs=[
                io.Model.Input("model"),
                io.Model.Input("modulation_model"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Combo.Input("cache_location", options=["auto", "ram", "vram"], default="auto"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, modulation_model, sampler, sigmas, cache_location="auto") -> io.NodeOutput:
        diffusion_model = model.model.diffusion_model
        modulation = modulation_model.model.diffusion_model
        if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model) or not diffusion_model.split_modulation:
            raise ValueError("model must be a split MiniMax H3 transformer")
        if not isinstance(modulation, minimax_model.MiniMaxH3ModulationModel):
            raise ValueError("modulation_model must be a MiniMax H3 modulation model")
        if len(diffusion_model.blocks) != len(modulation.blocks) or diffusion_model.hidden_size != modulation.blocks[0].adaln_proj.hidden:
            raise ValueError("MiniMax H3 transformer and modulation model configurations do not match")

        model_wrap = types.SimpleNamespace(inner_model=types.SimpleNamespace(
            diffusion_model=modulation,
            model_sampling=modulation_model.get_model_object("model_sampling"),
        ))
        sigmas = sigmas.to(modulation_model.load_device)
        model_call_sigmas = sampler.get_model_call_sigmas(model_wrap, sigmas)
        if model_call_sigmas is None:
            raise RuntimeError("This sampler cannot precompute its model-call sigma schedule")

        transformer_options = modulation_model.model_options.get("transformer_options", {})
        timesteps = _modulation_timesteps(modulation_model, model_call_sigmas, transformer_options)
        hidden = modulation.blocks[0].adaln_proj.hidden
        block_elements = len(modulation.blocks) * 6 * len(timesteps) * 3 * hidden
        final_elements = 2 * len(timesteps) * hidden
        block_dtype = modulation.blocks[0].adaln_proj.linear.bias.dtype
        final_dtype = modulation.final_layer.adaln_proj.linear.bias.dtype
        cache_memory = (block_elements * comfy.model_management.dtype_size(block_dtype)
                        + final_elements * comfy.model_management.dtype_size(final_dtype))
        block_scratch = block_elements // len(modulation.blocks) * comfy.model_management.dtype_size(block_dtype)
        timestep_scratch = len(timesteps) * modulation.time_embedder.proj_out.out_features * 4
        comfy.model_management.load_models_gpu([modulation_model],
                                               memory_required=cache_memory + block_scratch + timestep_scratch)
        timesteps = timesteps.to(modulation_model.load_device)
        modulation_model.pre_run()
        try:
            modulation_options = {"prefetch_dynamic_vbars": modulation_model.is_dynamic()}
            blocks, final = modulation(timesteps, transformer_options=modulation_options)
        finally:
            modulation_model.cleanup()

        if cache_location == "auto":
            cache_location = "vram" if modulation_model.load_device.type != "cpu" and comfy.model_management.get_total_memory(modulation_model.load_device) > 40 * 1024 ** 3 else "ram"
        if cache_location == "ram":
            timesteps = timesteps.cpu()
            blocks = blocks.cpu()
            final = final.cpu()

        m = model.clone()
        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to[MODULATION_MODEL_KEY] = minimax_model.MiniMaxH3ModulationCache(timesteps, blocks, final)
        cached_bytes = sum(x.numel() * x.element_size() for x in (timesteps, blocks, final))
        preview = f"Cached {cached_bytes / 1024 ** 3:.2f} GiB of modulation data in {cache_location.upper()}."
        return io.NodeOutput(m, ui=ui.PreviewText(preview))


class MiniMaxH3SeparateAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SeparateAVLatent",
            display_name="MiniMax H3 Separate AV Latent",
            category="latent/video",
            inputs=[io.Latent.Input("av_latent")],
            outputs=[io.Latent.Output(display_name="video"), io.Latent.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, av_latent) -> io.NodeOutput:
        samples = av_latent["samples"]
        if samples.is_nested:
            video, audio = samples.unbind()
        else:
            video, audio = samples, None
        video_out = {k: v for k, v in av_latent.items() if k != "samples"}
        audio_out = dict(video_out)
        video_out["samples"] = video
        audio_out["samples"] = audio
        audio_out["type"] = "audio"
        return io.NodeOutput(video_out, audio_out)


class MiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [EmptyMiniMaxH3LatentAV, MiniMaxH3TextToVideo, MiniMaxH3ReferenceToVideo,
                MiniMaxH3SigmaShift, MiniMaxH3Modulation, MiniMaxH3SeparateAVLatent]


async def comfy_entrypoint() -> MiniMaxH3Extension:
    return MiniMaxH3Extension()
