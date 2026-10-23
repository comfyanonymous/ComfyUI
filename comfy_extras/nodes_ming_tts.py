from comfy.ldm.ming_tts.runtime import generate, speaker_embedding
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_audio import VAEEncodeAudio


class MingTTSSample(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingTTSSample",
            display_name="Ming TTS Sample",
            category="audio/ming tts",
            description="Autoregressive speech sampling with the Ming flow model. Connect the result to VAE Decode Audio.",
            inputs=[
                io.Clip.Input("clip"),
                io.Model.Input("model"),
                io.String.Input("text", multiline=True, tooltip="Words to speak. For Tiny dialogue, start every line with speaker_1:, speaker_2:, etc."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("max_steps", default=250, min=1, max=4096, tooltip="Total audio-patch limit for the whole output, including all dialogue turns."),
                io.Float.Input("cfg", default=2.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("temperature", default=0.0, min=0.0, max=100.0, step=0.1),
                io.String.Input("instruction", multiline=True, default="", optional=True, tooltip="Optional voice or delivery description. Uses each model's released instruction format automatically. Tiny instruction tasks omit the language line and use voice identity without VAE audio context."),
                io.Combo.Input("language", options=["English", "Chinese", "German", "Spanish", "French", "Korean", "Japanese", "Auto"], default="Auto", optional=True, tooltip="Language hint for the supplied text; does not translate it. Tiny ignores this setting when instruction is nonempty, matching the reference. Auto lets Tiny infer Chinese or English and leaves the 0.5B language prompt unset."),
                io.Autogrow.Input("speaker", optional=True, template=io.Autogrow.TemplateNames(
                    input=io.Custom("MING_SPEAKER").Input("speaker"),
                    names=[f"speaker_{i}" for i in range(1, 101)], min=0,
                ), tooltip="Connect Ming Speaker Encode outputs in order. Multiple speakers require Tiny. VAE and reference text are optional."),
                io.Boolean.Input("mute_speaker", default=False, optional=True, advanced=True),
                io.Float.Input("sigma", default=0.25, min=0.0, max=100.0, step=0.01, optional=True, advanced=True),
                io.Boolean.Input("separate_turns", default=True, optional=True, tooltip="Generate speaker turns and blank-line-separated paragraphs independently, splitting long passages at chunk_length. max_steps still limits the whole output. Disable for joint generation; multiple speakers then need reference context."),
                io.Int.Input("chunk_length", default=90, min=1, max=4096, optional=True, advanced=True, tooltip="When separate_turns is on, target this many words or CJK characters per chunk. Prefer sentences, then clauses or word boundaries. Larger chunks give more continuous delivery; lower this if words are skipped. This is a text limit, not a duration in seconds."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, model, text, seed, max_steps, cfg, temperature, speaker: io.Autogrow.Type = None, language="Auto", **kwargs):
        speakers = []
        if speaker:
            names = [f"speaker_{i}" for i in range(1, len(speaker) + 1)]
            if set(speaker) != set(names):
                raise ValueError("Connect speakers consecutively, starting at speaker_1.")
            speakers = [speaker[name] for name in names]
        latents, _ = generate(
            clip, model, text, seed=seed, max_steps=max_steps, cfg=cfg, temperature=temperature,
            speakers=speakers, language="" if language == "Auto" else language, **kwargs,
        )
        return io.NodeOutput({"samples": latents, "type": "audio", "downscale_ratio_temporal": 3528})


class MingSpeakerEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingSpeakerEncode",
            display_name="Ming Speaker Encode",
            category="audio/ming tts",
            inputs=[
                io.Clip.Input("clip"),
                io.Audio.Input("audio"),
                io.Vae.Input("vae", optional=True, tooltip="Optional. With reference_text, also encode the recording as audio context. Without both, encode voice identity only."),
                io.String.Input("reference_text", multiline=True, default="", optional=True, tooltip="Optional transcript of this recording. Used only when a VAE is connected; do not add speaker labels."),
            ],
            outputs=[io.Custom("MING_SPEAKER").Output()],
        )

    @classmethod
    def execute(cls, clip, audio, vae=None, reference_text=""):
        speaker = {"embedding": speaker_embedding(clip, audio)}
        if speaker["embedding"].shape != (1, 192):
            raise ValueError("Use one recording per Ming Speaker Encode node.")
        if vae is not None and reference_text.strip():
            speaker["reference_latents"] = VAEEncodeAudio.execute(vae, audio)[0]["samples"]
            speaker["reference_text"] = reference_text.strip()
        return io.NodeOutput(speaker)


class MingTTSExtension(ComfyExtension):
    async def get_node_list(self):
        return [MingTTSSample, MingSpeakerEncode]


async def comfy_entrypoint():
    return MingTTSExtension()
