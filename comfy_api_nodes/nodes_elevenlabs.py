import json
import uuid

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.elevenlabs import (
    DialogueInput,
    DialogueSettings,
    TextToSpeechVoiceSettings,
)
from comfy_api_nodes.util import (
    audio_bytes_to_audio_input,
    audio_ndarray_to_bytesio,
    audio_tensor_to_contiguous_ndarray,
    download_url_as_bytesio,
    validate_string,
)
from comfy_api_nodes.util.client import fal_run
from comfy_api_nodes.util.upload_helpers import upload_file_to_fal

FAL_ELEVENLABS_TTS = "fal-ai/elevenlabs/tts/turbo-v2.5"

ELEVENLABS_MUSIC_SECTIONS = "ELEVENLABS_MUSIC_SECTIONS"  # Custom type for music sections
ELEVENLABS_COMPOSITION_PLAN = "ELEVENLABS_COMPOSITION_PLAN"  # Custom type for composition plan
ELEVENLABS_VOICE = "ELEVENLABS_VOICE"  # Custom type for voice selection

# Predefined ElevenLabs voices: (voice_id, display_name, gender, accent)
ELEVENLABS_VOICES = [
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger", "male", "american"),
    ("EXAVITQu4vr4xnSDxMaL", "Sarah", "female", "american"),
    ("FGY2WhTYpPnrIDTdsKH5", "Laura", "female", "american"),
    ("IKne3meq5aSn9XLyUdCD", "Charlie", "male", "australian"),
    ("JBFqnCBsd6RMkjVDRZzb", "George", "male", "british"),
    ("N2lVS1w4EtoT3dr4eOWO", "Callum", "male", "american"),
    ("SAz9YHcvj6GT2YYXdXww", "River", "neutral", "american"),
    ("SOYHLrjzK2X1ezoPC6cr", "Harry", "male", "american"),
    ("TX3LPaxmHKxFdv7VOQHJ", "Liam", "male", "american"),
    ("Xb7hH8MSUJpSbSDYk0k2", "Alice", "female", "british"),
    ("XrExE9yKIg1WjnnlVkGX", "Matilda", "female", "american"),
    ("bIHbv24MWmeRgasZH58o", "Will", "male", "american"),
    ("cgSgspJ2msm6clMCkdW9", "Jessica", "female", "american"),
    ("cjVigY5qzO86Huf0OWal", "Eric", "male", "american"),
    ("hpp4J3VqNfWAUOO0d1Us", "Bella", "female", "american"),
    ("iP95p4xoKVk53GoZ742B", "Chris", "male", "american"),
    ("nPczCjzI2devNBz1zQrb", "Brian", "male", "american"),
    ("onwK4e9ZLuTAKqWW03F9", "Daniel", "male", "british"),
    ("pFZP5JQG7iQjIQuC4Bku", "Lily", "female", "british"),
    ("pNInz6obpgDQGcFmaJgB", "Adam", "male", "american"),
    ("pqHfZKP75CvOlQylNhV4", "Bill", "male", "american"),
]

ELEVENLABS_VOICE_OPTIONS = [f"{name} ({gender}, {accent})" for _, name, gender, accent in ELEVENLABS_VOICES]
ELEVENLABS_VOICE_MAP = {
    f"{name} ({gender}, {accent})": voice_id for voice_id, name, gender, accent in ELEVENLABS_VOICES
}


async def _upload_audio_to_fal(audio: Input.Audio) -> str:
    """Convert audio input to bytes and upload to fal.ai CDN. Returns the CDN URL."""
    audio_data_np = audio_tensor_to_contiguous_ndarray(audio["waveform"])
    audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, audio["sample_rate"], "mp4", "aac")
    return await upload_file_to_fal(audio_bytes_io, "audio/mp4")


class ElevenLabsSpeechToText(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsSpeechToText",
            display_name="ElevenLabs Speech to Text",
            category="api node/audio/ElevenLabs",
            description="Transcribe audio to text. "
            "Supports automatic language detection, speaker diarization, and audio event tagging.",
            inputs=[
                IO.Audio.Input(
                    "audio",
                    tooltip="Audio to transcribe.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "scribe_v2",
                            [
                                IO.Boolean.Input(
                                    "tag_audio_events",
                                    default=False,
                                    tooltip="Annotate sounds like (laughter), (music), etc. in transcript.",
                                ),
                                IO.Boolean.Input(
                                    "diarize",
                                    default=False,
                                    tooltip="Annotate which speaker is talking.",
                                ),
                                IO.Float.Input(
                                    "diarization_threshold",
                                    default=0.22,
                                    min=0.1,
                                    max=0.4,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Speaker separation sensitivity. "
                                    "Lower values are more sensitive to speaker changes.",
                                ),
                                IO.Float.Input(
                                    "temperature",
                                    default=0.0,
                                    min=0.0,
                                    max=2.0,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Randomness control. "
                                    "0.0 uses model default. Higher values increase randomness.",
                                ),
                                IO.Combo.Input(
                                    "timestamps_granularity",
                                    options=["word", "character", "none"],
                                    default="word",
                                    tooltip="Timing precision for transcript words.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use for transcription.",
                ),
                IO.String.Input(
                    "language_code",
                    default="",
                    tooltip="ISO-639-1 or ISO-639-3 language code (e.g., 'en', 'es', 'fra'). "
                    "Leave empty for automatic detection.",
                ),
                IO.Int.Input(
                    "num_speakers",
                    default=0,
                    min=0,
                    max=32,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Maximum number of speakers to predict. Set to 0 for automatic detection.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    tooltip="Seed for reproducibility (determinism not guaranteed).",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="text"),
                IO.String.Output(display_name="language_code"),
                IO.String.Output(display_name="words_json"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        audio: Input.Audio,
        model: dict,
        language_code: str,
        num_speakers: int,
        seed: int,
    ) -> IO.NodeOutput:
        if model["diarize"] and num_speakers:
            raise ValueError(
                "Number of speakers cannot be specified when diarization is enabled. "
                "Either disable diarization or set num_speakers to 0."
            )
        audio_url = await _upload_audio_to_fal(audio)
        result = await fal_run(cls, FAL_ELEVENLABS_TTS, {
            "model_id": model["model"],
            "audio_url": audio_url,
            "language_code": language_code if language_code.strip() else None,
            "tag_audio_events": model["tag_audio_events"],
            "num_speakers": num_speakers if num_speakers > 0 else None,
            "timestamps_granularity": model["timestamps_granularity"],
            "diarize": model["diarize"],
            "diarization_threshold": model["diarization_threshold"] if model["diarize"] else None,
            "seed": seed,
            "temperature": model["temperature"],
        })
        # TODO: verify fal.ai field names
        words_json = json.dumps(
            result.get("words", []),
            indent=2,
        )
        return IO.NodeOutput(result["text"], result.get("language_code", ""), words_json)


class ElevenLabsVoiceSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsVoiceSelector",
            display_name="ElevenLabs Voice Selector",
            category="api node/audio/ElevenLabs",
            description="Select a predefined ElevenLabs voice for text-to-speech generation.",
            inputs=[
                IO.Combo.Input(
                    "voice",
                    options=ELEVENLABS_VOICE_OPTIONS,
                    tooltip="Choose a voice from the predefined ElevenLabs voices.",
                ),
            ],
            outputs=[
                IO.Custom(ELEVENLABS_VOICE).Output(display_name="voice"),
            ],
            is_api_node=False,
        )

    @classmethod
    def execute(cls, voice: str) -> IO.NodeOutput:
        voice_id = ELEVENLABS_VOICE_MAP.get(voice)
        if not voice_id:
            raise ValueError(f"Unknown voice: {voice}")
        return IO.NodeOutput(voice_id)


class ElevenLabsTextToSpeech(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsTextToSpeech",
            display_name="ElevenLabs Text to Speech",
            category="api node/audio/ElevenLabs",
            description="Convert text to speech.",
            inputs=[
                IO.Custom(ELEVENLABS_VOICE).Input(
                    "voice",
                    tooltip="Voice to use for speech synthesis. Connect from Voice Selector or Instant Voice Clone.",
                ),
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="The text to convert to speech.",
                ),
                IO.Float.Input(
                    "stability",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Voice stability. Lower values give broader emotional range, "
                    "higher values produce more consistent but potentially monotonous speech.",
                ),
                IO.Combo.Input(
                    "apply_text_normalization",
                    options=["auto", "on", "off"],
                    tooltip="Text normalization mode. 'auto' lets the system decide, "
                    "'on' always applies normalization, 'off' skips it.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "eleven_multilingual_v2",
                            [
                                IO.Float.Input(
                                    "speed",
                                    default=1.0,
                                    min=0.7,
                                    max=1.3,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Speech speed. 1.0 is normal, <1.0 slower, >1.0 faster.",
                                ),
                                IO.Float.Input(
                                    "similarity_boost",
                                    default=0.75,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Similarity boost. Higher values make the voice more similar to the original.",
                                ),
                                IO.Boolean.Input(
                                    "use_speaker_boost",
                                    default=False,
                                    tooltip="Boost similarity to the original speaker voice.",
                                ),
                                IO.Float.Input(
                                    "style",
                                    default=0.0,
                                    min=0.0,
                                    max=0.2,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Style exaggeration. Higher values increase stylistic expression "
                                    "but may reduce stability.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "eleven_v3",
                            [
                                IO.Float.Input(
                                    "speed",
                                    default=1.0,
                                    min=0.7,
                                    max=1.3,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Speech speed. 1.0 is normal, <1.0 slower, >1.0 faster.",
                                ),
                                IO.Float.Input(
                                    "similarity_boost",
                                    default=0.75,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Similarity boost. Higher values make the voice more similar to the original.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use for text-to-speech.",
                ),
                IO.String.Input(
                    "language_code",
                    default="",
                    tooltip="ISO-639-1 or ISO-639-3 language code (e.g., 'en', 'es', 'fra'). "
                    "Leave empty for automatic detection.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    tooltip="Seed for reproducibility (determinism not guaranteed).",
                ),
                IO.Combo.Input(
                    "output_format",
                    options=["mp3_44100_192", "opus_48000_192"],
                    tooltip="Audio output format.",
                ),
            ],
            outputs=[
                IO.Audio.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        voice: str,
        text: str,
        stability: float,
        apply_text_normalization: str,
        model: dict,
        language_code: str,
        seed: int,
        output_format: str,
    ) -> IO.NodeOutput:
        validate_string(text, min_length=1)
        voice_settings = TextToSpeechVoiceSettings(
            stability=stability,
            similarity_boost=model["similarity_boost"],
            speed=model["speed"],
            use_speaker_boost=model.get("use_speaker_boost", None),
            style=model.get("style", None),
        )
        result = await fal_run(cls, FAL_ELEVENLABS_TTS, {
            "text": text,
            "voice_id": voice,
            "model_id": model["model"],
            "language_code": language_code if language_code.strip() else None,
            "voice_settings": voice_settings.model_dump(exclude_none=True),
            "seed": seed,
            "apply_text_normalization": apply_text_normalization,
            "output_format": output_format,
        })
        # TODO: verify fal.ai field names
        audio_url = result["audio"]["url"]
        audio_bytes_io = await download_url_as_bytesio(audio_url)
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes_io.read()))


class ElevenLabsAudioIsolation(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsAudioIsolation",
            display_name="ElevenLabs Voice Isolation",
            category="api node/audio/ElevenLabs",
            description="Remove background noise from audio, isolating vocals or speech.",
            inputs=[
                IO.Audio.Input(
                    "audio",
                    tooltip="Audio to process for background noise removal.",
                ),
            ],
            outputs=[
                IO.Audio.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        audio: Input.Audio,
    ) -> IO.NodeOutput:
        audio_url = await _upload_audio_to_fal(audio)
        result = await fal_run(cls, FAL_ELEVENLABS_TTS, {
            "audio_url": audio_url,
        })
        # TODO: verify fal.ai field names
        output_url = result["audio"]["url"]
        audio_bytes_io = await download_url_as_bytesio(output_url)
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes_io.read()))


class ElevenLabsTextToSoundEffects(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsTextToSoundEffects",
            display_name="ElevenLabs Text to Sound Effects",
            category="api node/audio/ElevenLabs",
            description="Generate sound effects from text descriptions.",
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Text description of the sound effect to generate.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "eleven_sfx_v2",
                            [
                                IO.Float.Input(
                                    "duration",
                                    default=5.0,
                                    min=0.5,
                                    max=30.0,
                                    step=0.1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of generated sound in seconds.",
                                ),
                                IO.Boolean.Input(
                                    "loop",
                                    default=False,
                                    tooltip="Create a smoothly looping sound effect.",
                                ),
                                IO.Float.Input(
                                    "prompt_influence",
                                    default=0.3,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="How closely generation follows the prompt. "
                                    "Higher values make the sound follow the text more closely.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use for sound effect generation.",
                ),
                IO.Combo.Input(
                    "output_format",
                    options=["mp3_44100_192", "opus_48000_192"],
                    tooltip="Audio output format.",
                ),
            ],
            outputs=[
                IO.Audio.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        text: str,
        model: dict,
        output_format: str,
    ) -> IO.NodeOutput:
        validate_string(text, min_length=1)
        result = await fal_run(cls, FAL_ELEVENLABS_TTS, {
            "text": text,
            "duration_seconds": model["duration"],
            "prompt_influence": model["prompt_influence"],
            "loop": model.get("loop", None),
            "output_format": output_format,
        })
        # TODO: verify fal.ai field names
        audio_url = result["audio"]["url"]
        audio_bytes_io = await download_url_as_bytesio(audio_url)
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes_io.read()))


class ElevenLabsInstantVoiceClone(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsInstantVoiceClone",
            display_name="ElevenLabs Instant Voice Clone",
            category="api node/audio/ElevenLabs",
            description="Create a cloned voice from audio samples. "
            "Provide 1-8 audio recordings of the voice to clone.",
            inputs=[
                IO.Autogrow.Input(
                    "files",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Audio.Input("audio"),
                        prefix="audio",
                        min=1,
                        max=8,
                    ),
                    tooltip="Audio recordings for voice cloning.",
                ),
                IO.Boolean.Input(
                    "remove_background_noise",
                    default=False,
                    tooltip="Remove background noise from voice samples using audio isolation.",
                ),
            ],
            outputs=[
                IO.Custom(ELEVENLABS_VOICE).Output(display_name="voice"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        files: IO.Autogrow.Type,
        remove_background_noise: bool,
    ) -> IO.NodeOutput:
        audio_urls = []
        for key in files:
            audio = files[key]
            audio_data_np = audio_tensor_to_contiguous_ndarray(audio["waveform"])
            audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, audio["sample_rate"], "mp4", "aac")
            url = await upload_file_to_fal(audio_bytes_io, "audio/mp4")
            audio_urls.append(url)

        result = await fal_run(cls, FAL_ELEVENLABS_TTS, {
            "name": str(uuid.uuid4()),
            "remove_background_noise": remove_background_noise,
            "audio_urls": audio_urls,
        })
        # TODO: verify fal.ai field names
        return IO.NodeOutput(result["voice_id"])


ELEVENLABS_STS_VOICE_SETTINGS = [
    IO.Float.Input(
        "speed",
        default=1.0,
        min=0.7,
        max=1.3,
        step=0.01,
        display_mode=IO.NumberDisplay.slider,
        tooltip="Speech speed. 1.0 is normal, <1.0 slower, >1.0 faster.",
    ),
    IO.Float.Input(
        "similarity_boost",
        default=0.75,
        min=0.0,
        max=1.0,
        step=0.01,
        display_mode=IO.NumberDisplay.slider,
        tooltip="Similarity boost. Higher values make the voice more similar to the original.",
    ),
    IO.Boolean.Input(
        "use_speaker_boost",
        default=False,
        tooltip="Boost similarity to the original speaker voice.",
    ),
    IO.Float.Input(
        "style",
        default=0.0,
        min=0.0,
        max=0.2,
        step=0.01,
        display_mode=IO.NumberDisplay.slider,
        tooltip="Style exaggeration. Higher values increase stylistic expression but may reduce stability.",
    ),
]


class ElevenLabsSpeechToSpeech(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsSpeechToSpeech",
            display_name="ElevenLabs Speech to Speech",
            category="api node/audio/ElevenLabs",
            description="Transform speech from one voice to another while preserving the original content and emotion.",
            inputs=[
                IO.Custom(ELEVENLABS_VOICE).Input(
                    "voice",
                    tooltip="Target voice for the transformation. "
                    "Connect from Voice Selector or Instant Voice Clone.",
                ),
                IO.Audio.Input(
                    "audio",
                    tooltip="Source audio to transform.",
                ),
                IO.Float.Input(
                    "stability",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Voice stability. Lower values give broader emotional range, "
                    "higher values produce more consistent but potentially monotonous speech.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "eleven_multilingual_sts_v2",
                            ELEVENLABS_STS_VOICE_SETTINGS,
                        ),
                        IO.DynamicCombo.Option(
                            "eleven_english_sts_v2",
                            ELEVENLABS_STS_VOICE_SETTINGS,
                        ),
                    ],
                    tooltip="Model to use for speech-to-speech transformation.",
                ),
                IO.Combo.Input(
                    "output_format",
                    options=["mp3_44100_192", "opus_48000_192"],
                    tooltip="Audio output format.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    tooltip="Seed for reproducibility.",
                ),
                IO.Boolean.Input(
                    "remove_background_noise",
                    default=False,
                    tooltip="Remove background noise from input audio using audio isolation.",
                ),
            ],
            outputs=[
                IO.Audio.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        voice: str,
        audio: Input.Audio,
        stability: float,
        model: dict,
        output_format: str,
        seed: int,
        remove_background_noise: bool,
    ) -> IO.NodeOutput:
        audio_url = await _upload_audio_to_fal(audio)
        voice_settings = TextToSpeechVoiceSettings(
            stability=stability,
            similarity_boost=model["similarity_boost"],
            style=model["style"],
            use_speaker_boost=model["use_speaker_boost"],
            speed=model["speed"],
        )
        result = await fal_run(cls, FAL_ELEVENLABS_TTS, {
            "voice_id": voice,
            "audio_url": audio_url,
            "model_id": model["model"],
            "voice_settings": voice_settings.model_dump(exclude_none=True),
            "seed": seed,
            "remove_background_noise": remove_background_noise,
            "output_format": output_format,
        })
        # TODO: verify fal.ai field names
        output_url = result["audio"]["url"]
        audio_bytes_io = await download_url_as_bytesio(output_url)
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes_io.read()))


def _generate_dialogue_inputs(count: int) -> list:
    """Generate input widgets for a given number of dialogue entries."""
    inputs = []
    for i in range(1, count + 1):
        inputs.extend(
            [
                IO.String.Input(
                    f"text{i}",
                    multiline=True,
                    default="",
                    tooltip=f"Text content for dialogue entry {i}.",
                ),
                IO.Custom(ELEVENLABS_VOICE).Input(
                    f"voice{i}",
                    tooltip=f"Voice for dialogue entry {i}. Connect from Voice Selector or Instant Voice Clone.",
                ),
            ]
        )
    return inputs


class ElevenLabsTextToDialogue(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ElevenLabsTextToDialogue",
            display_name="ElevenLabs Text to Dialogue",
            category="api node/audio/ElevenLabs",
            description="Generate multi-speaker dialogue from text. Each dialogue entry has its own text and voice.",
            inputs=[
                IO.Float.Input(
                    "stability",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.5,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Voice stability. Lower values give broader emotional range, "
                    "higher values produce more consistent but potentially monotonous speech.",
                ),
                IO.Combo.Input(
                    "apply_text_normalization",
                    options=["auto", "on", "off"],
                    tooltip="Text normalization mode. 'auto' lets the system decide, "
                    "'on' always applies normalization, 'off' skips it.",
                ),
                IO.Combo.Input(
                    "model",
                    options=["eleven_v3"],
                    tooltip="Model to use for dialogue generation.",
                ),
                IO.DynamicCombo.Input(
                    "inputs",
                    options=[
                        IO.DynamicCombo.Option("1", _generate_dialogue_inputs(1)),
                        IO.DynamicCombo.Option("2", _generate_dialogue_inputs(2)),
                        IO.DynamicCombo.Option("3", _generate_dialogue_inputs(3)),
                        IO.DynamicCombo.Option("4", _generate_dialogue_inputs(4)),
                        IO.DynamicCombo.Option("5", _generate_dialogue_inputs(5)),
                        IO.DynamicCombo.Option("6", _generate_dialogue_inputs(6)),
                        IO.DynamicCombo.Option("7", _generate_dialogue_inputs(7)),
                        IO.DynamicCombo.Option("8", _generate_dialogue_inputs(8)),
                        IO.DynamicCombo.Option("9", _generate_dialogue_inputs(9)),
                        IO.DynamicCombo.Option("10", _generate_dialogue_inputs(10)),
                    ],
                    tooltip="Number of dialogue entries.",
                ),
                IO.String.Input(
                    "language_code",
                    default="",
                    tooltip="ISO-639-1 or ISO-639-3 language code (e.g., 'en', 'es', 'fra'). "
                    "Leave empty for automatic detection.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=4294967295,
                    tooltip="Seed for reproducibility.",
                ),
                IO.Combo.Input(
                    "output_format",
                    options=["mp3_44100_192", "opus_48000_192"],
                    tooltip="Audio output format.",
                ),
            ],
            outputs=[
                IO.Audio.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        stability: float,
        apply_text_normalization: str,
        model: str,
        inputs: dict,
        language_code: str,
        seed: int,
        output_format: str,
    ) -> IO.NodeOutput:
        num_entries = int(inputs["inputs"])
        dialogue_inputs = []
        for i in range(1, num_entries + 1):
            text = inputs[f"text{i}"]
            voice_id = inputs[f"voice{i}"]
            validate_string(text, min_length=1)
            dialogue_inputs.append({"text": text, "voice_id": voice_id})
        result = await fal_run(cls, FAL_ELEVENLABS_TTS, {
            "inputs": dialogue_inputs,
            "model_id": model,
            "language_code": language_code if language_code.strip() else None,
            "settings": {"stability": stability},
            "seed": seed,
            "apply_text_normalization": apply_text_normalization,
            "output_format": output_format,
        })
        # TODO: verify fal.ai field names
        audio_url = result["audio"]["url"]
        audio_bytes_io = await download_url_as_bytesio(audio_url)
        return IO.NodeOutput(audio_bytes_to_audio_input(audio_bytes_io.read()))


class ElevenLabsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ElevenLabsSpeechToText,
            ElevenLabsVoiceSelector,
            ElevenLabsTextToSpeech,
            ElevenLabsAudioIsolation,
            ElevenLabsTextToSoundEffects,
            ElevenLabsInstantVoiceClone,
            ElevenLabsSpeechToSpeech,
            ElevenLabsTextToDialogue,
        ]


async def comfy_entrypoint() -> ElevenLabsExtension:
    return ElevenLabsExtension()
