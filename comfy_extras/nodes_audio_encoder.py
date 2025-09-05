import folder_paths
import comfy.audio_encoders.audio_encoders
import comfy.utils


class AudioEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio_encoder_name": (folder_paths.get_filename_list("audio_encoders"), ),
                             }}
    RETURN_TYPES = ("AUDIO_ENCODER",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, audio_encoder_name):
        audio_encoder_name = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder_name)
        sd = comfy.utils.load_torch_file(audio_encoder_name, safe_load=True)
        audio_encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(sd)
        if audio_encoder is None:
            raise RuntimeError("ERROR: audio encoder file is invalid and does not contain a valid model.")
        return (audio_encoder,)


class AudioEncoderEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio_encoder": ("AUDIO_ENCODER",),
                              "audio": ("AUDIO",),
                             }}
    RETURN_TYPES = ("AUDIO_ENCODER_OUTPUT",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, audio_encoder, audio):
        output = audio_encoder.encode_audio(audio["waveform"], audio["sample_rate"])
        return (output,)


NODE_CLASS_MAPPINGS = {
    "AudioEncoderLoader": AudioEncoderLoader,
    "AudioEncoderEncode": AudioEncoderEncode,
}
