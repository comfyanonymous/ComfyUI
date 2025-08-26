from .wav2vec2 import Wav2Vec2Model
import comfy.model_management
import comfy.ops
import comfy.utils
import logging
import torchaudio


class AudioEncoderModel():
    def __init__(self, config):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        self.model = Wav2Vec2Model(dtype=self.dtype, device=offload_device, operations=comfy.ops.manual_cast)
        self.model.eval()
        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)
        self.model_sample_rate = 16000

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def get_sd(self):
        return self.model.state_dict()

    def encode_audio(self, audio, sample_rate):
        comfy.model_management.load_model_gpu(self.patcher)
        audio = torchaudio.functional.resample(audio, sample_rate, self.model_sample_rate)
        out, all_layers = self.model(audio.to(self.load_device))
        outputs = {}
        outputs["encoded_audio"] = out
        outputs["encoded_audio_all_layers"] = all_layers
        return outputs


def load_audio_encoder_from_sd(sd, prefix=""):
    audio_encoder = AudioEncoderModel(None)
    sd = comfy.utils.state_dict_prefix_replace(sd, {"wav2vec2.": ""})
    m, u = audio_encoder.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing audio encoder: {}".format(m))

    return audio_encoder
