import torchaudio
import torch
import comfy.model_management
import folder_paths
import os

class EmptyLatentAudio:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "_for_testing/audio"

    def generate(self):
        batch_size = 1
        latent = torch.zeros([batch_size, 64, 1024], device=self.device)
        return ({"samples":latent, "type": "audio"}, )

class VAEEncodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "_for_testing/audio"

    def encode(self, vae, audio):
        t = vae.encode(audio["waveform"].movedim(1, -1))
        return ({"samples":t}, )

class VAEDecodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"

    CATEGORY = "_for_testing/audio"

    def decode(self, vae, samples):
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        return ({"waveform": audio, "sample_rate": 44100}, )

class SaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                              "filename_prefix": ("STRING", {"default": "audio/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "_for_testing/audio"

    def save_audio(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()
        for (batch_number, waveform) in enumerate(audio["waveform"]):
            #TODO: metadata
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"
            torchaudio.save(os.path.join(full_output_folder, file), waveform, audio["sample_rate"], format="FLAC")
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "audio": results } }

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"audio": [sorted(files), ]}, }

    CATEGORY = "_for_testing/audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        multiplier = 1.0
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

    @classmethod
    def IS_CHANGED(s, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True

NODE_CLASS_MAPPINGS = {
    "EmptyLatentAudio": EmptyLatentAudio,
    "VAEEncodeAudio": VAEEncodeAudio,
    "VAEDecodeAudio": VAEDecodeAudio,
    "SaveAudio": SaveAudio,
    "LoadAudio": LoadAudio,
}
