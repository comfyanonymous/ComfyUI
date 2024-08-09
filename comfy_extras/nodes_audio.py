import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import io
import json
import struct
import random
import hashlib
from comfy.cli_args import args

class EmptyLatentAudio:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"seconds": ("FLOAT", {"default": 47.6, "min": 1.0, "max": 1000.0, "step": 0.1})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/audio"

    def generate(self, seconds):
        batch_size = 1
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=self.device)
        return ({"samples":latent, "type": "audio"}, )

class VAEEncodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/audio"

    def encode(self, vae, audio):
        sample_rate = audio["sample_rate"]
        if 44100 != sample_rate:
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
        else:
            waveform = audio["waveform"]

        t = vae.encode(waveform.movedim(1, -1))
        return ({"samples":t}, )

class VAEDecodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"

    CATEGORY = "latent/audio"

    def decode(self, vae, samples):
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        return ({"waveform": audio, "sample_rate": 44100}, )


def create_vorbis_comment_block(comment_dict, last_block):
    vendor_string = b'ComfyUI'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    user_comment_list_length = len(comments)
    user_comments = b''.join(comments)

    comment_data = struct.pack('<I', vendor_length) + vendor_string + struct.pack('<I', user_comment_list_length) + user_comments
    if last_block:
        id = b'\x84'
    else:
        id = b'\x04'
    comment_block = id + struct.pack('>I', len(comment_data))[1:] + comment_data

    return comment_block

def insert_or_replace_vorbis_comment(flac_io, comment_dict):
    if len(comment_dict) == 0:
        return flac_io

    flac_io.seek(4)

    blocks = []
    last_block = False

    while not last_block:
        header = flac_io.read(4)
        last_block = (header[0] & 0x80) != 0
        block_type = header[0] & 0x7F
        block_length = struct.unpack('>I', b'\x00' + header[1:])[0]
        block_data = flac_io.read(block_length)

        if block_type == 4 or block_type == 1:
            pass
        else:
            header = bytes([(header[0] & (~0x80))]) + header[1:]
            blocks.append(header + block_data)

    blocks.append(create_vorbis_comment_block(comment_dict, last_block=True))

    new_flac_io = io.BytesIO()
    new_flac_io.write(b'fLaC')
    for block in blocks:
        new_flac_io.write(block)

    new_flac_io.write(flac_io.read())
    return new_flac_io


class SaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                              "filename_prefix": ("STRING", {"default": "audio/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "audio"

    def save_audio(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

            buff = insert_or_replace_vorbis_comment(buff, metadata)

            with open(os.path.join(full_output_folder, file), 'wb') as f:
                f.write(buff.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "audio": results } }

class PreviewAudio(SaveAudio):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

class LoadAudio:
    SUPPORTED_FORMATS = None

    @classmethod
    def get_supported_formats(s):
        if s.SUPPORTED_FORMATS is None:
            supported = set()
            available_backends = torchaudio.list_audio_backends()
            backend_formats = {
                "soundfile": ['.aiff', '.au', '.avr', '.caf', '.flac', '.htk', '.svx', '.mat4', '.mat5', '.mpc2k', '.mp3', '.ogg', '.paf', '.pvf', '.raw', '.rf64', '.sd2', '.sds', '.ircam', '.voc', '.w64', '.wav', '.nist', '.wavex', '.wve', '.xi'],
                "sox": ['.8svx', '.aif', '.aifc', '.aiff', '.aiffc', '.al', '.amb', '.amr-nb', '.amr-wb', '.anb', '.au', '.avr', '.awb', '.caf', '.cdda', '.cdr', '.cvs', '.cvsd', '.cvu', '.dat', '.dvms', '.f32', '.f4', '.f64', '.f8', '.fap', '.flac', '.fssd', '.gsm', '.gsrt', '.hcom', '.htk', '.ima', '.ircam', '.la', '.lpc', '.lpc10', '.lu', '.mat', '.mat4', '.mat5', '.maud', '.nist', '.ogg', '.paf', '.prc', '.pvf', '.raw', '.s1', '.s16', '.s2', '.s24', '.s3', '.s32', '.s4', '.s8', '.sb', '.sd2', '.sds', '.sf', '.sl', '.sln', '.smp', '.snd', '.sndfile', '.sndr', '.sndt', '.sou', '.sox', '.sph', '.sw', '.txw', '.u1', '.u16', '.u2', '.u24', '.u3', '.u32', '.u4', '.u8', '.ub', '.ul', '.uw', '.vms', '.voc', '.vorbis', '.vox', '.w64', '.wav', '.wavpcm', '.wv', '.wve', '.xa', '.xi'],
                "ffmpeg": ['.3dostr', '.4xm', '.aa', '.aac', '.aax', '.ace', '.acm', '.act', '.adf', '.adp', '.ads', '.aea', '.afc', '.aix', '.alias_pix', '.amrnb', '.amrwb', '.anm', '.apac', '.apc', '.ape', '.aqtitle', '.argo_brp', '.asf_o', '.av1', '.avr', '.avs', '.bethsoftvid', '.bfi', '.bfstm', '.bin', '.bink', '.binka', '.bitpacked', '.bmp_pipe', '.bmv', '.boa', '.bonk', '.brender_pix', '.brstm', '.c93', '.cdg', '.cdxl', '.cine', '.concat', '.cri_pipe', '.dcstr', '.dds_pipe', '.derf', '.dfa', '.dhav', '.dpx_pipe', '.dsf', '.dsicin', '.dss', '.dtshd', '.dvbsub', '.dvbtxt', '.dxa', '.ea', '.ea_cdata', '.epaf', '.exr_pipe', '.flic', '.frm', '.fsb', '.fwse', '.g729', '.gdv', '.gem_pipe', '.genh', '.gif_pipe', '.hca', '.hcom', '.hdr_pipe', '.hnm', '.idcin', '.idf', '.iec61883', '.iff', '.ifv', '.imf', '.ingenient', '.ipmovie', '.ipu', '.iss', '.iv8', '.ivr', '.j2k_pipe', '.jack', '.jpeg_pipe', '.jpegls_pipe', '.jpegxl_pipe', '.jv', '.kmsgrab', '.kux', '.laf', '.lavfi', '.libcdio', '.libdc1394', '.libgme', '.libopenmpt', '.live_flv', '.lmlm4', '.loas', '.luodat', '.lvf', '.lxf', '.matroska', '.webm', '.mca', '.mcc', '.mgsts', '.mjpeg_2000', '.mlv', '.mm', '.mods', '.moflex', '.mov', '.mp4', '.m4a', '.3gp', '.3g2', '.mj2', '.mpc', '.mpc8', '.mpegtsraw', '.mpegvideo', '.mpl2', '.mpsub', '.msf', '.msnwctcp', '.msp', '.mtaf', '.mtv', '.musx', '.mv', '.mvi', '.mxg', '.nc', '.nistsphere', '.nsp', '.nsv', '.nuv', '.openal', '.paf', '.pam_pipe', '.pbm_pipe', '.pcx_pipe', '.pfm_pipe', '.pgm_pipe', '.pgmyuv_pipe', '.pgx_pipe', '.phm_pipe', '.photocd_pipe', '.pictor_pipe', '.pjs', '.pmp', '.png_pipe', '.pp_bnk', '.ppm_pipe', '.psd_pipe', '.psxstr', '.pva', '.pvf', '.qcp', '.qdraw_pipe', '.qoi_pipe', '.r3d', '.realtext', '.redspark', '.rka', '.rl2', '.rpl', '.rsd', '.s337m', '.sami', '.sbg', '.scd', '.sdns', '.sdp', '.sdr2', '.sds', '.sdx', '.ser', '.sga', '.sgi_pipe', '.shn', '.siff', '.simbiosis_imx', '.sln', '.smk', '.smush', '.sol', '.stl', '.subviewer', '.subviewer1', '.sunrast_pipe', '.svag', '.svg_pipe', '.svs', '.tak', '.tedcaptions', '.thp', '.tiertexseq', '.tiff_pipe', '.tmv', '.tty', '.txd', '.ty', '.v210', '.v210x', '.vag', '.vbn_pipe', '.vividas', '.vivo', '.vmd', '.vobsub', '.vpk', '.vplayer', '.vqf', '.wady', '.wavarc', '.wc3movie', '.webp_pipe', '.wsd', '.wsvqa', '.wve', '.x11grab', '.xa', '.xbin', '.xbm_pipe', '.xmd', '.xmv', '.xpm_pipe', '.xvag', '.xwd_pipe', '.xwma', '.yop'],
            }

            for backend, formats in backend_formats.items():
                if backend in available_backends:
                    supported.update(formats)

            s.SUPPORTED_FORMATS = tuple(supported)
        return s.SUPPORTED_FORMATS

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        supported_formats = s.get_supported_formats()
        files = [
            f for f in os.listdir(input_dir)
            if (os.path.isfile(os.path.join(input_dir, f))
                and f.endswith(supported_formats)
            )
        ]
        return {"required": {"audio": (sorted(files), {"audio_upload": True})}}

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
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
    "PreviewAudio": PreviewAudio,
}
