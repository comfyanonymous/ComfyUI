from __future__ import annotations

import json
import os
import random
from io import BytesIO

import av
import numpy as np
import torchaudio
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

import folder_paths

# used for image preview
from comfy.cli_args import args
from comfy_api.v3.io import ComfyNodeV3, FolderType, Image, _UIOutput


class SavedResult(dict):
    def __init__(self, filename: str, subfolder: str, type: FolderType):
        super().__init__(filename=filename, subfolder=subfolder,type=type.value)

    @property
    def filename(self) -> str:
        return self["filename"]

    @property
    def subfolder(self) -> str:
        return self["subfolder"]

    @property
    def type(self) -> FolderType:
        return FolderType(self["type"])


class PreviewImage(_UIOutput):
    def __init__(self, image: Image.Type, animated: bool=False, cls: ComfyNodeV3=None, **kwargs):
        output_dir = folder_paths.get_temp_directory()
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        filename_prefix = "ComfyUI" + prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, image[0].shape[1], image[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(image):
            img = PILImage.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata and cls is not None:
                metadata = PngInfo()
                if cls.hidden.prompt is not None:
                    metadata.add_text("prompt", json.dumps(cls.hidden.prompt))
                if cls.hidden.extra_pnginfo is not None:
                    for x in cls.hidden.extra_pnginfo:
                        metadata.add_text(x, json.dumps(cls.hidden.extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=1)
            results.append(SavedResult(file, subfolder, FolderType.temp))
            counter += 1

        self.values = results
        self.animated = animated

    def as_dict(self):
        return {
            "images": self.values,
            "animated": (self.animated,)
        }


class PreviewMask(PreviewImage):
    def __init__(self, mask: PreviewMask.Type, animated: bool=False, cls: ComfyNodeV3=None, **kwargs):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        super().__init__(preview, animated, cls, **kwargs)


# class UILatent(_UIOutput):
#     def __init__(self, values: list[SavedResult | dict], **kwargs):
#         output_dir = folder_paths.get_temp_directory()
#         type = "temp"
#         prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
#         compress_level = 1
#         filename_prefix = "ComfyUI"


#         full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

#         # support save metadata for latent sharing
#         prompt_info = ""
#         if prompt is not None:
#             prompt_info = json.dumps(prompt)

#         metadata = None
#         if not args.disable_metadata:
#             metadata = {"prompt": prompt_info}
#             if extra_pnginfo is not None:
#                 for x in extra_pnginfo:
#                     metadata[x] = json.dumps(extra_pnginfo[x])

#         file = f"{filename}_{counter:05}_.latent"

#         results: list[FileLocator] = []
#         results.append({
#             "filename": file,
#             "subfolder": subfolder,
#             "type": "output"
#         })

#         file = os.path.join(full_output_folder, file)

#         output = {}
#         output["latent_tensor"] = samples["samples"].contiguous()
#         output["latent_format_version_0"] = torch.tensor([])

#         comfy.utils.save_torch_file(output, file, metadata=metadata)

#         self.values = values

#     def as_dict(self):
#         return {
#             "latents": self.values,
#         }


class PreviewAudio(_UIOutput):
    def __init__(self, audio, cls: ComfyNodeV3=None, **kwargs):
        quality = "128k"
        format = "flac"

        filename_prefix = "ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_temp_directory()
        )

        # Prepare metadata dictionary
        metadata = {}
        if not args.disable_metadata and cls is not None:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        # Opus supported sample rates
        OPUS_RATES = [8000, 12000, 16000, 24000, 48000]
        results = []
        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.{format}"
            output_path = os.path.join(full_output_folder, file)

            # Use original sample rate initially
            sample_rate = audio["sample_rate"]

            # Handle Opus sample rate requirements
            if format == "opus":
                if sample_rate > 48000:
                    sample_rate = 48000
                elif sample_rate not in OPUS_RATES:
                    # Find the next highest supported rate
                    for rate in sorted(OPUS_RATES):
                        if rate > sample_rate:
                            sample_rate = rate
                            break
                    if sample_rate not in OPUS_RATES:  # Fallback if still not supported
                        sample_rate = 48000

                # Resample if necessary
                if sample_rate != audio["sample_rate"]:
                    waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)

            # Create output with specified format
            output_buffer = BytesIO()
            output_container = av.open(output_buffer, mode='w', format=format)

            # Set metadata on the container
            for key, value in metadata.items():
                output_container.metadata[key] = value

            # Set up the output stream with appropriate properties
            if format == "opus":
                out_stream = output_container.add_stream("libopus", rate=sample_rate)
                if quality == "64k":
                    out_stream.bit_rate = 64000
                elif quality == "96k":
                    out_stream.bit_rate = 96000
                elif quality == "128k":
                    out_stream.bit_rate = 128000
                elif quality == "192k":
                    out_stream.bit_rate = 192000
                elif quality == "320k":
                    out_stream.bit_rate = 320000
            elif format == "mp3":
                out_stream = output_container.add_stream("libmp3lame", rate=sample_rate)
                if quality == "V0":
                    # TODO i would really love to support V3 and V5 but there doesn't seem to be a way to set the qscale level, the property below is a bool
                    out_stream.codec_context.qscale = 1
                elif quality == "128k":
                    out_stream.bit_rate = 128000
                elif quality == "320k":
                    out_stream.bit_rate = 320000
            else:  # format == "flac":
                out_stream = output_container.add_stream("flac", rate=sample_rate)

            frame = av.AudioFrame.from_ndarray(waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format='flt',
                                               layout='mono' if waveform.shape[0] == 1 else 'stereo')
            frame.sample_rate = sample_rate
            frame.pts = 0
            output_container.mux(out_stream.encode(frame))

            # Flush encoder
            output_container.mux(out_stream.encode(None))

            # Close containers
            output_container.close()

            # Write the output to file
            output_buffer.seek(0)
            with open(output_path, 'wb') as f:
                f.write(output_buffer.getbuffer())

            results.append(SavedResult(file, subfolder, FolderType.temp))
            counter += 1

        self.values = results

    def as_dict(self):
        return {"audio": self.values}


class PreviewVideo(_UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values

    def as_dict(self):
        return {"images": self.values, "animated": (True,)}


class PreviewUI3D(_UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values

    def as_dict(self):
        return {"3d": self.values}


class PreviewText(_UIOutput):
    def __init__(self, value: str, **kwargs):
        self.value = value

    def as_dict(self):
        return {"text": (self.value,)}
