from __future__ import annotations

import logging
import os
from itertools import chain
from os.path import join
from typing import List, Any, Optional, Union

from huggingface_hub import hf_hub_download
from requests import Session

from .cmd import folder_paths
from .model_downloader_types import CivitFile, HuggingFile, CivitModelsGetResponse, CivitFile_
from .interruption import InterruptProcessingException
from .utils import ProgressBar, comfy_tqdm

session = Session()


def get_filename_list_with_downloadable(folder_name: str, known_files: List[Any]) -> List[str]:
    existing = frozenset(folder_paths.get_filename_list(folder_name))
    downloadable = frozenset(str(f) for f in known_files)
    return sorted(list(existing | downloadable))


def get_or_download(folder_name: str, filename: str, known_files: List[HuggingFile | CivitFile]) -> Optional[str]:
    path = folder_paths.get_full_path(folder_name, filename)

    if path is None:
        try:
            # todo: should this be the first or last path?
            destination = folder_paths.get_folder_paths(folder_name)[0]
            known_file: Optional[HuggingFile | CivitFile] = None
            for candidate in known_files:
                if candidate.filename == filename or filename in candidate.alternate_filenames or filename == candidate.save_with_filename:
                    known_file = candidate
                    break
            if known_file is None:
                return path
            with comfy_tqdm():
                if isinstance(known_file, HuggingFile):
                    save_filename = known_file.save_with_filename or known_file.filename
                    path = hf_hub_download(repo_id=known_file.repo_id,
                                           filename=save_filename,
                                           local_dir=destination,
                                           resume_download=True)
                else:
                    url: Optional[str] = None
                    save_filename = known_file.save_with_filename or known_file.filename

                    if isinstance(known_file, CivitFile):
                        model_info_res = session.get(
                            f"https://civitai.com/api/v1/models/{known_file.model_id}?modelVersionId={known_file.model_version_id}")
                        model_info: CivitModelsGetResponse = model_info_res.json()

                        file: CivitFile_
                        for file in chain.from_iterable(version['files'] for version in model_info['modelVersions']):
                            if file['name'] == filename:
                                url = file['downloadUrl']
                                break
                    else:
                        raise RuntimeError("unknown file type")

                    if url is None:
                        logging.warning(f"Could not retrieve file {str(known_file)}")
                    else:
                        destination_with_filename = join(destination, save_filename)
                        try:

                            with session.get(url, stream=True, allow_redirects=True) as response:
                                total_size = int(response.headers.get("content-length", 0))
                                progress_bar = ProgressBar(total=total_size)
                                with open(destination_with_filename, "wb") as file:
                                    for chunk in response.iter_content(chunk_size=512 * 1024):
                                        progress_bar.update(len(chunk))
                                        file.write(chunk)
                        except InterruptProcessingException:
                            os.remove(destination_with_filename)

                        path = folder_paths.get_full_path(folder_name, filename)
                        assert path is not None
        except StopIteration:
            pass
        except Exception as exc:
            logging.error("Error while trying to download a file", exc_info=exc)
        finally:
            # a path was found for any reason, so we should invalidate the cache
            if path is not None:
                folder_paths.invalidate_cache(folder_name)
    return path


KNOWN_CHECKPOINTS = [
    HuggingFile("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors"),
    HuggingFile("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors"),
    HuggingFile("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0.safetensors", show_in_ui=False),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_b.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_c.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stage_a.safetensors"),
    HuggingFile("runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"),
    HuggingFile("runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.ckpt", show_in_ui=False),
    HuggingFile("runwayml/stable-diffusion-v1-5", "v1-5-pruned.ckpt", show_in_ui=False),
    HuggingFile("runwayml/stable-diffusion-v1-5", "v1-5-pruned.safetensors", show_in_ui=False),
    # from https://github.com/comfyanonymous/ComfyUI_examples/tree/master/2_pass_txt2img
    HuggingFile("stabilityai/stable-diffusion-2-1", "v2-1_768-ema-pruned.ckpt", show_in_ui=False),
    HuggingFile("waifu-diffusion/wd-1-5-beta3", "wd-illusion-fp16.safetensors", show_in_ui=False),
    HuggingFile("jomcs/NeverEnding_Dream-Feb19-2023", "CarDos Anime/cardosAnime_v10.safetensors", show_in_ui=False),
    # from https://github.com/comfyanonymous/ComfyUI_examples/blob/master/area_composition/README.md
    HuggingFile("ckpt/anything-v3.0", "Anything-V3.0.ckpt", show_in_ui=False),
    # latest, popular civitai models
    CivitFile(133005, 357609, filename="juggernautXL_v9Rundiffusionphoto2.safetensors"),
    CivitFile(112902, 351306, filename="dreamshaperXL_v21TurboDPMSDE.safetensors"),
    CivitFile(139562, 344487, filename="realvisxlV40_v40Bakedvae.safetensors"),
]

KNOWN_UNCLIP_CHECKPOINTS = [
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_c.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-2-1-unclip", "sd21-unclip-h.ckpt"),
    HuggingFile("stabilityai/stable-diffusion-2-1-unclip", "sd21-unclip-l.ckpt"),
]

KNOWN_IMAGE_ONLY_CHECKPOINTS = [
    HuggingFile("stabilityai/stable-zero123", "stable_zero123.ckpt")
]

KNOWN_UPSCALERS = [
    HuggingFile("lllyasviel/Annotators", "RealESRGAN_x4plus.pth")
]

KNOWN_GLIGEN_MODELS = [
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned.safetensors"),
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned_fp16.safetensors"),
]

KNOWN_CLIP_VISION_MODELS = [
    HuggingFile("comfyanonymous/clip_vision_g", "clip_vision_g.safetensors")
]

KNOWN_LORAS = [
    CivitFile(model_id=211577, model_version_id=238349, filename="openxl_handsfix.safetensors"),
]


def add_known_models(folder_name: str, symbol: List[Union[CivitFile, HuggingFile]], *models: Union[CivitFile, HuggingFile]) -> List[Union[CivitFile, HuggingFile]]:
    symbol += models
    folder_paths.invalidate_cache(folder_name)
    return symbol
