from __future__ import annotations

import collections
import logging
import operator
import os
import shutil
import sys
from collections.abc import Sequence, MutableSequence
from functools import reduce
from itertools import chain
from os.path import join
from pathlib import Path
from typing import List, Optional, Final, Set

import requests
import requests_cache
import tqdm
from huggingface_hub import dump_environment_info, hf_hub_download, scan_cache_dir, snapshot_download, HfFileSystem, CacheNotFound
from huggingface_hub.utils import GatedRepoError, LocalEntryNotFoundError
from requests import Session
from safetensors import safe_open
from safetensors.torch import save_file

from .cli_args import args
from .cmd import folder_paths
from .cmd.folder_paths import add_model_folder_path, supported_pt_extensions  # pylint: disable=import-error
from .component_model.deprecation import _deprecate_method
from .component_model.files import canonicalize_path
from .interruption import InterruptProcessingException
from .model_downloader_types import CivitFile, HuggingFile, CivitModelsGetResponse, CivitFile_, Downloadable, UrlFile, DownloadableFileList
from .utils import ProgressBar, comfy_tqdm

_session = Session()
_hf_fs = HfFileSystem()

logger = logging.getLogger(__name__)


def get_filename_list(folder_name: str) -> Sequence[str]:
    return get_filename_list_with_downloadable(folder_name)


def get_folder_paths(*args, **kwargs):
    return folder_paths.get_folder_paths(*args, **kwargs)


def get_filename_list_with_downloadable(folder_name: str, known_files: Optional[List[Downloadable] | KnownDownloadables] = None) -> DownloadableFileList | list[str]:
    if known_files is None:
        known_files = _get_known_models_for_folder_name(folder_name)

    # workaround for lora loading issue, still needs to be investigated
    if sys.platform == "nt":
        existing = frozenset(folder_paths.get_filename_list(folder_name))
        downloadable = frozenset() if args.disable_known_models else frozenset(str(f) for f in known_files)
        return list(map(canonicalize_path, sorted(list(existing | downloadable))))
    else:
        existing = folder_paths.get_filename_list(folder_name)

        downloadable_files = []
        if not args.disable_known_models:
            downloadable_files = known_files

        return DownloadableFileList(existing, downloadable_files)


def get_full_path_or_raise(folder_name: str, filename: str, known_files: Optional[List[Downloadable] | KnownDownloadables] = None) -> str:
    res = get_or_download(folder_name, filename, known_files=known_files)
    if res is None:
        raise FileNotFoundError(f"{folder_name} does not contain {filename}")
    return res


def get_full_path(folder_name: str, filename: str) -> Optional[str]:
    return get_or_download(folder_name, filename)


def get_or_download(folder_name: str, filename: str, known_files: Optional[List[Downloadable] | KnownDownloadables] = None) -> Optional[str]:
    if known_files is None:
        known_files = _get_known_models_for_folder_name(folder_name)

    filename = canonicalize_path(filename)
    path = folder_paths.get_full_path(folder_name, filename)

    candidate_str_match = False
    candidate_filename_match = False
    candidate_alternate_filenames_match = False
    candidate_save_filename_match = False
    if path is None and not args.disable_known_models:
        try:
            # todo: should this be the first or last path?
            this_model_directory = folder_paths.get_folder_paths(folder_name)[0]
            known_file: Optional[HuggingFile | CivitFile] = None
            for candidate in known_files:
                candidate_str_match = canonicalize_path(str(candidate)) == filename
                candidate_filename_match = canonicalize_path(candidate.filename) == filename
                candidate_alternate_filenames_match = filename in list(map(canonicalize_path, candidate.alternate_filenames))
                candidate_save_filename_match = filename == canonicalize_path(candidate.save_with_filename)
                if (candidate_str_match
                        or candidate_filename_match
                        or candidate_alternate_filenames_match
                        or candidate_save_filename_match):
                    known_file = candidate
                    break
            if known_file is None:
                logger.debug(f"get_or_download could not find {filename} in {folder_name}, known_files={known_files}")
                return path
            with comfy_tqdm() as watcher:
                if isinstance(known_file, HuggingFile):
                    if known_file.save_with_filename is not None:
                        linked_filename = known_file.save_with_filename
                    elif not known_file.force_save_in_repo_id and os.path.basename(known_file.filename) != known_file.filename:
                        linked_filename = os.path.basename(known_file.filename)
                    else:
                        linked_filename = known_file.filename

                    if known_file.force_save_in_repo_id or linked_filename is not None and os.path.dirname(known_file.filename) == "":
                        # if the known file has an overridden linked name, save it into a repo_id sub directory
                        # this deals with situations like
                        # jschoormans/controlnet-densepose-sdxl repo having diffusion_pytorch_model.safetensors
                        # it should be saved to controlnet-densepose-sdxl.safetensors
                        # since there are a bajillion diffusion_pytorch_model.safetensors, it should be downloaded by hf into jschoormans/controlnet-densepose-sdxl/diffusion_pytorch_model.safetensors
                        # then linked to the local folder to controlnet-densepose-sdxl.safetensors or some other canonical name
                        hf_destination_dir = os.path.join(this_model_directory, known_file.repo_id)
                    else:
                        hf_destination_dir = this_model_directory

                    # converted 16 bit files should be skipped
                    # todo: the file size should be replaced with a file hash
                    path = os.path.join(hf_destination_dir, known_file.filename)
                    try:
                        file_size = os.stat(path, follow_symlinks=True).st_size if os.path.isfile(path) else None
                    except:
                        file_size = None
                    if os.path.isfile(path) and file_size == known_file.size:
                        return path
                    # at this point, the file was not found with its candidate name
                    path = None

                    cache_hit = False
                    hf_hub_download_kwargs = dict(repo_id=known_file.repo_id,
                                  filename=known_file.filename,
                                  repo_type=known_file.repo_type,
                                  revision=known_file.revision,
                                  local_files_only=True,
                                  local_dir=hf_destination_dir if args.force_hf_local_dir_mode else None,
                                  token=True,
                                                  )

                    with requests_cache.disabled():
                        try:
                            # always retrieve this from the cache if it already exists there
                            path = hf_hub_download(**hf_hub_download_kwargs)
                            logger.debug(f"hf_hub_download cache hit for {known_file.repo_id}/{known_file.filename}")
                            cache_hit = True
                        except LocalEntryNotFoundError:
                            try:
                                logger.debug(f"{folder_name}/{filename} is being downloaded from {known_file.repo_id}/{known_file.filename} candidate_str_match={candidate_str_match} candidate_filename_match={candidate_filename_match} candidate_alternate_filenames_match={candidate_alternate_filenames_match} candidate_save_filename_match={candidate_save_filename_match}")
                                hf_hub_download_kwargs.pop("local_files_only")
                                path = hf_hub_download(**hf_hub_download_kwargs)
                            except requests.exceptions.HTTPError as exc_info:
                                if exc_info.response.status_code == 401:
                                    raise GatedRepoError(f"{known_file.repo_id}/{known_file.filename}", response=exc_info.response)
                            except IOError as exc_info:
                                logger.error(f"cannot reach huggingface {known_file.repo_id}/{known_file.filename}", exc_info=exc_info)
                            except Exception as exc_info:
                                logger.error(f"an exception occurred while downloading {known_file.repo_id}/{known_file.filename}. hf_hub_download kwargs={hf_hub_download_kwargs}", exc_info=exc_info)
                                dump_environment_info()
                                for key, value in os.environ.items():
                                    if key.startswith("HF_"):
                                        if key == "HF_TOKEN":
                                            value = "*****"
                                        print(f"{key}={value}", file=sys.stderr)

                    if path is not None and known_file.convert_to_16_bit and file_size is not None and file_size != 0:
                        tensors = {}
                        with safe_open(path, framework="pt") as f:
                            with tqdm.tqdm(total=len(f.keys())) as pb:
                                for k in f.keys():
                                    x = f.get_tensor(k)
                                    tensors[k] = x.half()
                                    del x
                                    pb.update()

                        # always save converted files to the destination so that the huggingface cache is not corrupted
                        save_file(tensors, os.path.join(hf_destination_dir, known_file.filename))

                        for _, v in tensors.items():
                            del v
                        logger.info(f"Converted {path} to 16 bit, size is {os.stat(path, follow_symlinks=True).st_size}")

                    link_successful = True
                    exc_info_link = {}
                    if path is not None:
                        if Path(linked_filename).is_absolute():
                            raise ValueError(f"{known_file.repo_id}/{known_file.filename} surprisingly was trying to link to an absolute path {linked_filename}, failing")

                        destination_link = Path(this_model_directory) / linked_filename
                        if destination_link.is_file():
                            logger.warning(f"{known_file.repo_id}/{known_file.filename} could not link to {destination_link} because the path already exists, which is unexpected")
                        else:
                            try:
                                # sometimes, linked filename has a path in it, on purpose, such as with controlnet_aux nodes
                                Path(destination_link).parent.mkdir(parents=True, exist_ok=True)
                                os.symlink(path, destination_link)
                            except FileExistsError:
                                # the download was resumed
                                pass
                            except Exception as exc_info:
                                exc_info_link = exc_info
                                logger.error("error while symbolic linking", exc_info=exc_info)
                                try:
                                    os.link(path, destination_link)
                                except Exception as hard_link_exc:
                                    logger.error("error while hard linking", exc_info=hard_link_exc)
                                    if cache_hit:
                                        shutil.copyfile(path, destination_link)
                                    link_successful = False
                                    exc_info_link = (exc_info, hard_link_exc)

                    if not link_successful:
                        logger.error(f"Failed to link file with alternative download save name in a way that is compatible with Hugging Face caching {repr(known_file)}. If cache_hit={cache_hit} is True, the file was copied into the destination. exc_info={exc_info_link}")
                else:
                    url: Optional[str] = None
                    save_filename = known_file.save_with_filename or known_file.filename

                    if isinstance(known_file, CivitFile):
                        model_info_res = _session.get(
                            f"https://civitai.com/api/v1/models/{known_file.model_id}?modelVersionId={known_file.model_version_id}")
                        model_info: CivitModelsGetResponse = model_info_res.json()

                        civit_file: CivitFile_
                        for civit_file in chain.from_iterable(version['files'] for version in model_info['modelVersions']):
                            if canonicalize_path(civit_file['name']) == filename:
                                url = civit_file['downloadUrl']
                                break
                    elif isinstance(known_file, UrlFile):
                        url = known_file.url
                    else:
                        raise RuntimeError("Unknown file type")

                    if url is None:
                        logger.warning(f"Could not retrieve file {str(known_file)}")
                    else:
                        destination_with_filename = join(this_model_directory, save_filename)
                        os.makedirs(os.path.dirname(destination_with_filename), exist_ok=True)
                        try:

                            with _session.get(url, stream=True, allow_redirects=True) as response:
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
        except GatedRepoError as exc_info:
            exc_info.append_to_message(f"""
Visit the repository, accept the terms, and then do one of the following:

 - Set the HF_TOKEN environment variable to your Hugging Face token; or,
 - Login to Hugging Face in your terminal using `huggingface-cli login`
""")
            raise exc_info
    return path


class KnownDownloadables(collections.UserList[Downloadable]):
    # we're not invoking the constructor because we want a reference to the passed list
    # noinspection PyMissingConstructor
    def __init__(self, data, folder_name: Optional[str | Sequence[str]] = None, folder_names: Optional[Sequence[str]] = None):
        # this should be a view
        self.data = data
        folder_names = folder_names or []
        if isinstance(folder_name, str):
            folder_names.append(folder_name)
        elif folder_name is not None and hasattr(folder_name, "__getitem__") and len(folder_name[0]) > 1:
            folder_names += folder_name
        self._folder_names = folder_names

    @property
    def folder_names(self) -> list[str]:
        return self._folder_names

    @folder_names.setter
    def folder_names(self, value: list[str]):
        self._folder_names = value

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._folder_names
        else:
            return item in self.data


KNOWN_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors"),
    HuggingFile("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors"),
    HuggingFile("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0.safetensors", show_in_ui=False),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_b.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_c.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stage_a.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly.safetensors"),
    HuggingFile("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors"),
    # from https://github.com/comfyanonymous/ComfyUI_examples/tree/master/2_pass_txt2img
    HuggingFile("stabilityai/stable-diffusion-2-1", "v2-1_768-ema-pruned.ckpt", show_in_ui=False),
    HuggingFile("waifu-diffusion/wd-1-5-beta3", "wd-illusion-fp16.safetensors", show_in_ui=False),
    HuggingFile("jomcs/NeverEnding_Dream-Feb19-2023", "CarDos Anime/cardosAnime_v10.safetensors", show_in_ui=False),
    # from https://github.com/comfyanonymous/ComfyUI_examples/blob/master/area_composition/README.md
    HuggingFile("ckpt/anything-v3.0", "Anything-V3.0.ckpt", show_in_ui=False),
    HuggingFile("stabilityai/cosxl", "cosxl.safetensors"),
    HuggingFile("stabilityai/cosxl", "cosxl_edit.safetensors"),
    # latest, popular civitai models
    CivitFile(133005, 357609, filename="juggernautXL_v9Rundiffusionphoto2.safetensors"),
    CivitFile(112902, 351306, filename="dreamshaperXL_v21TurboDPMSDE.safetensors"),
    CivitFile(139562, 344487, filename="realvisxlV40_v40Bakedvae.safetensors"),
    HuggingFile("SG161222/Realistic_Vision_V6.0_B1_noVAE", "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"),
    HuggingFile("SG161222/Realistic_Vision_V5.1_noVAE", "Realistic_Vision_V5.1_fp16-no-ema.safetensors"),
    HuggingFile("Lykon/DreamShaper", "DreamShaper_8_pruned.safetensors", save_with_filename="dreamshaper_8.safetensors", alternate_filenames=("DreamShaper_8_pruned.safetensors",)),
    CivitFile(7371, 425083, filename="revAnimated_v2Rebirth.safetensors"),
    CivitFile(4468, 57618, filename="counterfeitV30_v30.safetensors"),
    CivitFile(241415, 272376, filename="picxReal_10.safetensors"),
    CivitFile(23900, 95489, filename="anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "sd3_medium.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips_t5xxlfp8.safetensors"),
    HuggingFile("fal/AuraFlow", "aura_flow_0.1.safetensors"),
    # stable audio, # uses names from https://comfyanonymous.github.io/ComfyUI_examples/audio/
    HuggingFile("Comfy-Org/stable-audio-open-1.0_repackaged", "stable-audio-open-1.0.safetensors", alternate_filenames=("stable_audio_open_1.0.safetensors",)),
    # hunyuandit
    HuggingFile("comfyanonymous/hunyuan_dit_comfyui", "hunyuan_dit_1.0.safetensors"),
    HuggingFile("comfyanonymous/hunyuan_dit_comfyui", "hunyuan_dit_1.1.safetensors"),
    HuggingFile("comfyanonymous/hunyuan_dit_comfyui", "hunyuan_dit_1.2.safetensors"),
    HuggingFile("lllyasviel/flux1-dev-bnb-nf4", "flux1-dev-bnb-nf4.safetensors"),
    HuggingFile("lllyasviel/flux1-dev-bnb-nf4", "flux1-dev-bnb-nf4-v2.safetensors"),
    HuggingFile("silveroxides/flux1-nf4-weights", "flux1-schnell-bnb-nf4.safetensors"),
    HuggingFile("Lightricks/LTX-Video", "ltx-video-2b-v0.9.safetensors"),
    HuggingFile("Lightricks/LTX-Video", "ltx-video-2b-v0.9.1.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "all_in_one/lumina_2.safetensors"),
    HuggingFile("Comfy-Org/flux1-schnell", "flux1-schnell-fp8.safetensors"),
    HuggingFile("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors"),
    HuggingFile("stabilityai/stable-video-diffusion-img2vid", "svd.safetensors"),
    HuggingFile("stabilityai/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-2-inpainting", "v2-inpainting-pruned-ema.safetensors"),
    HuggingFile("runwayml/stable-diffusion-inpainting", "sd-v1-5-inpainting.ckpt", show_in_ui=False),
    HuggingFile("stabilityai/stable-diffusion-3.5-large", "sd3.5_large.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-medium", "sd3.5_medium.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-large-turbo", "sd3.5_large_turbo.safetensors"),
    HuggingFile("Comfy-Org/stable-diffusion-3.5-fp8", "sd3.5_large_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/stable-diffusion-3.5-fp8", "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"),
    HuggingFile("fal/AuraFlow-v0.2", "aura_flow_0.2.safetensors"),
    HuggingFile("lodestones/Chroma", "Chroma_v1.0.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "all_in_one/mochi_preview_fp8_scaled.safetensors"),
    HuggingFile("Lightricks/LTX-Video", "ltx-video-2b-v0.9.5.safetensors"),
    HuggingFile("Comfy-Org/ACE-Step_ComfyUI_repackaged", "all_in_one/ace_step_v1_3.5b.safetensors"),
    CivitFile(8714, 13359, filename="AOM2-Hard.safetensors"),
    CivitFile(4291, 132454, filename="AOM3A3.safetensors"),
    CivitFile(140737, 357037, filename="albedobaseXL_v21.safetensors"),
], folder_name="checkpoints")

KNOWN_UNCLIP_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_c.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-2-1-unclip", "sd21-unclip-h.ckpt"),
    HuggingFile("stabilityai/stable-diffusion-2-1-unclip", "sd21-unclip-l.ckpt"),
    HuggingFile("comfyanonymous/wd-1.5-beta2_unCLIP", "wd-1-5-beta2-aesthetic-unclip-h.safetensors"),
    HuggingFile("comfyanonymous/illuminatiDiffusionV1_v11_unCLIP", "illuminatiDiffusionV1_v11-unclip-h.safetensors"),
], folder_name="checkpoints")

KNOWN_IMAGE_ONLY_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("stabilityai/stable-zero123", "stable_zero123.ckpt")
], folder_name="checkpoints")

KNOWN_UPSCALERS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("lllyasviel/Annotators", "RealESRGAN_x4plus.pth")
], folder_name="upscale_models")

KNOWN_GLIGEN_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned.safetensors", show_in_ui=False),
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned_fp16.safetensors"),
], folder_name="gligen")

KNOWN_CLIP_VISION_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("comfyanonymous/clip_vision_g", "clip_vision_g.safetensors"),
    HuggingFile("Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/clip_vision/llava_llama3_vision.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors"),
], folder_name="clip_vision")

KNOWN_LORAS: Final[KnownDownloadables] = KnownDownloadables([
    CivitFile(model_id=211577, model_version_id=238349, filename="openxl_handsfix.safetensors"),
    CivitFile(model_id=324815, model_version_id=364137, filename="blur_control_xl_v1.safetensors"),
    CivitFile(model_id=47085, model_version_id=55199, filename="GoodHands-beta2.safetensors"),
    HuggingFile("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5", "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SDXL-12steps-CFG-lora.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SD15-12steps-CFG-lora.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Canny-dev-lora", "flux1-canny-dev-lora.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Depth-dev-lora", "flux1-depth-dev-lora.safetensors"),
    HuggingFile("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors", save_with_filename="lcm_lora_sdxl.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V1.0.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V2.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V1.0.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V1.1.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors", show_in_ui=False),
], folder_name="loras")

KNOWN_CONTROLNETS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("thibaud/controlnet-openpose-sdxl-1.0", "OpenPoseXL2.safetensors", convert_to_16_bit=True, size=2502139104),
    HuggingFile("thibaud/controlnet-openpose-sdxl-1.0", "control-lora-openposeXL2-rank256.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11e_sd15_ip2p_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11e_sd15_shuffle_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_canny_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_inpaint_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_lineart_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_mlsd_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_openpose_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_scribble_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_seg_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_softedge_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15s2_lineart_anime_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11e_sd15_ip2p_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11e_sd15_shuffle_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1e_sd15_tile_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1p_sd15_depth_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_canny_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_inpaint_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_lineart_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_mlsd_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_normalbae_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_openpose_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_scribble_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_seg_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_softedge_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15s2_lineart_anime_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11u_sd15_tile_fp16.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_canny_full.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_canny_mid.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_canny_small.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_depth_full.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_depth_mid.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_depth_small.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "ioclab_sd15_recolor.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_blur.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_blur_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_blur_anime_beta.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_canny.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_canny_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_depth.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_depth_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_openpose_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_openpose_anime_v2.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_scribble_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_canny_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_canny_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_depth_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_depth_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_recolor_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_recolor_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_sketch_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_sketch_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_depth.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_depth_faid_vidit.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_depth_zeed.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_softedge.safetensors"),
    HuggingFile("SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe", "depth-zoe-xl-v1.0-controlnet.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_canny.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_depth_midas.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_depth_zoe.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_lineart.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_openpose.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_sketch.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_xl_canny.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_xl_openpose.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_xl_sketch.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "thibaud_xl_openpose.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "thibaud_xl_openpose_256lora.safetensors"),
    HuggingFile("jschoormans/controlnet-densepose-sdxl", "diffusion_pytorch_model.safetensors", save_with_filename="controlnet-densepose-sdxl.safetensors", convert_to_16_bit=True, size=2502139104),
    HuggingFile("stabilityai/stable-cascade", "controlnet/canny.safetensors", save_with_filename="stable_cascade_canny.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "controlnet/inpainting.safetensors", save_with_filename="stable_cascade_inpainting.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "controlnet/super_resolution.safetensors", save_with_filename="stable_cascade_super_resolution.safetensors"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/canny/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ControlNet-Plus-Plus_sd15_canny.safetensors", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/depth/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ControlNet-Plus-Plus_sd15_grayscale_depth.safetensors", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/hed/controlnet/diffusion_pytorch_model.bin", save_with_filename="ControlNet-Plus-Plus_sd15_hed.bin", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/lineart/controlnet/diffusion_pytorch_model.bin", save_with_filename="ControlNet-Plus-Plus_sd15_lineart.bin", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/seg/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ControlNet-Plus-Plus_sd15_ade20k_seg.safetensors", repo_type="space"),
    HuggingFile("xinsir/controlnet-scribble-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-scribble-sdxl-1.0.safetensors"),
    HuggingFile("xinsir/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-canny-sdxl-1.0.safetensors"),
    HuggingFile("xinsir/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model_V2.safetensors", save_with_filename="xinsir-controlnet-canny-sdxl-1.0_V2.safetensors"),
    HuggingFile("xinsir/controlnet-openpose-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-openpose-sdxl-1.0.safetensors"),
    HuggingFile("xinsir/anime-painter", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-anime-painter-scribble-sdxl-1.0.safetensors"),
    HuggingFile("TheMistoAI/MistoLine", "mistoLine_rank256.safetensors"),
    HuggingFile("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors", save_with_filename="xinsir-controlnet-union-sdxl-1.0-promax.safetensors"),
    HuggingFile("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-union-sdxl-1.0.safetensors"),
    HuggingFile("InstantX/FLUX.1-dev-Controlnet-Canny", "diffusion_pytorch_model.safetensors", save_with_filename="instantx-flux.1-dev-controlnet-canny.safetensors"),
    HuggingFile("InstantX/FLUX.1-dev-Controlnet-Union", "diffusion_pytorch_model.safetensors", save_with_filename="instantx-flux.1-dev-controlnet-union.safetensors"),
    HuggingFile("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", "diffusion_pytorch_model.safetensors", save_with_filename="shakker-labs-flux.1-dev-controlnet-union-pro.safetensors"),
    HuggingFile("TheMistoAI/MistoLine_Flux.dev", "mistoline_flux.dev_v1.safetensors"),
    HuggingFile("XLabs-AI/flux-controlnet-collections", "flux-canny-controlnet-v3.safetensors"),
    HuggingFile("XLabs-AI/flux-controlnet-collections", "flux-depth-controlnet-v3.safetensors"),
    HuggingFile("XLabs-AI/flux-controlnet-collections", "flux-hed-controlnet-v3.safetensors"),
    HuggingFile("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", "diffusion_pytorch_model.safetensors", save_with_filename="alimama-creative-flux.1-dev-controlnet-inpainting-alpha.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_canny.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_depth.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_blur.safetensors"),
    HuggingFile("Shakker-Labs/FLUX.1-dev-ControlNet-Depth", "diffusion_pytorch_model.safetensors", save_with_filename="shakker-labs-flux.1-dev-controlnet-depth.safetensors"),
], folder_name="controlnet")

KNOWN_DIFF_CONTROLNETS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_canny_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_depth_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_hed_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_mlsd_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_normal_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_openpose_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_scribble_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_seg_fp16.safetensors"),
], folder_name="diff_controlnet")

KNOWN_APPROX_VAES: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("madebyollin/taesd", "taesd_decoder.safetensors", show_in_ui=False),
    HuggingFile("madebyollin/taesd", "taesd_encoder.safetensors", show_in_ui=False),
    HuggingFile("madebyollin/taesdxl", "taesdxl_decoder.safetensors", show_in_ui=False),
    HuggingFile("madebyollin/taesdxl", "taesdxl_encoder.safetensors", show_in_ui=False),
    # todo: these are both the encoder and decoder, so it is not clear what should be done here
    # HuggingFile("madebyollin/taef1", "diffusion_pytorch_model.safetensors", save_with_filename="taef1_decoder.safetensors", show_in_ui=False),
    # HuggingFile("madebyollin/taesd3", "diffusion_pytorch_model.safetensors", save_with_filename="taesd3_decoder.safetensors", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_decoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesdxl_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesdxl_decoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd3_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd3_decoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taef1_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taef1_decoder.pth", show_in_ui=False),
    # todo: update this with the video VAEs
], folder_name="vae_approx")

KNOWN_VAES: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("stabilityai/sdxl-vae", "sdxl_vae.safetensors"),
    HuggingFile("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors"),
    # this is the flux VAE
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/vae/ae.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/vae/mochi_vae.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/vae/hunyuan_video_vae_bf16.safetensors"),
    HuggingFile("comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI", "vae/cosmos_cv8x8x8_1.0.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/vae/ae.safetensors", save_with_filename="lumina_image_2.0-ae.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
    # Flux 2
    HuggingFile("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
    # Z Image Turbo
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", save_with_filename="z_image_turbo_vae.safetensors"),
    # Hunyuan Image
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/vae/hunyuan_image_refiner_vae_fp16.safetensors"),
], folder_name="vae")

KNOWN_HUGGINGFACE_MODEL_REPOS: Final[Set[str]] = {
    'JingyeChen22/textdiffuser2_layout_planner',
    'JingyeChen22/textdiffuser2-full-ft',
    'microsoft/Phi-4-mini-instruct',
    'llava-hf/llava-v1.6-mistral-7b-hf',
    'facebook/nllb-200-distilled-1.3B',
    'THUDM/chatglm3-6b',
    'roborovski/superprompt-v1',
    'Qwen/Qwen2-VL-7B-Instruct',
    'microsoft/Florence-2-large-ft',
    'google/paligemma2-10b-pt-896',
    'google/paligemma2-28b-pt-896',
    'google/paligemma-3b-ft-refcoco-seg-896',
    'microsoft/phi-4',
    'appmana/Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf',
    'llava-hf/llava-onevision-qwen2-7b-si-hf',
    'llava-hf/llama3-llava-next-8b-hf',
    'PromptEnhancer/PromptEnhancer-32B',
}

KNOWN_UNET_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet-Comfyui.fp16.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Fill-dev", "flux1-fill-dev.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Canny-dev", "flux1-canny-dev.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Depth-dev", "flux1-depth-dev.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Kontext-dev", "flux1-kontext-dev.safetensors"),
    HuggingFile("Kijai/flux-fp8", "flux1-dev-fp8.safetensors"),
    HuggingFile("Kijai/flux-fp8", "flux1-schnell-fp8.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/diffusion_models/mochi_preview_bf16.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/diffusion_models/mochi_preview_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-14B-Text2World.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-14B-Video2World.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-7B-Text2World.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-7B-Video2World.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/diffusion_models/lumina_2_model_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/diffusion_models/hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_dev_bf16.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_full_fp16.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_e1_full_bf16.safetensors"),
    HuggingFile("Comfy-Org/Cosmos_Predict2_repackaged", "cosmos_predict2_2B_t2i.safetensors"),
    HuggingFile("Comfy-Org/Cosmos_Predict2_repackaged", "cosmos_predict2_14B_t2i.safetensors"),
    HuggingFile("Comfy-Org/Cosmos_Predict2_repackaged", "cosmos_predict2_2B_video2world_480p_16fps.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"),
    HuggingFile("lodestones/Chroma", "chroma-unlocked-v37.safetensors"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"),
    HuggingFile("city96/Qwen-Image-gguf", "qwen-image-Q4_K_M.gguf"),
    HuggingFile("city96/Qwen-Image-gguf", "qwen-image-Q8_0.gguf"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "non_official/diffusion_models/qwen_image_distill_full_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"),
    # Flux 2
    HuggingFile("Comfy-Org/flux2-dev", "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"),
    # Z Image Turbo
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    # Omnigen 2
    HuggingFile("Comfy-Org/Omnigen2_ComfyUI_repackaged", "split_files/diffusion_models/omnigen2_fp16.safetensors"),
    # Hunyuan Image
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/diffusion_models/hunyuanimage2.1_bf16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/diffusion_models/hunyuanimage2.1_refiner_bf16.safetensors"),
    # Ovis
    HuggingFile("Comfy-Org/Ovis-Image", "split_files/diffusion_models/ovis_image_bf16.safetensors"),
], folder_names=["diffusion_models", "unet"])
KNOWN_CLIP_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    # todo: is this correct?
    HuggingFile("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    HuggingFile("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp8_scaled.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "text_encoders/clip_g.safetensors"),
    HuggingFile("comfyanonymous/flux_text_encoders", "clip_l.safetensors", save_with_filename="clip_l.safetensors"),
    # uses names from https://comfyanonymous.github.io/ComfyUI_examples/audio/
    HuggingFile("google-t5/t5-base", "model.safetensors", save_with_filename="t5_base.safetensors"),
    HuggingFile("zer0int/CLIP-GmP-ViT-L-14", "ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors"),
    HuggingFile("comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI", "text_encoders/oldt5_xxl_fp16.safetensors"),
    HuggingFile("comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI", "text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/text_encoders/gemma_2_2b_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/clip_l_hidream.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/clip_g_hidream.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    # Flux 2
    HuggingFile("Comfy-Org/flux2-dev", "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors"),
    HuggingFile("Comfy-Org/flux2-dev", "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors"),
    # Z Image Turbo
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"),
    # Omnigen 2
    HuggingFile("Comfy-Org/Omnigen2_ComfyUI_repackaged", "split_files/text_encoders/qwen_2.5_vl_fp16.safetensors"),
    # Hunyuan Image
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
    HuggingFile("Comfy-Org/Ovis-Image", "split_files/text_encoders/ovis_2.5.safetensors"),
], folder_names=["clip", "text_encoders"])

KNOWN_STYLE_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("black-forest-labs/FLUX.1-Redux-dev", "flux1-redux-dev.safetensors"),
], folder_name="style_models")

_known_models_db: list[KnownDownloadables] = [
    KNOWN_CHECKPOINTS,
    KNOWN_VAES,
    KNOWN_LORAS,
    KNOWN_UNET_MODELS,
    KNOWN_APPROX_VAES,
    KNOWN_DIFF_CONTROLNETS,
    KNOWN_CLIP_MODELS,
    KNOWN_CLIP_VISION_MODELS,
    KNOWN_CONTROLNETS,
    KNOWN_GLIGEN_MODELS,
    KNOWN_IMAGE_ONLY_CHECKPOINTS,
    KNOWN_UNCLIP_CHECKPOINTS,
    KNOWN_UPSCALERS,
    KNOWN_STYLE_MODELS,
]


def _is_known_model_in_models_db(obj: list[Downloadable] | KnownDownloadables):
    return any(candidate is obj or candidate.data is obj for candidate in _known_models_db)


def _get_known_models_for_folder_name(folder_name: str) -> List[Downloadable]:
    return list(chain.from_iterable([candidate for candidate in _known_models_db if folder_name in candidate]))


def add_known_models(folder_name: str, known_models: KnownDownloadables | Optional[List[Downloadable]] | Downloadable = None, *models: Downloadable) -> MutableSequence[Downloadable]:
    if isinstance(known_models, Downloadable):
        models = [known_models] + list(models) or []
        known_models = None

    if known_models is None:
        try:
            known_models = next(candidate for candidate in _known_models_db if folder_name in candidate)
        except StopIteration:
            add_model_folder_path(folder_name, extensions=supported_pt_extensions)
            known_models = KnownDownloadables([], folder_name=folder_name)

    # check if any of the pre-existing known models already reference this list
    if not _is_known_model_in_models_db(known_models):
        if not isinstance(known_models, KnownDownloadables):
            # wrap it
            known_models = KnownDownloadables(known_models)
        # meets protocol at this point
        _known_models_db.append(known_models)

    if len(models) < 1:
        return known_models

    if args.disable_known_models:
        logger.warning(f"Known models have been disabled in the options (while adding {folder_name}/{','.join(map(str, models))})")

    pre_existing = frozenset(known_models)
    known_models.extend([model for model in models if model not in pre_existing])
    return known_models


@_deprecate_method(version="1.0.0", message="use get_huggingface_repo_list instead")
def huggingface_repos() -> List[str]:
    return get_huggingface_repo_list()


def get_huggingface_repo_list(*extra_cache_dirs: str) -> List[str]:
    if len(extra_cache_dirs) == 0:
        extra_cache_dirs = folder_paths.get_folder_paths("huggingface_cache")

    # all in cache directories
    try:
        default_cache_dir = [scan_cache_dir()]
    except CacheNotFound as exc_info:
        default_cache_dir = []
    existing_repo_ids = frozenset(
        cache_item.repo_id for cache_item in \
        reduce(operator.or_,
               map(lambda cache_info: cache_info.repos, default_cache_dir + [scan_cache_dir(cache_dir=cache_dir) for cache_dir in extra_cache_dirs if os.path.isdir(cache_dir)]))
        if cache_item.repo_type == "model" or cache_item.repo_type == "space"
    )

    # also check local-dir style directories
    existing_local_dir_repos = set()
    local_dirs = folder_paths.get_folder_paths("huggingface")
    for local_dir_root in local_dirs:
        # enumerate all the two-directory paths
        if not os.path.isdir(local_dir_root):
            continue

        for user_dir in Path(local_dir_root).iterdir():
            for model_dir in user_dir.iterdir():
                existing_local_dir_repos.add(f"{user_dir.name}/{model_dir.name}")

    known_repo_ids = frozenset(KNOWN_HUGGINGFACE_MODEL_REPOS)
    if args.disable_known_models:
        return list(existing_repo_ids | existing_local_dir_repos)
    else:
        return list(existing_repo_ids | existing_local_dir_repos | known_repo_ids)


def get_or_download_huggingface_repo(repo_id: str, cache_dirs: Optional[list] = None, local_dirs: Optional[list] = None, force: bool = False, subset: bool = False) -> Optional[str]:
    with comfy_tqdm():
        return _get_or_download_huggingface_repo(repo_id, cache_dirs, local_dirs, force=force, subset=subset)


def _get_or_download_huggingface_repo(repo_id: str, cache_dirs: Optional[list] = None, local_dirs: Optional[list] = None, force: bool = False, subset: bool = False) -> Optional[str]:
    cache_dirs = cache_dirs or folder_paths.get_folder_paths("huggingface_cache")
    local_dirs = local_dirs or folder_paths.get_folder_paths("huggingface")
    cache_dirs_snapshots, local_dirs_snapshots = _get_cache_hits(cache_dirs, local_dirs, repo_id, subset=subset)

    local_dirs_cache_hit = len(local_dirs_snapshots) > 0
    cache_dirs_cache_hit = len(cache_dirs_snapshots) > 0
    logger.debug(f"cache {'hit' if local_dirs_cache_hit or cache_dirs_cache_hit else 'miss'} for repo_id={repo_id} because local_dirs={local_dirs_cache_hit}, cache_dirs={cache_dirs_cache_hit}")

    # if we're in forced local directory mode, only use the local dir snapshots, and otherwise, download
    if args.force_hf_local_dir_mode:
        # todo: we still have to figure out a way to download things to the right places by default
        if len(local_dirs_snapshots) > 0 and not force:
            return local_dirs_snapshots[0]
        elif not args.disable_known_models:
            destination = os.path.join(local_dirs[0], repo_id)
            logger.debug(f"downloading repo_id={repo_id}, local_dir={destination}")
            return snapshot_download(repo_id, local_dir=destination, force_download=force)

    snapshots = local_dirs_snapshots + cache_dirs_snapshots
    if len(snapshots) > 0 and not force:
        return snapshots[0]
    elif not args.disable_known_models:
        logger.debug(f"downloading repo_id={repo_id}")
        return snapshot_download(repo_id, force_download=force)

    # this repo was not found
    return None


def _get_cache_hits(cache_dirs: Sequence[str], local_dirs: Sequence[str], repo_id, subset=False):
    local_dirs_snapshots = []
    cache_dirs_snapshots = []
    # find all the pre-existing downloads for this repo_id
    try:
        repo_files = set(_hf_fs.ls(repo_id, detail=False))
    except:
        repo_files = []

    if len(repo_files) > 0:
        for local_dir in local_dirs:
            local_path = Path(local_dir) / repo_id
            local_files = frozenset(f"{repo_id}/{f.relative_to(local_path)}" for f in local_path.rglob("*") if f.is_file())
            # fix path representation
            local_files = frozenset(f.replace("\\", "/") for f in local_files)
            # remove .huggingface
            local_files = frozenset(f for f in local_files if not f.startswith(f"{repo_id}/.huggingface") and not f.startswith(f"{repo_id}/.cache"))
            if len(local_files) > 0 and ((subset and local_files.issubset(repo_files)) or (not subset and repo_files.issubset(local_files))):
                local_dirs_snapshots.append(str(local_path))
    else:
        # an empty repository or unknown repository info, trust that if the directory exists, it matches
        for local_dir in local_dirs:
            local_path = Path(local_dir) / repo_id
            if local_path.is_dir():
                local_dirs_snapshots.append(str(local_path))

    for cache_dir in (None, *cache_dirs):
        try:
            cache_dirs_snapshots.append(snapshot_download(repo_id, local_files_only=True, cache_dir=cache_dir))
        except FileNotFoundError:
            continue
        except:
            continue
    return cache_dirs_snapshots, local_dirs_snapshots


def _delete_repo_from_huggingface_cache(repo_id: str, cache_dir: Optional[str] = None) -> List[str]:
    results = scan_cache_dir(cache_dir)
    matching = [repo for repo in results.repos if repo.repo_id == repo_id]
    if len(matching) == 0:
        return []
    revisions: List[str] = []
    for repo in matching:
        for revision_info in repo.revisions:
            revisions.append(revision_info.commit_hash)
    results.delete_revisions(*revisions).execute()
    return revisions
