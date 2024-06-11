from __future__ import annotations

import logging
import os
from itertools import chain
from os.path import join
from typing import List, Any, Optional, Union

import tqdm
from huggingface_hub import hf_hub_download, scan_cache_dir
from requests import Session
from safetensors import safe_open
from safetensors.torch import save_file

from .cli_args import args
from .cmd import folder_paths
from .interruption import InterruptProcessingException
from .model_downloader_types import CivitFile, HuggingFile, CivitModelsGetResponse, CivitFile_
from .utils import ProgressBar, comfy_tqdm

_session = Session()


def get_filename_list_with_downloadable(folder_name: str, known_files: List[Any]) -> List[str]:
    existing = frozenset(folder_paths.get_filename_list(folder_name))
    downloadable = frozenset() if args.disable_known_models else frozenset(str(f) for f in known_files if not isinstance(f, HuggingFile) or f.show_in_ui)
    return sorted(list(existing | downloadable))


def get_or_download(folder_name: str, filename: str, known_files: List[HuggingFile | CivitFile]) -> Optional[str]:
    path = folder_paths.get_full_path(folder_name, filename)

    if path is None and not args.disable_known_models:
        try:
            # todo: should this be the first or last path?
            this_model_directory = folder_paths.get_folder_paths(folder_name)[0]
            known_file: Optional[HuggingFile | CivitFile] = None
            for candidate in known_files:
                if str(candidate) == filename or candidate.filename == filename or filename in candidate.alternate_filenames or filename == candidate.save_with_filename:
                    known_file = candidate
                    break
            if known_file is None:
                return path
            with comfy_tqdm():
                if isinstance(known_file, HuggingFile):
                    if known_file.save_with_filename is not None:
                        linked_filename = known_file.save_with_filename
                    elif not known_file.force_save_in_repo_id and os.path.basename(known_file.filename) != known_file.filename:
                        linked_filename = os.path.basename(known_file.filename)
                    else:
                        linked_filename = None

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
                    path = os.path.join(hf_destination_dir, known_file.filename)
                    try:
                        file_size = os.stat(path, follow_symlinks=True).st_size if os.path.isfile(path) else None
                    except:
                        file_size = None
                    if os.path.isfile(path) and file_size == known_file.size:
                        return path

                    path = hf_hub_download(repo_id=known_file.repo_id,
                                           filename=known_file.filename,
                                           local_dir=hf_destination_dir,
                                           repo_type=known_file.repo_type,
                                           )

                    if known_file.convert_to_16_bit and file_size is not None and file_size != 0:
                        tensors = {}
                        with safe_open(path, framework="pt") as f:
                            with tqdm.tqdm(total=len(f.keys())) as pb:
                                for k in f.keys():
                                    x = f.get_tensor(k)
                                    tensors[k] = x.half()
                                    del x
                                    pb.update()

                        save_file(tensors, path)

                        for _, v in tensors.items():
                            del v
                        logging.info(f"Converted {path} to 16 bit, size is {os.stat(path, follow_symlinks=True).st_size}")

                    try:
                        if linked_filename is not None:
                            os.symlink(os.path.join(hf_destination_dir, known_file.filename), os.path.join(this_model_directory, linked_filename))
                    except Exception as exc_info:
                        logging.error(f"Failed to link file with alternative download save name in a way that is compatible with Hugging Face caching {repr(known_file)}", exc_info=exc_info)
                else:
                    url: Optional[str] = None
                    save_filename = known_file.save_with_filename or known_file.filename

                    if isinstance(known_file, CivitFile):
                        model_info_res = _session.get(
                            f"https://civitai.com/api/v1/models/{known_file.model_id}?modelVersionId={known_file.model_version_id}")
                        model_info: CivitModelsGetResponse = model_info_res.json()

                        civit_file: CivitFile_
                        for civit_file in chain.from_iterable(version['files'] for version in model_info['modelVersions']):
                            if civit_file['name'] == filename:
                                url = civit_file['downloadUrl']
                                break
                    else:
                        raise RuntimeError("unknown file type")

                    if url is None:
                        logging.warning(f"Could not retrieve file {str(known_file)}")
                    else:
                        destination_with_filename = join(this_model_directory, save_filename)
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
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stage_a.safetensors", show_in_ui=False),
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
    HuggingFile("stabilityai/cosxl", "cosxl.safetensors"),
    HuggingFile("stabilityai/cosxl", "cosxl_edit.safetensors"),
    # latest, popular civitai models
    CivitFile(133005, 357609, filename="juggernautXL_v9Rundiffusionphoto2.safetensors"),
    CivitFile(112902, 351306, filename="dreamshaperXL_v21TurboDPMSDE.safetensors"),
    CivitFile(139562, 344487, filename="realvisxlV40_v40Bakedvae.safetensors"),
    HuggingFile("SG161222/Realistic_Vision_V6.0_B1_noVAE","Realistic_Vision_V6.0_NV_B1_fp16.safetensors"),
    HuggingFile("SG161222/Realistic_Vision_V5.1_noVAE","Realistic_Vision_V5.1_fp16-no-ema.safetensors"),
    CivitFile(4384, 128713, filename="dreamshaper_8.safetensors"),
    CivitFile(7371, 425083, filename="revAnimated_v2Rebirth.safetensors"),
    CivitFile(4468, 57618, filename="counterfeitV30_v30.safetensors"),
    CivitFile(241415, 272376, filename="picxReal_10.safetensors"),
    CivitFile(23900, 95489, filename="anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"),
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
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned.safetensors", show_in_ui=False),
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned_fp16.safetensors"),
]

KNOWN_CLIP_VISION_MODELS = [
    HuggingFile("comfyanonymous/clip_vision_g", "clip_vision_g.safetensors")
]

KNOWN_LORAS = [
    CivitFile(model_id=211577, model_version_id=238349, filename="openxl_handsfix.safetensors"),
    CivitFile(model_id=324815, model_version_id=364137, filename="blur_control_xl_v1.safetensors"),
    CivitFile(model_id=47085, model_version_id=55199, filename="GoodHands-beta2.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SDXL-12steps-CFG-lora.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SD15-12steps-CFG-lora.safetensors"),
]

KNOWN_CONTROLNETS = [
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
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/canny/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ft_controlnet_sd15_canny_res512_bs256_lr1e-5_warmup100_iter5k_fp16ft0-1000.safetensors"),
    # not clear if this should use black to white or some other depth rendering
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/depth/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ft_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_iter5k_fp16ft0-200.safetensors"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/hed/controlnet/diffusion_pytorch_model.bin", save_with_filename="reward_ft5k_controlnet_sd15_hed_res512_bs256_lr1e-5_warmup100_scale-1_iter5k_fp16_train0-1k_reward0-200.bin"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/lineart/controlnet/diffusion_pytorch_model.bin", save_with_filename="reward_ft5k_controlnet_sd15_lineart_res512_bs256_lr1e-5_warmup100_scale-10_iter5k_fp16_train0-1k_reward0-200.bin"),
    # should be used with ADE20K
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/seg/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_iter5k_fp16.safetensors"),
]

KNOWN_DIFF_CONTROLNETS = [
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_canny_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_depth_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_hed_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_mlsd_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_normal_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_openpose_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_scribble_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_seg_fp16.safetensors"),
]

KNOWN_APPROX_VAES = [
    HuggingFile("madebyollin/taesd", "taesd_decoder.safetensors"),
    HuggingFile("madebyollin/taesdxl", "taesdxl_decoder.safetensors"),
]

KNOWN_VAES = [
    HuggingFile("stabilityai/sdxl-vae", "sdxl_vae.safetensors"),
    HuggingFile("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors"),
]

KNOWN_HUGGINGFACE_MODEL_REPOS = {
    "JingyeChen22/textdiffuser2_layout_planner",
    'JingyeChen22/textdiffuser2-full-ft',
    "microsoft/Phi-3-mini-4k-instruct",
    "llava-hf/llava-v1.6-mistral-7b-hf"
}

KNOWN_UNET_MODELS: List[Union[CivitFile | HuggingFile]] = [
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet-Comfyui.fp16.safetensors")
]

KNOWN_CLIP_MODELS: List[Union[CivitFile | HuggingFile]] = []


def add_known_models(folder_name: str, symbol: List[Union[CivitFile, HuggingFile]], *models: Union[CivitFile, HuggingFile]) -> List[Union[CivitFile, HuggingFile]]:
    if args.disable_known_models:
        logging.warning(f"Known models have been disabled in the options (while adding {folder_name}/{','.join(map(str, models))})")
    symbol += models
    folder_paths.invalidate_cache(folder_name)
    return symbol


def huggingface_repos() -> List[str]:
    cache_info = scan_cache_dir()
    existing_repo_ids = frozenset(cache_item.repo_id for cache_item in cache_info.repos if cache_item.repo_type == "model")
    known_repo_ids = frozenset(KNOWN_HUGGINGFACE_MODEL_REPOS)
    return list(existing_repo_ids | known_repo_ids)
