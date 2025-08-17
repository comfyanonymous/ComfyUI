import os
import warnings

import torch
from segment_anything import SamPredictor

from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from collections import namedtuple
import numpy as np
from PIL import ImageOps, Image

import nodes
import comfy_extras.nodes_upscale_model as model_upscale
from server import PromptServer
import comfy
import impact.wildcards as wildcards
import math
import cv2
import time
from comfy import model_management
from impact import utils
from impact import impact_sampling
from concurrent.futures import ThreadPoolExecutor
import inspect
from collections import OrderedDict
import torch.nn.functional as F
import logging
import sys
import importlib


is_sam2_available = importlib.util.find_spec("sam2")
sam2_unavailable_message = f"\n----------------------------------------------------------------------------\n[Impact Pack] The SAM2 functionality is unavailable because the `facebook/sam2` dependency is not installed.\n\nInstallation command:\n{sys.executable} -m pip install git+https://github.com/facebookresearch/sam2\n----------------------------------------------------------------------------\n"
if is_sam2_available:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
else:
    logging.warning(sam2_unavailable_message)

try:
    from comfy_extras import nodes_differential_diffusion
except Exception:
    logging.warning("\n#############################################\n[Impact Pack] ComfyUI is an outdated version.\n#############################################\n")
    raise Exception("[Impact Pack] ComfyUI is an outdated version.")


SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

pb_id_cnt = time.time()
preview_bridge_image_id_map = {}
preview_bridge_image_name_map = {}

preview_bridge_cache = {}
preview_bridge_last_mask_cache = {}

current_prompt = None

SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma']


def is_execution_model_version_supported():
    try:
        import comfy_execution  # noqa: F401
        return True
    except Exception:
        return False


def set_previewbridge_image(node_id, file, item):
    global pb_id_cnt

    if file in preview_bridge_image_name_map:
        pb_id = preview_bridge_image_name_map[node_id, file]
        if pb_id.startswith(f"${node_id}"):
            return pb_id

    pb_id = f"${node_id}-{pb_id_cnt}"
    preview_bridge_image_id_map[pb_id] = (file, item)
    preview_bridge_image_name_map[node_id, file] = (pb_id, item)
    if os.path.isfile(file):
        i = Image.open(file)
        i = ImageOps.exif_transpose(i)
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
            preview_bridge_last_mask_cache[node_id] = mask.unsqueeze(0)
    pb_id_cnt += 1

    return pb_id


def erosion_mask(mask, grow_mask_by):
    mask = utils.make_2d_mask(mask)

    w = mask.shape[1]
    h = mask.shape[0]

    device = comfy.model_management.get_torch_device()
    mask = mask.clone().to(device)
    mask2 = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(w, h), mode="bilinear").to(device)
    if grow_mask_by == 0:
        mask_erosion = mask2
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by)).to(device)
        padding = math.ceil((grow_mask_by - 1) / 2)

        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask2.round(), kernel_tensor, padding=padding), 0, 1)

    return mask_erosion[:, :, :w, :h].round().cpu()


# CREDIT: https://github.com/BlenderNeko/ComfyUI_Noise/blob/afb14757216257b12268c91845eac248727a55e2/nodes.py#L68
#         https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    dims = low.shape

    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high

    return res.reshape(dims)


def mix_noise(from_noise, to_noise, strength, variation_method):
    if variation_method == 'slerp':
        mixed_noise = slerp(strength, from_noise, to_noise)
    else:
        # linear
        mixed_noise = (1 - strength) * from_noise + strength * to_noise

        # NOTE: Since the variance of the Gaussian noise in mixed_noise has changed, it must be corrected through scaling.
        scale_factor = math.sqrt((1 - strength) ** 2 + strength ** 2)
        mixed_noise /= scale_factor

    return mixed_noise


class REGIONAL_PROMPT:
    def __init__(self, mask, sampler, variation_seed=0, variation_strength=0.0, variation_method='linear'):
        mask = utils.make_2d_mask(mask)

        self.mask = mask
        self.sampler = sampler
        self.mask_erosion = None
        self.erosion_factor = None
        self.variation_seed = variation_seed
        self.variation_strength = variation_strength
        self.variation_method = variation_method

    def clone_with_sampler(self, sampler):
        rp = REGIONAL_PROMPT(self.mask, sampler)
        rp.mask_erosion = self.mask_erosion
        rp.erosion_factor = self.erosion_factor
        rp.variation_seed = self.variation_seed
        rp.variation_strength = self.variation_strength
        rp.variation_method = self.variation_method
        return rp

    def get_mask_erosion(self, factor):
        if self.mask_erosion is None or self.erosion_factor != factor:
            self.mask_erosion = erosion_mask(self.mask, factor)
            self.erosion_factor = factor

        return self.mask_erosion

    def touch_noise(self, noise):
        if self.variation_strength > 0.0:
            mask = utils.make_3d_mask(self.mask)
            mask = utils.resize_mask(mask, (noise.shape[2], noise.shape[3])).unsqueeze(0)

            regional_noise = Noise_RandomNoise(self.variation_seed).generate_noise({'samples': noise})
            mixed_noise = mix_noise(noise, regional_noise, self.variation_strength, variation_method=self.variation_method)

            return (mask == 1).float() * mixed_noise + (mask == 0).float() * noise

        return noise


class NO_BBOX_DETECTOR:
    pass


class NO_SEGM_DETECTOR:
    pass


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def gen_detection_hints_from_mask_area(x, y, mask, threshold, use_negative):
    mask = utils.make_2d_mask(mask)

    points = []
    plabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(mask.shape[0] / 20))
    x_step = max(3, int(mask.shape[1] / 20))

    for i in range(0, len(mask), y_step):
        for j in range(0, len(mask[i]), x_step):
            if mask[i][j] > threshold:
                points.append((x + j, y + i))
                plabs.append(1)
            elif use_negative and mask[i][j] == 0:
                points.append((x + j, y + i))
                plabs.append(0)

    return points, plabs


def gen_negative_hints(w, h, x1, y1, x2, y2):
    npoints = []
    nplabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(w / 20))
    x_step = max(3, int(h / 20))

    for i in range(10, h - 10, y_step):
        for j in range(10, w - 10, x_step):
            if not (x1 - 10 <= j and j <= x2 + 10 and y1 - 10 <= i and i <= y2 + 10):
                npoints.append((j, i))
                nplabs.append(0)

    return npoints, nplabs


def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name,
                   scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                   wildcard_opt=None, wildcard_opt_concat_mode=None,
                   detailer_hook=None,
                   refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                   refiner_negative=None, control_net_wrapper=None, cycle=1,
                   inpaint_model=False, noise_mask_feather=0, scheduler_func=None,
                   vae_tiled_encode=False, vae_tiled_decode=False):

    if noise_mask is not None:
        noise_mask = utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

        if noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().apply(model)[0]

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = wildcards.process_with_loras(wildcard_opt, model, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive
            positive = [positive[0].copy()]
            if 'pooled_output' in wildcard_positive[0][1]:
                positive[0][1]['pooled_output'] = wildcard_positive[0][1]['pooled_output']
            elif 'pooled_output' in positive[0][1]:
                del positive[0][1]['pooled_output']

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        logging.info("Detailer: segment skip (enough big)")
        return None, None

    if guide_size_for_bbox:  # == "bbox"
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        # for cropped_size
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    # safeguard
    if 'aitemplate_keep_loaded' in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            logging.info(f"Detailer: segment skip [determined upscale factor={upscale}]")
            return None, None

        if new_w == 0 or new_h == 0:
            logging.info(f"Detailer: segment skip [zero size={new_w, new_h}]")
            return None, None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            logging.info("Detailer: force inpaint")
            upscale = 1.0
            new_w = w
            new_h = h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    logging.info(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    # upscale
    upscaled_image = utils.tensor_resize(image, new_w, new_h)

    if detailer_hook is not None:
        upscaled_image = detailer_hook.post_upscale(upscaled_image, noise_mask)

    cnet_pils = None
    if control_net_wrapper is not None:
        positive, negative, cnet_pils = control_net_wrapper.apply(positive, negative, upscaled_image, noise_mask)
        model, cnet_pils2 = control_net_wrapper.doit_ipadapter(model)
        cnet_pils.extend(cnet_pils2)

    # prepare mask
    if detailer_hook is None or not detailer_hook.get_skip_sampling():
        if noise_mask is not None and inpaint_model:
            imc_encode = nodes.InpaintModelConditioning().encode
            if 'noise_mask' in inspect.signature(imc_encode).parameters:
                positive, negative, latent_image = imc_encode(positive, negative, upscaled_image, vae, mask=noise_mask, noise_mask=True)
            else:
                logging.warning("[Impact Pack] ComfyUI is an outdated version.")
                positive, negative, latent_image = imc_encode(positive, negative, upscaled_image, vae, noise_mask)
        else:
            latent_image = utils.to_latent_image(upscaled_image, vae, vae_tiled_encode=vae_tiled_encode)
            if noise_mask is not None:
                latent_image['noise_mask'] = noise_mask

        if detailer_hook is not None:
            latent_image = detailer_hook.post_encode(latent_image)

        refined_latent = latent_image

        sampler_opt=None
        if detailer_hook is not None:
            sampler_opt = detailer_hook.get_custom_sampler()

        # ksampler
        for i in range(0, cycle):
            if detailer_hook is not None:
                if detailer_hook is not None:
                    detailer_hook.set_steps((i, cycle))

                refined_latent = detailer_hook.cycle_latent(refined_latent)

                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = \
                    detailer_hook.pre_ksample(model, seed+i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
                noise, is_touched = detailer_hook.get_custom_noise(seed+i, torch.zeros(latent_image['samples'].size()), is_touched=False)
                if not is_touched:
                    noise = None
            else:
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, _, denoise2 = \
                    model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
                noise = None

            refined_latent = impact_sampling.ksampler_wrapper(model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2,
                                                              refined_latent, denoise2, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative,
                                                              noise=noise, scheduler_func=scheduler_func, sampler_opt=sampler_opt)

        if detailer_hook is not None:
            refined_latent = detailer_hook.pre_decode(refined_latent)

        # non-latent downscale - latent downscale cause bad quality
        start = time.time()
        if vae_tiled_decode:
            (refined_image,) = nodes.VAEDecodeTiled().decode(vae, refined_latent, 512) # using default settings
            logging.info(f"[Impact Pack] vae decoded (tiled) in {time.time() - start:.1f}s")
        else:
            try:
                refined_image = vae.decode(refined_latent['samples'])
            except Exception:
                # usually an out-of-memory exception from the decode, so try a tiled approach
                logging.warning(f"[Impact Pack] failed after {time.time() - start:.1f}s, doing vae.decode_tiled 64...")
                refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64, )
            logging.info(f"[Impact Pack] vae decoded in {time.time() - start:.1f}s")
    else:
        # skipped
        refined_image = upscaled_image

    if detailer_hook is not None:
        refined_image = detailer_hook.post_decode(refined_image)

    # downscale

    # workaround: support WAN as an i2i model
    if len(refined_image.shape) == 5:
        refined_image = refined_image.squeeze(0)

    refined_image = utils.tensor_resize(refined_image, w, h)

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image, cnet_pils


def enhance_detail_for_animatediff(image_frames, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                                   sampler_name,
                                   scheduler, positive, negative, denoise, noise_mask,
                                   wildcard_opt=None, wildcard_opt_concat_mode=None,
                                   detailer_hook=None,
                                   refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                                   refiner_negative=None, control_net_wrapper=None, noise_mask_feather=0, scheduler_func=None):
    if noise_mask is not None:
        noise_mask = utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

    if noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
        model = nodes_differential_diffusion.DifferentialDiffusion().apply(model)[0]

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = wildcards.process_with_loras(wildcard_opt, model, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive

    h = image_frames.shape[1]
    w = image_frames.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if guide_size_for_bbox:  # == "bbox"
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        # for cropped_size
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    # safeguard
    if 'aitemplate_keep_loaded' in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if upscale <= 1.0 or new_w == 0 or new_h == 0:
        logging.info("Detailer: force inpaint")
        upscale = 1.0
        new_w = w
        new_h = h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    logging.info(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    # upscale the mask tensor by a factor of 2 using bilinear interpolation
    if isinstance(noise_mask, np.ndarray):
        noise_mask = torch.from_numpy(noise_mask)

    if len(noise_mask.shape) == 2:
        noise_mask = noise_mask.unsqueeze(0)
    else:  # == 3
        noise_mask = noise_mask

    upscaled_mask = None

    for single_mask in noise_mask:
        single_mask = single_mask.unsqueeze(0).unsqueeze(0)
        upscaled_single_mask = torch.nn.functional.interpolate(single_mask, size=(new_h, new_w), mode='bilinear', align_corners=False)
        upscaled_single_mask = upscaled_single_mask.squeeze(0)

        if upscaled_mask is None:
            upscaled_mask = upscaled_single_mask
        else:
            upscaled_mask = torch.cat((upscaled_mask, upscaled_single_mask), dim=0)

    latent_frames = None
    for image in image_frames:
        image = torch.from_numpy(image).unsqueeze(0)

        # upscale
        upscaled_image = utils.tensor_resize(image, new_w, new_h)

        # ksampler
        samples = utils.to_latent_image(upscaled_image, vae)['samples']

        if latent_frames is None:
            latent_frames = samples
        else:
            latent_frames = torch.concat((latent_frames, samples), dim=0)

    cnet_images = None
    if control_net_wrapper is not None:
        positive, negative, cnet_images = control_net_wrapper.apply(positive, negative, torch.from_numpy(image_frames), noise_mask, use_acn=True)

    if len(upscaled_mask) != len(image_frames) and len(upscaled_mask) > 1:
        logging.warning(f"[Impact Pack] DetailerForAnimateDiff: The number of the mask frames({len(upscaled_mask)}) and the image frames({len(image_frames)}) are different. Combine the mask frames and apply.")
        combined_mask = upscaled_mask[0].to(torch.uint8)

        for frame_mask in upscaled_mask[1:]:
            combined_mask |= (frame_mask * 255).to(torch.uint8)

        combined_mask = (combined_mask/255.0).to(torch.float32)

        upscaled_mask = combined_mask.expand(len(image_frames), -1, -1)
        upscaled_mask = utils.to_binary_mask(upscaled_mask, 0.1)

    latent = {
        'noise_mask': upscaled_mask,
        'samples': latent_frames
    }


    sampler_opt=None
    if detailer_hook is not None:
        sampler_opt = detailer_hook.get_custom_sampler()

    if detailer_hook is not None:
        latent = detailer_hook.post_encode(latent)

    refined_latent = impact_sampling.ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                                      latent, denoise, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative, scheduler_func=scheduler_func, sampler_opt=sampler_opt)

    if detailer_hook is not None:
        refined_latent = detailer_hook.pre_decode(refined_latent)

    refined_image_frames = None
    for refined_sample in refined_latent['samples']:
        refined_sample = refined_sample.unsqueeze(0)

        # non-latent downscale - latent downscale cause bad quality
        refined_image = vae.decode(refined_sample)

        if refined_image_frames is None:
            refined_image_frames = refined_image
        else:
            refined_image_frames = torch.concat((refined_image_frames, refined_image), dim=0)

    if detailer_hook is not None:
        refined_image_frames = detailer_hook.post_decode(refined_image_frames)

    refined_image_frames = nodes.ImageScale().upscale(image=refined_image_frames, upscale_method='lanczos', width=w, height=h, crop='disabled')[0]

    return refined_image_frames, cnet_images


def composite_to(dest_latent, crop_region, src_latent):
    x1 = crop_region[0]
    y1 = crop_region[1]

    # composite to original latent
    lc = nodes.LatentComposite()
    orig_image = lc.composite(dest_latent, src_latent, x1, y1)

    return orig_image[0]


def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)

    total_masks = []

    selected = False
    max_score = 0
    max_mask = None
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected and max_mask is not None:
        total_masks.append(max_mask)

    return total_masks


class SAMWrapper:
    def __init__(self, model, is_auto_mode, safe_to_gpu=None):
        self.model = model
        self.safe_to_gpu = safe_to_gpu if safe_to_gpu is not None else SafeToGPU_stub()
        self.is_auto_mode = is_auto_mode

    def prepare_device(self):
        if self.is_auto_mode:
            device = comfy.model_management.get_torch_device()
            self.safe_to_gpu.to_device(self.model, device=device)

    def release_device(self):
        if self.is_auto_mode:
            self.model.to(device="cpu")

    def predict(self, image, points, plabs, bbox, threshold):
        predictor = SamPredictor(self.model)
        predictor.set_image(image, "RGB")

        return sam_predict(predictor, points, plabs, bbox, threshold)


class SAM2Wrapper:
    def __init__(self, config, modelname, is_auto_mode, safe_to_gpu=None, device_mode="AUTO"):
        self.config = config
        self.modelname = modelname
        self.image_predictor = None
        self.video_predictor = None
        self.device_mode = device_mode
        self.safe_to_gpu = safe_to_gpu if safe_to_gpu is not None else SafeToGPU_stub()
        self.is_auto_mode = is_auto_mode

    def prepare_device(self):
        pass

    def prepare_image_device(self):
        if self.is_auto_mode:
            device = comfy.model_management.get_torch_device()
            self.safe_to_gpu.to_device(self.image_predictor.model, device=device)

    def prepare_video_device(self):
        if self.is_auto_mode:
            device = comfy.model_management.get_torch_device()
            self.safe_to_gpu.to_device(self.video_predictor, device=device)

    def release_device(self):
        if self.is_auto_mode:
            if self.image_predictor:
                self.image_predictor.model.to(device="cpu")
            if self.video_predictor:
                self.video_predictor.to(device="cpu")

    def predict(self, image, points, plabs, bbox, threshold):
        if not is_sam2_available:
            raise Exception(sam2_unavailable_message)

        if self.image_predictor is None:
            self.image_predictor = SAM2ImagePredictor(build_sam2(self.config, self.modelname))

        self.prepare_image_device()

        self.image_predictor.set_image(image)

        return sam_predict(self.image_predictor, points, plabs, bbox, threshold)

    def predict_video_segs(self, image_frames, segs):
        if not is_sam2_available:
            raise Exception(sam2_unavailable_message)

        if self.video_predictor is None:
            self.video_predictor = build_sam2_video_predictor(self.config, self.modelname)

        self.prepare_video_device()

        orig_video_height = image_frames.shape[1]
        orig_video_width = image_frames.shape[2]

        image_frames, padding = utils.resize_with_padding(image_frames, self.video_predictor.image_size, self.video_predictor.image_size)
        image_frames = image_frames.permute(0, 3, 1, 2)

        inference_state = {}
        inference_state["images"] = image_frames
        inference_state["num_frames"] = len(image_frames)
        inference_state["video_height"] = self.video_predictor.image_size
        inference_state["video_width"] = self.video_predictor.image_size
        inference_state["offload_video_to_cpu"] = True
        inference_state["offload_state_to_cpu"] = self.device_mode == "CPU"
        inference_state["device"] = self.video_predictor.device

        if inference_state["offload_state_to_cpu"]:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = self.video_predictor.device

        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}

        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []

        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        self.video_predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        temp_masks = {}
        for i in range(0, len(segs[1])):
            bbox = segs[1][i].bbox

            adjusted_bbox = utils.adjust_bbox_after_resize(
                bbox,
                (orig_video_height, orig_video_width),
                (self.video_predictor.image_size, self.video_predictor.image_size),
                padding
            )

            points = [utils.center_of_bbox(adjusted_bbox)]
            plabs = [1]
            self.video_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=i, points=points, labels=plabs, box=adjusted_bbox)
            temp_masks[i] = []

        for frame_idx, object_ids, masks in self.video_predictor.propagate_in_video(inference_state):
            for i in object_ids:
                m = masks[i]
                m = m.permute(1, 2, 0)
                temp_masks[i].append(m)

        result = {}
        for k, v in temp_masks.items():
            m = torch.stack(v, dim=0)
            m = utils.remove_padding(m, padding)
            result[k] = utils.resize_with_padding(m, orig_video_width, orig_video_height)[0]

        return result

class ESAMWrapper:
    def __init__(self, model, device):
        self.model = model
        self.func_inference = nodes.NODE_CLASS_MAPPINGS['Yoloworld_ESAM_Zho']
        self.device = device

    def prepare_device(self):
        pass

    def release_device(self):
        pass

    def predict(self, image, points, plabs, bbox, threshold):
        if self.device == 'CPU':
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        detected_masks = self.func_inference.inference_sam_with_boxes(image=image, xyxy=[bbox], model=self.model, device=self.device)
        return [detected_masks.squeeze(0)]


def make_sam_mask(sam, segs, image, detection_hint, dilation,
                  threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):

    if not hasattr(sam, 'sam_wrapper') and not isinstance(sam, SAM2Wrapper):
        raise Exception("[Impact Pack] Invalid SAMLoader is connected. Make sure 'SAMLoader (Impact)'.\nKnown issue: The ComfyUI-YOLO node overrides the SAMLoader (Impact), making it unusable. You need to uninstall ComfyUI-YOLO.\n\n\n")


    if isinstance(sam, SAM2Wrapper):
        sam_obj = sam
    else:
        sam_obj = sam.sam_wrapper

    sam_obj.prepare_device()

    try:
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        total_masks = []

        use_small_negative = mask_hint_use_negative == "Small"

        # seg_shape = segs[0]
        segs = segs[1]
        if detection_hint == "mask-points":
            points = []
            plabs = []

            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = utils.center_of_bbox(segs[i].bbox)
                points.append(center)

                # small point is background, big point is foreground
                if use_small_negative and bbox[2] - bbox[0] < 10:
                    plabs.append(0)
                else:
                    plabs.append(1)

            detected_masks = sam_obj.predict(image, points, plabs, None, threshold)
            total_masks += detected_masks

        else:
            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = utils.center_of_bbox(bbox)

                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image.shape[0])

                dilated_bbox = [x1, y1, x2, y2]

                points = []
                plabs = []
                if detection_hint == "center-1":
                    points.append(center)
                    plabs = [1]  # 1 = foreground point, 0 = background point

                elif detection_hint == "horizontal-2":
                    gap = (x2 - x1) / 3
                    points.append((x1 + gap, center[1]))
                    points.append((x1 + gap * 2, center[1]))
                    plabs = [1, 1]

                elif detection_hint == "vertical-2":
                    gap = (y2 - y1) / 3
                    points.append((center[0], y1 + gap))
                    points.append((center[0], y1 + gap * 2))
                    plabs = [1, 1]

                elif detection_hint == "rect-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, center[1]))
                    points.append((x1 + x_gap * 2, center[1]))
                    points.append((center[0], y1 + y_gap))
                    points.append((center[0], y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "diamond-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, y1 + y_gap))
                    points.append((x1 + x_gap * 2, y1 + y_gap))
                    points.append((x1 + x_gap, y1 + y_gap * 2))
                    points.append((x1 + x_gap * 2, y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "mask-point-bbox":
                    center = utils.center_of_bbox(segs[i].bbox)
                    points.append(center)
                    plabs = [1]

                elif detection_hint == "mask-area":
                    points, plabs = gen_detection_hints_from_mask_area(segs[i].crop_region[0], segs[i].crop_region[1],
                                                                       segs[i].cropped_mask,
                                                                       mask_hint_threshold, use_small_negative)

                if mask_hint_use_negative == "Outter":
                    npoints, nplabs = gen_negative_hints(image.shape[0], image.shape[1],
                                                         segs[i].crop_region[0], segs[i].crop_region[1],
                                                         segs[i].crop_region[2], segs[i].crop_region[3])

                    points += npoints
                    plabs += nplabs

                detected_masks = sam_obj.predict(image, points, plabs, dilated_bbox, threshold)
                total_masks += detected_masks

        # merge every collected masks
        mask = utils.combine_masks2(total_masks)

    finally:
        sam_obj.release_device()

    if mask is not None:
        mask = mask.float()
        mask = utils.dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)
    else:
        size = image.shape[0], image.shape[1]
        mask = torch.zeros(size, dtype=torch.float32, device="cpu")  # empty mask

    mask = utils.make_3d_mask(mask)
    return mask


def generate_detection_hints(image, seg, center, detection_hint, dilated_bbox, mask_hint_threshold, use_small_negative,
                             mask_hint_use_negative):
    [x1, y1, x2, y2] = dilated_bbox

    points = []
    plabs = []
    if detection_hint == "center-1":
        points.append(center)
        plabs = [1]  # 1 = foreground point, 0 = background point

    elif detection_hint == "horizontal-2":
        gap = (x2 - x1) / 3
        points.append((x1 + gap, center[1]))
        points.append((x1 + gap * 2, center[1]))
        plabs = [1, 1]

    elif detection_hint == "vertical-2":
        gap = (y2 - y1) / 3
        points.append((center[0], y1 + gap))
        points.append((center[0], y1 + gap * 2))
        plabs = [1, 1]

    elif detection_hint == "rect-4":
        x_gap = (x2 - x1) / 3
        y_gap = (y2 - y1) / 3
        points.append((x1 + x_gap, center[1]))
        points.append((x1 + x_gap * 2, center[1]))
        points.append((center[0], y1 + y_gap))
        points.append((center[0], y1 + y_gap * 2))
        plabs = [1, 1, 1, 1]

    elif detection_hint == "diamond-4":
        x_gap = (x2 - x1) / 3
        y_gap = (y2 - y1) / 3
        points.append((x1 + x_gap, y1 + y_gap))
        points.append((x1 + x_gap * 2, y1 + y_gap))
        points.append((x1 + x_gap, y1 + y_gap * 2))
        points.append((x1 + x_gap * 2, y1 + y_gap * 2))
        plabs = [1, 1, 1, 1]

    elif detection_hint == "mask-point-bbox":
        center = utils.center_of_bbox(seg.bbox)
        points.append(center)
        plabs = [1]

    elif detection_hint == "mask-area":
        points, plabs = gen_detection_hints_from_mask_area(seg.crop_region[0], seg.crop_region[1],
                                                           seg.cropped_mask,
                                                           mask_hint_threshold, use_small_negative)

    if mask_hint_use_negative == "Outter":
        npoints, nplabs = gen_negative_hints(image.shape[0], image.shape[1],
                                             seg.crop_region[0], seg.crop_region[1],
                                             seg.crop_region[2], seg.crop_region[3])

        points += npoints
        plabs += nplabs

    return points, plabs


def convert_and_stack_masks(masks):
    if len(masks) == 0:
        return None

    mask_tensors = []
    for mask in masks:
        mask_array = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_array)
        mask_tensors.append(mask_tensor)

    stacked_masks = torch.stack(mask_tensors, dim=0)
    stacked_masks = stacked_masks.unsqueeze(1)

    return stacked_masks


def merge_and_stack_masks(stacked_masks, group_size):
    if stacked_masks is None:
        return None

    num_masks = stacked_masks.size(0)
    merged_masks = []

    for i in range(0, num_masks, group_size):
        subset_masks = stacked_masks[i:i + group_size]
        merged_mask = torch.any(subset_masks, dim=0)
        merged_masks.append(merged_mask)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks, dim=0)

    return merged_masks


def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1*rw), int(y1*rw), int(x2*rh), int(y2*rh)
        bbox = int(bx1*rw), int(by1*rw), int(bx2*rh), int(by2*rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask)

        if isinstance(cropped_mask, torch.Tensor) and len(cropped_mask.shape) == 3:
            cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            cropped_mask = cropped_mask.squeeze(0)
        else:
            cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            cropped_image = utils.tensor_resize(cropped_image if isinstance(cropped_image, torch.Tensor) else torch.from_numpy(cropped_image), new_w, new_h)
            cropped_image = cropped_image.numpy()

        new_seg = SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return (th, tw), new_segs


# Used Python's slicing feature. stacked_masks[2::3] means starting from index 2, selecting every third tensor with a step size of 3.
# This allows for quickly obtaining the last tensor of every three tensors in stacked_masks.
def every_three_pick_last(stacked_masks):
    selected_masks = stacked_masks[2::3]
    return selected_masks


def make_sam_mask_segmented(sam, segs, image, detection_hint, dilation,
                            threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):

    if not hasattr(sam, 'sam_wrapper'):
        raise Exception("[Impact Pack] Invalid SAMLoader is connected. Make sure 'SAMLoader (Impact)'.")

    sam_obj = sam.sam_wrapper
    sam_obj.prepare_device()

    try:
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        total_masks = []

        use_small_negative = mask_hint_use_negative == "Small"

        # seg_shape = segs[0]
        segs = segs[1]
        if detection_hint == "mask-points":
            points = []
            plabs = []

            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = utils.center_of_bbox(bbox)
                points.append(center)

                # small point is background, big point is foreground
                if use_small_negative and bbox[2] - bbox[0] < 10:
                    plabs.append(0)
                else:
                    plabs.append(1)

            detected_masks = sam_obj.predict(image, points, plabs, None, threshold)
            total_masks += detected_masks

        else:
            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = utils.center_of_bbox(bbox)
                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image.shape[0])

                dilated_bbox = [x1, y1, x2, y2]

                points, plabs = generate_detection_hints(image, segs[i], center, detection_hint, dilated_bbox,
                                                         mask_hint_threshold, use_small_negative,
                                                         mask_hint_use_negative)

                detected_masks = sam_obj.predict(image, points, plabs, dilated_bbox, threshold)

                total_masks += detected_masks

        # merge every collected masks
        mask = utils.combine_masks2(total_masks)

    finally:
        sam_obj.release_device()

    mask_working_device = torch.device("cpu")

    if mask is not None:
        mask = mask.float()
        mask = utils.dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)
        mask = mask.to(device=mask_working_device)
    else:
        # Extracting batch, height and width
        height, width, _ = image.shape
        mask = torch.zeros(
            (height, width), dtype=torch.float32, device=mask_working_device
        )  # empty mask

    stacked_masks = convert_and_stack_masks(total_masks)

    return (mask, merge_and_stack_masks(stacked_masks, group_size=3))
    # return every_three_pick_last(stacked_masks)


def segs_bitwise_and_mask(segs, mask):
    mask = utils.make_2d_mask(mask)

    if mask is None:
        logging.warning("[SegsBitwiseAndMask] Cannot operate: MASK is empty.")
        return ([],)

    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
        items.append(item)

    return segs[0], items


def segs_bitwise_subtract_mask(segs, mask):
    mask = utils.make_2d_mask(mask)

    if mask is None:
        logging.warning("[SegsBitwiseSubtractMask] Cannot operate: MASK is empty.")
        return ([],)

    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = cv2.subtract(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
        items.append(item)

    return segs[0], items


def apply_mask_to_each_seg(segs, masks):
    if masks is None:
        logging.warning("[SegsBitwiseAndMask] Cannot operate: MASK is empty.")
        return (segs[0], [],)

    items = []

    masks = masks.squeeze(1)

    for seg, mask in zip(segs[1], masks):
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = (mask.cpu().numpy() * 255).astype(np.uint8)
        cropped_mask2 = cropped_mask2[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
        items.append(item)

    return segs[0], items


def dilate_segs(segs, factor):
    if factor == 0:
        return segs

    new_segs = []
    for seg in segs[1]:
        new_mask = utils.dilate_mask(seg.cropped_mask, factor)
        new_seg = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return (segs[0], new_segs)


class ONNXDetector:
    onnx_model = None

    def __init__(self, onnx_model):
        self.onnx_model = onnx_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        try:
            import impact.impact_onnx as onnx

            h = image.shape[1]
            w = image.shape[2]

            labels, scores, boxes = onnx.onnx_inference(image, self.onnx_model)

            # collect feasible item
            result = []

            for i in range(len(labels)):
                if scores[i] > threshold:
                    item_bbox = boxes[i]
                    x1, y1, x2, y2 = item_bbox

                    if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                        crop_region = utils.make_crop_region(w, h, item_bbox, crop_factor)

                        if detailer_hook is not None:
                            crop_region = item_bbox.post_crop_region(w, h, item_bbox, crop_region)

                        crop_x1, crop_y1, crop_x2, crop_y2, = crop_region

                        # prepare cropped mask
                        cropped_mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1))
                        cropped_mask[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1] = 1
                        cropped_mask = utils.dilate_mask(cropped_mask, dilation)

                        # make items. just convert the integer label to a string
                        item = SEG(None, cropped_mask, scores[i], crop_region, item_bbox, str(labels[i]), None)
                        result.append(item)

            shape = h, w
            segs = shape, result

            if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
                segs = detailer_hook.post_detection(segs)

            return segs
        except Exception as e:
            logging.error(f"ONNXDetector: unable to execute.\n{e}")

    def detect_combined(self, image, threshold, dilation):
        return segs_to_combined_mask(self.detect(image, threshold, dilation, 1))

    def setAux(self, x):
        pass


def batch_mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1, label='A', crop_min_size=None, detailer_hook=None):
    combined_mask = mask.max(dim=0).values

    segs = mask_to_segs(combined_mask, combined, crop_factor, bbox_fill, drop_size, label, crop_min_size, detailer_hook)

    new_segs = []
    for seg in segs[1]:
        x1, y1, x2, y2 = seg.crop_region
        cropped_mask = mask[:, y1:y2, x1:x2]
        item = SEG(None, cropped_mask, 1.0, seg.crop_region, seg.bbox, label, None)
        new_segs.append(item)

    return segs[0], new_segs


def mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1, label='A', crop_min_size=None, detailer_hook=None, is_contour=True):
    drop_size = max(drop_size, 1)
    if mask is None:
        logging.info("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)

    if isinstance(mask, np.ndarray):
        pass  # `mask` is already a NumPy array
    else:
        try:
            mask = mask.numpy()
        except AttributeError:
            logging.info("[mask_to_segs] Cannot operate: MASK is not a NumPy array or Tensor.")
            return ([],)

    if mask is None:
        logging.info("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)

    result = []

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)

    for i in range(mask.shape[0]):
        mask_i = mask[i]

        if combined:
            indices = np.nonzero(mask_i)
            if len(indices[0]) > 0 and len(indices[1]) > 0:
                bbox = (
                    np.min(indices[1]),
                    np.min(indices[0]),
                    np.max(indices[1]),
                    np.max(indices[0]),
                )
                crop_region = utils.make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor
                )
                x1, y1, x2, y2 = crop_region

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask_i[y1:y2, x1:x2]

                    if bbox_fill:
                        bx1, by1, bx2, by2 = bbox
                        cropped_mask = cropped_mask.copy()
                        cropped_mask[by1:by2, bx1:bx2] = 1.0

                    if cropped_mask is not None:
                        item = SEG(None, cropped_mask, 1.0, crop_region, bbox, label, None)
                        result.append(item)

        else:
            mask_i_uint8 = (mask_i * 255.0).astype(np.uint8)
            contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours):
                hierarchy = ctree[0][j]
                if hierarchy[3] != -1:
                    continue

                separated_mask = np.zeros_like(mask_i_uint8)
                cv2.drawContours(separated_mask, [contour], 0, 255, -1)
                separated_mask = np.array(separated_mask / 255.0).astype(np.float32)

                x, y, w, h = cv2.boundingRect(contour)
                bbox = x, y, x + w, y + h
                crop_region = utils.make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor, crop_min_size
                )

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)

                if w > drop_size and h > drop_size:
                    if is_contour:
                        mask_src = separated_mask
                    else:
                        mask_src = mask_i * separated_mask

                    cropped_mask = np.array(
                        mask_src[
                            crop_region[1]: crop_region[3],
                            crop_region[0]: crop_region[2],
                        ]
                    )

                    if bbox_fill:
                        cx1, cy1, _, _ = crop_region
                        bx1 = x - cx1
                        bx2 = x+w - cx1
                        by1 = y - cy1
                        by2 = y+h - cy1
                        cropped_mask[by1:by2, bx1:bx2] = 1.0

                    if cropped_mask is not None:
                        cropped_mask = torch.clip(torch.from_numpy(cropped_mask), 0, 1.0)
                        item = SEG(None, cropped_mask.numpy(), 1.0, crop_region, bbox, label, None)
                        result.append(item)

    if not result:
        logging.info("[mask_to_segs] Empty mask.")

    logging.info(f"# of Detected SEGS: {len(result)}")
    # for r in result:
    #     print(f"\tbbox={r.bbox}, crop={r.crop_region}, label={r.label}")

    # shape: (b,h,w) -> (h,w)
    return (mask.shape[1], mask.shape[2]), result


def mediapipe_facemesh_to_segs(image, crop_factor, bbox_fill, crop_min_size, drop_size, dilation, face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil):
    parts = {
        "face": np.array([0x0A, 0xC8, 0x0A]),
        "mouth": np.array([0x0A, 0xB4, 0x0A]),
        "left_eyebrow": np.array([0xB4, 0xDC, 0x0A]),
        "left_eye": np.array([0xB4, 0xC8, 0x0A]),
        "left_pupil": np.array([0xFA, 0xC8, 0x0A]),
        "right_eyebrow": np.array([0x0A, 0xDC, 0xB4]),
        "right_eye": np.array([0x0A, 0xC8, 0xB4]),
        "right_pupil": np.array([0x0A, 0xC8, 0xFA]),
    }

    def create_segments(image, color):
        image = (image * 255).to(torch.uint8)
        image = image.squeeze(0).numpy()
        mask = cv2.inRange(image, color, color)

        contours, ctree = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_list = []
        for i, contour in enumerate(contours):
            hierarchy = ctree[0][i]
            if hierarchy[3] == -1:
                convex_hull = cv2.convexHull(contour)
                convex_segment = np.zeros_like(image)
                cv2.fillPoly(convex_segment, [convex_hull], (255, 255, 255))

                convex_segment = np.expand_dims(convex_segment, axis=0).astype(np.float32) / 255.0
                tensor = torch.from_numpy(convex_segment)
                mask_tensor = torch.any(tensor != 0, dim=-1).float()
                mask_tensor = mask_tensor.squeeze(0)
                mask_tensor = torch.from_numpy(utils.dilate_mask(mask_tensor.numpy(), dilation))
                mask_list.append(mask_tensor.unsqueeze(0))

        return mask_list

    segs = []

    def create_seg(label):
        mask_list = create_segments(image, parts[label])
        for mask in mask_list:
            seg = mask_to_segs(mask, False, crop_factor, bbox_fill, drop_size=drop_size, label=label, crop_min_size=crop_min_size)
            if len(seg[1]) > 0:
                segs.extend(seg[1])

    if face:
        create_seg('face')

    if mouth:
        create_seg('mouth')

    if left_eyebrow:
        create_seg('left_eyebrow')

    if left_eye:
        create_seg('left_eye')

    if left_pupil:
        create_seg('left_pupil')

    if right_eyebrow:
        create_seg('right_eyebrow')

    if right_eye:
        create_seg('right_eye')

    if right_pupil:
        create_seg('right_pupil')

    return (image.shape[1], image.shape[2]), segs


def segs_to_combined_mask(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    mask = np.zeros((h, w), dtype=np.uint8)

    for seg in segs[1]:
        cropped_mask = seg.cropped_mask
        crop_region = seg.crop_region
        mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

    return torch.from_numpy(mask.astype(np.float32) / 255.0)


def segs_to_masklist(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    masks = []
    for seg in segs[1]:
        if isinstance(seg.cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(seg.cropped_mask)
        else:
            cropped_mask = seg.cropped_mask

        if cropped_mask.ndim == 2:
            cropped_mask = cropped_mask.unsqueeze(0)

        n = len(cropped_mask)

        mask = torch.zeros((n, h, w), dtype=torch.uint8)
        crop_region = seg.crop_region
        mask[:, crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).to(torch.uint8)
        mask = (mask / 255.0).to(torch.float32)

        for x in mask:
            masks.append(x)

    if len(masks) == 0:
        empty_mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
        masks = [empty_mask]

    return masks


def vae_decode(vae, samples, use_tile, hook, tile_size=512, overlap=64):
    if use_tile:
        decoder = nodes.VAEDecodeTiled()
        if 'overlap' in inspect.signature(decoder.decode).parameters:
            pixels = decoder.decode(vae, samples, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            pixels = decoder.decode(vae, samples, tile_size)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]

    if hook is not None:
        pixels = hook.post_decode(pixels)

    return pixels


def vae_encode(vae, pixels, use_tile, hook, tile_size=512, overlap=64):
    if use_tile:
        encoder = nodes.VAEEncodeTiled()
        if 'overlap' in inspect.signature(encoder.encode).parameters:
            samples = encoder.encode(vae, pixels, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            samples = encoder.encode(vae, pixels, tile_size)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]

    if hook is not None:
        samples = hook.post_encode(samples)

    return samples


def latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2] * scale_factor
    h = pixels.shape[1] * scale_factor
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        pixels = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, pixels)[0]
        current_w = pixels.shape[2]
        if current_w == w:
            logging.info("[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space_with_model(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                             tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]

def latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                              tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]
    h = pixels.shape[1]

    new_w = w * scale_factor
    new_h = h * scale_factor

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        pixels = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, pixels)[0]
        current_w = pixels.shape[2]
        if current_w == w:
            logging.info("[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


class TwoSamplersForMaskUpscaler:
    def __init__(self, scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae,
                 full_sampler_opt=None, upscale_model_opt=None, hook_base_opt=None, hook_mask_opt=None,
                 hook_full_opt=None,
                 tile_size=512):

        mask = utils.make_2d_mask(mask)

        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

        self.params = scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae
        self.upscale_model = upscale_model_opt
        self.full_sampler = full_sampler_opt
        self.hook_base = hook_base_opt
        self.hook_mask = hook_mask_opt
        self.hook_full = hook_full_opt
        self.use_tiled_vae = use_tiled_vae
        self.tile_size = tile_size
        self.is_tiled = False
        self.vae = vae

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae = self.params

        mask = utils.make_2d_mask(mask)

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                            use_tile=self.use_tiled_vae,
                                                            save_temp_prefix=save_temp_prefix,
                                                            hook=self.hook_base, tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model,
                                                                       upscale_factor, vae,
                                                                       use_tile=self.use_tiled_vae,
                                                                       save_temp_prefix=save_temp_prefix,
                                                                       hook=self.hook_mask, tile_size=self.tile_size)

        return self.do_samples(step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent)

    def prepare_hook(self, step_info):
        if self.hook_base is not None:
            self.hook_base.set_steps(step_info)
        if self.hook_mask is not None:
            self.hook_mask.set_steps(step_info)
        if self.hook_full is not None:
            self.hook_full.set_steps(step_info)

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae = self.params

        mask = utils.make_2d_mask(mask)

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae,
                                                                  use_tile=self.use_tiled_vae,
                                                                  save_temp_prefix=save_temp_prefix,
                                                                  hook=self.hook_base,
                                                                  tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, self.upscale_model,
                                                                             w, h, vae,
                                                                             use_tile=self.use_tiled_vae,
                                                                             save_temp_prefix=save_temp_prefix,
                                                                             hook=self.hook_mask,
                                                                             tile_size=self.tile_size)

        return self.do_samples(step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent)

    def is_full_sample_time(self, step_info, sample_schedule):
        cur_step, total_step = step_info

        # make start from 1 instead of zero
        cur_step += 1
        total_step += 1

        if sample_schedule == "none":
            return False

        elif sample_schedule == "interleave1":
            return cur_step % 2 == 0

        elif sample_schedule == "interleave2":
            return cur_step % 3 == 0

        elif sample_schedule == "interleave3":
            return cur_step % 4 == 0

        elif sample_schedule == "last1":
            return cur_step == total_step

        elif sample_schedule == "last2":
            return cur_step >= total_step - 1

        elif sample_schedule == "interleave1+last1":
            return cur_step % 2 == 0 or cur_step >= total_step - 1

        elif sample_schedule == "interleave2+last1":
            return cur_step % 2 == 0 or cur_step >= total_step - 1

        elif sample_schedule == "interleave3+last1":
            return cur_step % 2 == 0 or cur_step >= total_step - 1

    def do_samples(self, step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent):
        mask = utils.make_2d_mask(mask)

        if self.is_full_sample_time(step_info, sample_schedule):
            logging.info(f"step_info={step_info} / full time")

            upscaled_latent = base_sampler.sample(upscaled_latent, self.hook_base)
            sampler = self.full_sampler if self.full_sampler is not None else base_sampler
            return sampler.sample(upscaled_latent, self.hook_full)

        else:
            logging.info(f"step_info={step_info} / non-full time")
            # upscale mask
            if mask.ndim == 2:
                mask = mask[None, :, :, None]
            upscaled_mask = F.interpolate(mask, size=(upscaled_latent['samples'].shape[2], upscaled_latent['samples'].shape[3]), mode='bilinear', align_corners=True)
            upscaled_mask = upscaled_mask[:, :, :upscaled_latent['samples'].shape[2], :upscaled_latent['samples'].shape[3]]

            # base sampler
            upscaled_inv_mask = torch.where(upscaled_mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))
            upscaled_latent['noise_mask'] = upscaled_inv_mask
            upscaled_latent = base_sampler.sample(upscaled_latent, self.hook_base)

            # mask sampler
            upscaled_latent['noise_mask'] = upscaled_mask
            upscaled_latent = mask_sampler.sample(upscaled_latent, self.hook_mask)

            # remove mask
            del upscaled_latent['noise_mask']
            return upscaled_latent


class PixelKSampleUpscaler:
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                 use_tiled_vae, upscale_model_opt=None, hook_opt=None, tile_size=512, scheduler_func=None,
                 tile_cnet_opt=None, tile_cnet_strength=1.0):
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.use_tiled_vae = use_tiled_vae
        self.tile_size = tile_size
        self.is_tiled = False
        self.vae = vae
        self.scheduler_func = scheduler_func
        self.tile_cnet = tile_cnet_opt
        self.tile_cnet_strength = tile_cnet_strength

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise, images):
        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]
                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=preprocessed,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0,
                                                                                      end_percent=1.0,
                                                                                      vae=self.vae)

        refined_latent = impact_sampling.impact_sample(model, seed, steps, cfg, sampler_name, scheduler,
                                                       positive, negative, upscaled_latent, denoise, scheduler_func=self.scheduler_func)

        return refined_latent

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space2(samples, scale_method, upscale_factor, vae,
                                               use_tile=self.use_tiled_vae,
                                               save_temp_prefix=save_temp_prefix, hook=self.hook, tile_size=512)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model2(samples, scale_method, self.upscale_model,
                                                          upscale_factor, vae,
                                                          use_tile=self.use_tiled_vae,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook,
                                                          tile_size=self.tile_size)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)

        if 'noise_mask' in samples:
            upscaled_latent['noise_mask'] = samples['noise_mask']

        refined_latent = self.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise, upscaled_images)
        return refined_latent

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae,
                                                     use_tile=self.use_tiled_vae,
                                                     save_temp_prefix=save_temp_prefix, hook=self.hook,
                                                     tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, self.upscale_model,
                                                                w, h, vae,
                                                                use_tile=self.use_tiled_vae,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)

        if 'noise_mask' in samples:
            upscaled_latent['noise_mask'] = samples['noise_mask']

        refined_latent = self.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise, upscaled_images)
        return refined_latent


class IPAdapterWrapper:
    def __init__(self, ipadapter_pipe, weight, noise, weight_type, start_at, end_at, unfold_batch, weight_v2, reference_image, neg_image=None, prev_control_net=None, combine_embeds='concat'):
        self.reference_image = reference_image
        self.ipadapter_pipe = ipadapter_pipe
        self.weight = weight
        self.weight_type = weight_type
        self.noise = noise
        self.start_at = start_at
        self.end_at = end_at
        self.unfold_batch = unfold_batch
        self.prev_control_net = prev_control_net
        self.weight_v2 = weight_v2
        self.image = reference_image
        self.neg_image = neg_image
        self.combine_embeds = combine_embeds

    # name 'apply_ipadapter' isn't allowed
    def doit_ipadapter(self, model):
        cnet_image_list = [self.image]
        prev_cnet_images = []

        if 'IPAdapterAdvanced' not in nodes.NODE_CLASS_MAPPINGS:
            if 'IPAdapterApply' in nodes.NODE_CLASS_MAPPINGS:
                raise Exception("[ERROR] 'ComfyUI IPAdapter Plus' is outdated.")

            utils.try_install_custom_node('https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                                          "To use 'IPAdapterApplySEGS' node, 'ComfyUI IPAdapter Plus' extension is required.")
            raise Exception("[ERROR] To use IPAdapterApplySEGS, you need to install 'ComfyUI IPAdapter Plus'")

        obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterAdvanced']

        ipadapter, _, clip_vision, insightface, lora_loader = self.ipadapter_pipe
        model = lora_loader(model)

        if self.prev_control_net is not None:
            model, prev_cnet_images = self.prev_control_net.doit_ipadapter(model)

        model = obj().apply_ipadapter(model=model, ipadapter=ipadapter, weight=self.weight, weight_type=self.weight_type,
                                      start_at=self.start_at, end_at=self.end_at, combine_embeds=self.combine_embeds,
                                      clip_vision=clip_vision, image=self.image, image_negative=self.neg_image, attn_mask=None,
                                      insightface=insightface, weight_faceidv2=self.weight_v2)[0]

        cnet_image_list.extend(prev_cnet_images)

        return model, cnet_image_list

    def apply(self, positive, negative, image, mask=None, use_acn=False):
        if self.prev_control_net is not None:
            return self.prev_control_net.apply(positive, negative, image, mask, use_acn=use_acn)
        else:
            return positive, negative, []


class ControlNetWrapper:
    def __init__(self, control_net, strength, preprocessor, prev_control_net=None, original_size=None, crop_region=None, control_image=None):
        self.control_net = control_net
        self.strength = strength
        self.preprocessor = preprocessor
        self.prev_control_net = prev_control_net

        if original_size is not None and crop_region is not None and control_image is not None:
            self.control_image = utils.tensor_resize(control_image, original_size[1], original_size[0])
            self.control_image = torch.tensor(utils.tensor_crop(self.control_image, crop_region))
        else:
            self.control_image = None

    def apply(self, positive, negative, image, mask=None, use_acn=False):
        cnet_image_list = []
        prev_cnet_images = []

        if self.prev_control_net is not None:
            positive, negative, prev_cnet_images = self.prev_control_net.apply(positive, negative, image, mask, use_acn=use_acn)

        if self.control_image is not None:
            cnet_image = self.control_image
        elif self.preprocessor is not None:
            cnet_image = self.preprocessor.apply(image, mask)
        else:
            cnet_image = image

        cnet_image_list.extend(prev_cnet_images)
        cnet_image_list.append(cnet_image)

        if use_acn:
            if "ACN_AdvancedControlNetApply" in nodes.NODE_CLASS_MAPPINGS:
                acn = nodes.NODE_CLASS_MAPPINGS['ACN_AdvancedControlNetApply']()
                positive, negative, _ = acn.apply_controlnet(positive=positive, negative=negative, control_net=self.control_net, image=cnet_image,
                                                             strength=self.strength, start_percent=0.0, end_percent=1.0)
            else:
                utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                              "To use 'ControlNetWrapper' for AnimateDiff, 'ComfyUI-Advanced-ControlNet' extension is required.")
                raise Exception("'ACN_AdvancedControlNetApply' node isn't installed.")
        else:
            positive = nodes.ControlNetApply().apply_controlnet(positive, self.control_net, cnet_image, self.strength)[0]

        return positive, negative, cnet_image_list

    def doit_ipadapter(self, model):
        if self.prev_control_net is not None:
            return self.prev_control_net.doit_ipadapter(model)
        else:
            return model, []


class ControlNetAdvancedWrapper:
    def __init__(self, control_net, strength, start_percent, end_percent, preprocessor, prev_control_net=None,
                 original_size=None, crop_region=None, control_image=None, vae=None):
        self.control_net = control_net
        self.strength = strength
        self.preprocessor = preprocessor
        self.prev_control_net = prev_control_net
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.vae = vae

        if original_size is not None and crop_region is not None and control_image is not None:
            self.control_image = utils.tensor_resize(control_image, original_size[1], original_size[0])
            self.control_image = torch.tensor(utils.tensor_crop(self.control_image, crop_region))
        else:
            self.control_image = None

    def doit_ipadapter(self, model):
        if self.prev_control_net is not None:
            return self.prev_control_net.doit_ipadapter(model)
        else:
            return model, []

    def apply(self, positive, negative, image, mask=None, use_acn=False):
        cnet_image_list = []
        prev_cnet_images = []

        if self.prev_control_net is not None:
            positive, negative, prev_cnet_images = self.prev_control_net.apply(positive, negative, image, mask)

        if self.control_image is not None:
            cnet_image = self.control_image
        elif self.preprocessor is not None:
            cnet_image = self.preprocessor.apply(image, mask)
        else:
            cnet_image = image

        cnet_image_list.extend(prev_cnet_images)
        cnet_image_list.append(cnet_image)

        if use_acn:
            if "ACN_AdvancedControlNetApply" in nodes.NODE_CLASS_MAPPINGS:
                acn = nodes.NODE_CLASS_MAPPINGS['ACN_AdvancedControlNetApply']()
                positive, negative, _ = acn.apply_controlnet(positive=positive, negative=negative, control_net=self.control_net, image=cnet_image,
                                                             strength=self.strength, start_percent=self.start_percent, end_percent=self.end_percent)
            else:
                utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                              "To use 'ControlNetAdvancedWrapper' for AnimateDiff, 'ComfyUI-Advanced-ControlNet' extension is required.")
                raise Exception("'ACN_AdvancedControlNetApply' node isn't installed.")
        else:
            if self.vae is not None:
                apply_controlnet = nodes.ControlNetApplyAdvanced().apply_controlnet
                signature = inspect.signature(apply_controlnet)

                if 'vae' in signature.parameters:
                    positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, self.control_net, cnet_image, self.strength, self.start_percent, self.end_percent, vae=self.vae)
                else:
                    logging.error("[Impact Pack] ERROR: The ComfyUI version is outdated. VAE cannot be used in ApplyControlNet.")
                    raise Exception("[Impact Pack] ERROR: The ComfyUI version is outdated. VAE cannot be used in ApplyControlNet.")
            else:
                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, self.control_net, cnet_image, self.strength, self.start_percent, self.end_percent)

        return positive, negative, cnet_image_list


# REQUIREMENTS: BlenderNeko/ComfyUI_TiledKSampler
class TiledKSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                 tile_width, tile_height, tiling_strategy):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy

    def sample(self, latent_image, hook=None):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            TiledKSampler = nodes.NODE_CLASS_MAPPINGS['BNK_TiledKSampler']
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                          "To use 'TiledKSamplerProvider', 'Tiled sampling for ComfyUI' extension is required.")
            raise Exception("'BNK_TiledKSampler' node isn't installed.")

        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                 denoise)

        return TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                                      scheduler, positive, negative, latent_image, denoise)[0]


class PixelTiledKSampleUpscaler:
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                 denoise,
                 tile_width, tile_height, tiling_strategy,
                 upscale_model_opt=None, hook_opt=None, tile_cnet_opt=None, tile_size=512, tile_cnet_strength=1.0, overlap=64):
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.vae = vae
        self.tile_params = tile_width, tile_height, tiling_strategy
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_cnet = tile_cnet_opt
        self.tile_size = tile_size
        self.is_tiled = True
        self.tile_cnet_strength = tile_cnet_strength
        self.overlap = overlap

    def tiled_ksample(self, latent, images):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            TiledKSampler = nodes.NODE_CLASS_MAPPINGS['BNK_TiledKSampler']
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                          "To use 'PixelTiledKSampleUpscalerProvider', 'Tiled sampling for ComfyUI' extension is required.")
            raise RuntimeError("'BNK_TiledKSampler' node isn't installed.")

        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params
        tile_width, tile_height, tiling_strategy = self.tile_params

        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]

                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=preprocessed,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0, end_percent=1.0,
                                                                                      vae=self.vae)

        return TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                                      scheduler, positive, negative, latent, denoise)[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space2(samples, scale_method, upscale_factor, vae,
                                               use_tile=True, save_temp_prefix=save_temp_prefix,
                                               hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model2(samples, scale_method, self.upscale_model,
                                                          upscale_factor, vae, use_tile=True,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook, tile_size=self.tile_size)

        refined_latent = self.tiled_ksample(upscaled_latent, upscaled_images)

        return refined_latent

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae,
                                                     use_tile=True, save_temp_prefix=save_temp_prefix,
                                                     hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method,
                                                                self.upscale_model, w, h, vae,
                                                                use_tile=True,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        refined_latent = self.tiled_ksample(upscaled_latent, upscaled_images)

        return refined_latent


# REQUIREMENTS: biegert/ComfyUI-CLIPSeg
class BBoxDetectorBasedOnCLIPSeg:
    prompt = None
    blur = None
    threshold = None
    dilation_factor = None
    aux = None

    def __init__(self, prompt, blur, threshold, dilation_factor):
        self.prompt = prompt
        self.blur = blur
        self.threshold = threshold
        self.dilation_factor = dilation_factor

    def detect(self, image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size=1, detailer_hook=None):
        mask = self.detect_combined(image, bbox_threshold, bbox_dilation)

        mask = utils.make_2d_mask(mask)

        segs = mask_to_segs(mask, False, bbox_crop_factor, True, drop_size, detailer_hook=detailer_hook)

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, bbox_threshold, bbox_dilation):
        if "CLIPSeg" in nodes.NODE_CLASS_MAPPINGS:
            CLIPSeg = nodes.NODE_CLASS_MAPPINGS['CLIPSeg']
        else:
            utils.try_install_custom_node('https://github.com/biegert/ComfyUI-CLIPSeg/raw/main/custom_nodes/clipseg.py',
                                          "To use 'CLIPSegDetectorProvider', 'CLIPSeg' extension is required.")
            raise Exception("'CLIPSeg' node isn't installed.")

        if self.threshold is None:
            threshold = bbox_threshold
        else:
            threshold = self.threshold

        if self.dilation_factor is None:
            dilation_factor = bbox_dilation
        else:
            dilation_factor = self.dilation_factor

        prompt = self.aux if self.prompt == '' and self.aux is not None else self.prompt

        mask, _, _ = CLIPSeg().segment_image(image, prompt, self.blur, threshold, dilation_factor)
        mask = utils.to_binary_mask(mask)
        return mask

    def setAux(self, x):
        self.aux = x


def update_node_status(node, text, progress=None):
    if PromptServer.instance.client_id is None:
        return

    PromptServer.instance.send_sync("impact/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, PromptServer.instance.client_id)


def random_mask_raw(mask, bbox, factor):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    factor = max(6, int(min(w, h) * factor / 4))

    def draw_random_circle(center, radius):
        i, j = center
        for x in range(int(i - radius), int(i + radius)):
            for y in range(int(j - radius), int(j + radius)):
                if np.linalg.norm(np.array([x, y]) - np.array([i, j])) <= radius:
                    mask[x, y] = 1

    def draw_irregular_line(start, end, pivot, is_vertical):
        i = start
        while i < end:
            base_radius = np.random.randint(5, factor)
            radius = int(base_radius)

            if is_vertical:
                draw_random_circle((i, pivot), radius)
            else:
                draw_random_circle((pivot, i), radius)

            i += radius

    def draw_irregular_line_parallel(start, end, pivot, is_vertical):
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            step = (end - start) // 16
            for i in range(start, end, step):
                future = executor.submit(draw_irregular_line, i, min(i + step, end), pivot, is_vertical)
                futures.append(future)

            for future in futures:
                future.result()

    draw_irregular_line_parallel(y1 + factor, y2 - factor, x1 + factor, True)
    draw_irregular_line_parallel(y1 + factor, y2 - factor, x2 - factor, True)
    draw_irregular_line_parallel(x1 + factor, x2 - factor, y1 + factor, False)
    draw_irregular_line_parallel(x1 + factor, x2 - factor, y2 - factor, False)

    mask[y1 + factor:y2 - factor, x1 + factor:x2 - factor] = 1.0


def random_mask(mask, bbox, factor, size=128):
    small_mask = np.zeros((size, size)).astype(np.float32)
    random_mask_raw(small_mask, (0, 0, size, size), factor)

    x1, y1, x2, y2 = bbox
    small_mask = torch.tensor(small_mask).unsqueeze(0).unsqueeze(0)
    bbox_mask = torch.nn.functional.interpolate(small_mask, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
    bbox_mask = bbox_mask.squeeze(0).squeeze(0)
    mask[y1:y2, x1:x2] = bbox_mask


def adaptive_mask_paste(dest_mask, src_mask, bbox):
    x1, y1, x2, y2 = bbox
    small_mask = torch.tensor(src_mask).unsqueeze(0).unsqueeze(0)
    bbox_mask = torch.nn.functional.interpolate(small_mask, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
    bbox_mask = bbox_mask.squeeze(0).squeeze(0)
    dest_mask[y1:y2, x1:x2] = bbox_mask


def crop_condition_mask(mask, image, crop_region):
    cond_scale = (mask.shape[1] / image.shape[1], mask.shape[2] / image.shape[2])
    mask_region = [round(v * cond_scale[i % 2]) for i, v in enumerate(crop_region)]
    return utils.crop_ndarray3(mask, mask_region)


class SafeToGPU:
    def __init__(self, size):
        self.size = size

    def to_device(self, obj, device):
        if utils.is_same_device(device, 'cpu'):
            obj.to(device)
        else:
            if utils.is_same_device(obj.device, 'cpu'):  # cpu to gpu
                model_management.free_memory(self.size * 1.3, device)
                if model_management.get_free_memory(device) > self.size * 1.3:
                    try:
                        obj.to(device)
                    except Exception:
                        logging.warning(f"[Impact Pack] The model is not moved to the '{device}' due to insufficient memory. [1]")
                else:
                    logging.warning(f"[Impact Pack] The model is not moved to the '{device}' due to insufficient memory. [2]")


class SafeToGPU_stub():
    def to_device(self, obj, device):
        pass


from comfy.cli_args import args, LatentPreviewMethod
import folder_paths
from latent_preview import TAESD, TAESDPreviewerImpl, Latent2RGBPreviewer

try:
    import comfy.latent_formats as latent_formats


    def get_previewer(device, latent_format=latent_formats.SD15(), force=False, method=None):
        previewer = None

        if method is None:
            method = args.preview_method

        if method != LatentPreviewMethod.NoPreviews or force:
            # TODO previewer methods
            taesd_decoder_path = None

            if hasattr(latent_format, "taesd_decoder_path"):
                taesd_decoder_path = folder_paths.get_full_path("vae_approx", latent_format.taesd_decoder_name)

            if method == LatentPreviewMethod.Auto:
                method = LatentPreviewMethod.Latent2RGB
                if taesd_decoder_path:
                    method = LatentPreviewMethod.TAESD

            if method == LatentPreviewMethod.TAESD:
                if taesd_decoder_path:
                    taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels).to(device)
                    previewer = TAESDPreviewerImpl(taesd)
                else:
                    logging.warning("[Impact Pack] TAESD previews enabled, but could not find models/vae_approx/{}".format(
                        latent_format.taesd_decoder_name))

            if previewer is None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
        return previewer

except Exception:
    logging.error("#########################################################################")
    logging.error("[ERROR] ComfyUI-Impact-Pack: Please update ComfyUI to the latest version.")
    logging.error("#########################################################################")
