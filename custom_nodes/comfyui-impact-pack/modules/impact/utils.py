import torch
import torchvision
import cv2
import numpy as np
import folder_paths
import nodes
from . import config
from PIL import Image
import comfy
import time
import logging


class TensorBatchBuilder:
    def __init__(self):
        self.tensor = None

    def concat(self, new_tensor):
        if self.tensor is None:
            self.tensor = new_tensor
        else:
            self.tensor = torch.concat((self.tensor, new_tensor), dim=0)


def tensor_convert_rgba(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image

    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
        return torch.cat((image, alpha), axis=-1)

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Similar error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 4)")


def tensor_convert_rgb(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 3:
        return image

    if n_channel == 4:
        image = image[..., :3]
        if prefer_copy:
            image = image.copy()
        return image

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Same error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 3)")


def resize_with_padding(image, target_w: int, target_h: int):
    _tensor_check_image(image)
    b, h, w, c = image.shape
    image = image.permute(0, 3, 1, 2)  # B, C, H, W

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    image = image.permute(0, 2, 3, 1)  # B, H, W, C
    return image, (pad_top, pad_bottom, pad_left, pad_right)


def remove_padding(image, padding):
    pad_top, pad_bottom, pad_left, pad_right = padding
    return image[:, pad_top:image.shape[1] - pad_bottom, pad_left:image.shape[2] - pad_right, :]


def adjust_bbox_after_resize(bbox, original_size, target_size, padding):
    """
    bbox: (x1, y1, x2, y2) in original image
    original_size: (original_h, original_w)
    target_size: (target_h, target_w)
    padding: (pad_top, pad_bottom, pad_left, pad_right)
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    pad_top, pad_bottom, pad_left, pad_right = padding

    scale = min(target_w / orig_w, target_h / orig_h)

    # Apply scale
    x1 = int(bbox[0] * scale + pad_left)
    y1 = int(bbox[1] * scale + pad_top)
    x2 = int(bbox[2] * scale + pad_left)
    y2 = int(bbox[3] * scale + pad_top)

    return x1, y1, x2, y2


def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


# TODO: Sadly, we need LANCZOS
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)

            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)

        return scaled_images.tensor
    else:
        return general_tensor_resize(image, w, h)


def tensor_get_size(image):
    """Mimicking `PIL.Image.size`"""
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(0), 0, 255).astype(np.uint8))


def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        return tensor2pil(image)
    if isinstance(image, np.ndarray):
        return numpy2pil(image)
    raise ValueError(f"Cannot convert {type(image)} to PIL.Image")


def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image)) / 255.0
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")


def to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, torch.Tensor):
        return image.numpy()
    if isinstance(image, np.ndarray):
        return image
    raise ValueError(f"Cannot convert {type(image)} to numpy.ndarray")

def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return


def _tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")
    return


def tensor_crop(image, crop_region):
    _tensor_check_image(image)
    return crop_ndarray4(image, crop_region)


def tensor2numpy(image):
    _tensor_check_image(image)
    return image.numpy()


def tensor_paste(image1, image2, left_top, mask):
    """
    Pastes image2 onto image1 at position left_top using mask.
    Supports both RGB and RGBA images.
    """
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    if image2.shape[1:3] != mask.shape[1:3]:
        mask = resize_mask(mask.squeeze(dim=3), image2.shape[1:3]).unsqueeze(dim=3)

    x, y = left_top
    _, h1, w1, c1 = image1.shape
    _, h2, w2, c2 = image2.shape

    # Calculate image patch size
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    # If the patch is out of bound, nothing to do!
    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]

    # Get the region to be modified
    region1 = image1[:, y:y+h, x:x+w, :]
    region2 = image2[:, :h, :w, :]

    # Handle RGB and RGBA cases
    if c1 == 3 and c2 == 3:
        # Both RGB - simple case
        image1[:, y:y+h, x:x+w, :] = (1 - mask) * region1 + mask * region2

    elif c1 == 4 and c2 == 4:
        # Both RGBA - need to handle alpha channel separately
        # RGB channels
        image1[:, y:y+h, x:x+w, :3] = (
            (1 - mask) * region1[:, :, :, :3] +
            mask * region2[:, :, :, :3]
        )

        # Alpha channel - use "over" composition
        a1 = region1[:, :, :, 3:4]
        a2 = region2[:, :, :, 3:4] * mask
        new_alpha = a1 + a2 * (1 - a1)
        image1[:, y:y+h, x:x+w, 3:4] = new_alpha

    elif c1 == 4 and c2 == 3:
        # Target is RGBA, source is RGB - assume source is fully opaque
        image1[:, y:y+h, x:x+w, :3] = (
            (1 - mask) * region1[:, :, :, :3] +
            mask * region2
        )
        # Alpha channel - reduce alpha where mask is applied
        image1[:, y:y+h, x:x+w, 3:4] = region1[:, :, :, 3:4] * (1 - mask) + mask

    elif c1 == 3 and c2 == 4:
        # Target is RGB, source is RGBA - apply source alpha to mask
        effective_mask = mask * region2[:, :, :, 3:4]
        image1[:, y:y+h, x:x+w, :] = (
            (1 - effective_mask) * region1 +
            effective_mask * region2[:, :, :, :3]
        )

    return


def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w/2, bbox[1] + h/2


def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def combine_masks2(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0]).astype(np.uint8)
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i]).astype(np.uint8)

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def bitwise_and_masks(mask1, mask2):
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)

    if cv2_mask1.shape == cv2_mask2.shape:
        cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
        return torch.from_numpy(cv2_mask)
    else:
        # do nothing - incompatible mask shape: mostly empty mask
        return mask1


def to_binary_mask(mask, threshold=0):
    mask = make_3d_mask(mask)

    mask = mask.clone().cpu()
    mask[mask > threshold] = 1.
    mask[mask <= threshold] = 0.
    return mask


def use_gpu_opencv():
    return not config.get_config()['disable_gpu_opencv']


def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return make_2d_mask(mask)

    mask = make_2d_mask(mask)

    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        mask = cv2.UMat(mask)
        kernel = cv2.UMat(kernel)

    if dilation_factor > 0:
        result = cv2.dilate(mask, kernel, iter)
    else:
        result = cv2.erode(mask, kernel, iter)

    if use_gpu_opencv():
        return result.get()
    else:
        return result


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        kernel = cv2.UMat(kernel)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        if use_gpu_opencv():
            cv2_mask = cv2.UMat(cv2_mask)

        if dilation_factor > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        else:
            dilated_mask = cv2.erode(cv2_mask, kernel, iter)

        if use_gpu_opencv():
            dilated_mask = dilated_mask.get()

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks

import torch.nn.functional as F
def feather_mask(mask, thickness):
    mask = mask.permute(0, 3, 1, 2)

    # Gaussian kernel for blurring
    kernel_size = 2 * int(thickness) + 1
    sigma = thickness / 3  # Adjust the sigma value as needed
    blur_kernel = _gaussian_kernel(kernel_size, sigma).to(mask.device, mask.dtype)

    # Apply blur to the mask
    blurred_mask = F.conv2d(mask, blur_kernel.unsqueeze(0).unsqueeze(0), padding=thickness)

    blurred_mask = blurred_mask.permute(0, 2, 3, 1)

    return blurred_mask

def _gaussian_kernel(kernel_size, sigma):
    # Generate a 1D Gaussian kernel
    kernel = torch.exp(-(torch.arange(kernel_size) - kernel_size // 2)**2 / (2 * sigma**2))
    return kernel / kernel.sum()


def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    _tensor_check_mask(mask)

    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size*2+1

    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest/2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask  # skip feathering

    prev_device = mask.device
    device = comfy.model_management.get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask


def subtract_masks(mask1, mask2):
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()
    cv2_mask1 = np.array(mask1) * 255
    cv2_mask2 = np.array(mask2) * 255

    if cv2_mask1.shape == cv2_mask2.shape:
        cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
        return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
    else:
        # do nothing - incompatible mask shape: mostly empty mask
        return mask1


def add_masks(mask1, mask2):
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()
    cv2_mask1 = np.array(mask1) * 255
    cv2_mask2 = np.array(mask2) * 255

    if cv2_mask1.shape == cv2_mask2.shape:
        cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
        return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
    else:
        # do nothing - incompatible mask shape: mostly empty mask
        return mask1


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


crop_tensor4 = crop_ndarray4


def crop_ndarray3(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2]

    return cropped


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_image(image, crop_region):
    return crop_tensor4(image, crop_region)


def to_latent_image(pixels, vae, vae_tiled_encode=False):
    x = pixels.shape[1]
    y = pixels.shape[2]
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]

    start = time.time()
    if vae_tiled_encode:
        encoded = nodes.VAEEncodeTiled().encode(vae, pixels, 512, overlap=64)[0] # using default settings
        logging.info(f"[Impact Pack] vae encoded (tiled) in {time.time() - start:.1f}s")
    else:
        encoded = nodes.VAEEncode().encode(vae, pixels)[0]
        logging.info(f"[Impact Pack] vae encoded in {time.time() - start:.1f}s")

    return encoded


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)

    elif len(mask.shape) == 3:
        return mask.squeeze(0)

    return mask


def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)

    return mask


def make_4d_mask(mask):
    if len(mask.shape) == 3:
        return mask.unsqueeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0).unsqueeze(0)

    return mask


def is_same_device(a, b):
    a_device = torch.device(a) if isinstance(a, str) else a
    b_device = torch.device(b) if isinstance(b, str) else b
    return a_device.type == b_device.type and a_device.index == b_device.index


def collect_non_reroute_nodes(node_map, links, res, node_id):
    if node_map[node_id]['type'] != 'Reroute' and node_map[node_id]['type'] != 'Reroute (rgthree)':
        res.append(node_id)
    else:
        for link in node_map[node_id]['outputs'][0]['links']:
            next_node_id = str(links[link][2])
            collect_non_reroute_nodes(node_map, links, res, next_node_id)


from torchvision.transforms.functional import to_pil_image


def resize_mask(mask, size):
    mask = make_4d_mask(mask)
    resized_mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear', align_corners=False)
    return resized_mask.squeeze(0)


def apply_mask_alpha_to_pil(decoded_pil, mask):
    decoded_rgba = decoded_pil.convert('RGBA')
    mask_pil = to_pil_image(mask)
    decoded_rgba.putalpha(mask_pil)

    return decoded_rgba


def flatten_mask(all_masks):
    merged_mask = (all_masks[0] * 255).to(torch.uint8)
    for mask in all_masks[1:]:
        merged_mask |= (mask * 255).to(torch.uint8)

    return merged_mask


def try_install_custom_node(custom_node_url, msg):
    try:
        import cm_global
        cm_global.try_call(api='cm.try-install-custom-node',
                           sender="Impact Pack", custom_node_url=custom_node_url, msg=msg)
    except Exception:
        logging.info(msg)
        logging.info("[Impact Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")


# author: Trung0246 --->
class TautologyStr(str):
    def __ne__(self, other):
        return False


class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item


class NonListIterable:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]


def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    # Iterate over the list of full folder paths
    for full_folder_path in full_folder_paths:
        # Use the provided function to add each model folder path
        folder_paths.add_model_folder_path(folder_name, full_folder_path)

    # Now handle the extensions. If the folder name already exists, update the extensions
    if folder_name in folder_paths.folder_names_and_paths:
        # Unpack the current paths and extensions
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        # Update the extensions set with the new extensions
        updated_extensions = current_extensions | extensions
        # Reassign the updated tuple back to the dictionary
        folder_paths.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        # If the folder name was not present, add_model_folder_path would have added it with the last path
        # Now we just need to update the set of extensions as it would be an empty set
        # Also ensure that all paths are included (since add_model_folder_path adds only one path at a time)
        folder_paths.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)
# <---

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")
