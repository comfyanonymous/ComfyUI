import os
from PIL import Image
import numpy as np
import torch
from torchvision.utils import make_grid
import cv2
import math
import logging
import hashlib
from insightface.app.common import Face
from safetensors.torch import save_file, safe_open
from tqdm import tqdm
import urllib.request
import onnxruntime
from typing import Any
import folder_paths
from comfy.utils import ProgressBar

ORT_SESSION = None

def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

# def tensor_to_pil(img, batch_index=0):
#     """Безопасное преобразование тензора в PIL.Image с обработкой особых случаев"""
#     try:
#         # Обработка пакетных данных
#         if isinstance(img, torch.Tensor):
#             if len(img.shape) == 4:
#                 img = img[batch_index]  # Выбор элемента батча
#             img = img.detach().cpu().numpy()
        
#         # Нормализация и приведение типа
#         if img.dtype == np.float32:
#             img = np.clip(255. * img, 0, 255).astype(np.uint8)
        
#         # Обработка нестандартных размерностей
#         if img.shape[-1] > 4:  # Если каналов больше 4
#             img = img[..., :3]  # Берем первые 3 канала
        
#         # Преобразование в 2D/3D массив
#         if len(img.shape) == 3 and img.shape[0] == 1:  # [C, H, W] → [H, W]
#             img = img.squeeze(0)
#         elif len(img.shape) == 3 and img.shape[2] == 1:  # [H, W, C=1]
#             img = img.squeeze(-1)
        
#         return Image.fromarray(img)
    
#     except Exception as e:
#         raise RuntimeError(f"Невозможно преобразовать тензор формы {img.shape} в PIL.Image") from e


def batch_tensor_to_pil(img_tensor):
    # Convert tensor of shape [batch_size, channels, height, width] to a list of PIL Images
    return [tensor_to_pil(img_tensor, i) for i in range(img_tensor.shape[0])]


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def batched_pil_to_tensor(images):
    # Takes a list of PIL images and returns a tensor of shape [batch_size, height, width, channels]
    return torch.cat([pil_to_tensor(image) for image in images], dim=0)


def img2tensor(imgs, bgr2rgb=True, float32=True):

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):

    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def rgba2rgb_tensor(rgba):
    r = rgba[...,0]
    g = rgba[...,1]
    b = rgba[...,2]
    return torch.stack([r, g, b], dim=3)


def download(url, path, name):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc=f'[ReActor] Downloading {name} to {path}', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))


def move_path(old_path, new_path):
    if os.path.exists(old_path):
        try:
            models = os.listdir(old_path)
            for model in models:
                move_old_path = os.path.join(old_path, model)
                move_new_path = os.path.join(new_path, model)
                os.rename(move_old_path, move_new_path)
            os.rmdir(old_path)
        except Exception as e:
            print(f"Error: {e}")
            new_path = old_path


def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def get_image_md5hash(image: Image.Image):
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()


def save_face_model(face: Face, filename: str) -> None:
    try:
        tensors = {
            "bbox": torch.tensor(face["bbox"]),
            "kps": torch.tensor(face["kps"]),
            "det_score": torch.tensor(face["det_score"]),
            "landmark_3d_68": torch.tensor(face["landmark_3d_68"]),
            "pose": torch.tensor(face["pose"]),
            "landmark_2d_106": torch.tensor(face["landmark_2d_106"]),
            "embedding": torch.tensor(face["embedding"]),
            "gender": torch.tensor(face["gender"]),
            "age": torch.tensor(face["age"]),
        }
        save_file(tensors, filename)
        print(f"Face model has been saved to '{filename}'")
    except Exception as e:
        print(f"Error: {e}")


def load_face_model(filename: str):
    face = {}
    with safe_open(filename, framework="pt") as f:
        for k in f.keys():
            face[k] = f.get_tensor(k).numpy()
    return Face(face)


def get_ort_session():
    global ORT_SESSION
    return ORT_SESSION

def set_ort_session(model_path, providers) -> Any:
    global ORT_SESSION
    onnxruntime.set_default_logger_severity(3)
    ORT_SESSION = onnxruntime.InferenceSession(model_path, providers=providers)
    return ORT_SESSION

def clear_ort_session() -> None:
    global ORT_SESSION
    ORT_SESSION = None

def prepare_cropped_face(cropped_face):
	cropped_face = cropped_face[:, :, ::-1] / 255.0
	cropped_face = (cropped_face - 0.5) / 0.5
	cropped_face = np.expand_dims(cropped_face.transpose(2, 0, 1), axis = 0).astype(np.float32)
	return cropped_face

def normalize_cropped_face(cropped_face):
	cropped_face = np.clip(cropped_face, -1, 1)
	cropped_face = (cropped_face + 1) / 2
	cropped_face = cropped_face.transpose(1, 2, 0)
	cropped_face = (cropped_face * 255.0).round()
	cropped_face = cropped_face.astype(np.uint8)[:, :, ::-1]
	return cropped_face


def progress_bar(total):
    return ProgressBar(total)

def progress_bar_reset(pbar):
    pbar.current = 0
    pbar.update(0)


# author: Trung0246 --->
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
