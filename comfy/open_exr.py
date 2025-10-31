"""
Portions of this code are adapted from the repository
https://github.com/spacepxl/ComfyUI-HQ-Image-Save

MIT License

Copyright (c) 2023 spacepxl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from .component_model.images_types import ImageMaskTuple


read_exr = lambda fp: cv2.imread(fp, cv2.IMREAD_UNCHANGED).astype(np.float32) 

def mut_srgb_to_linear(np_array) -> None:
    less = np_array <= 0.0404482362771082
    np_array[less] = np_array[less] / 12.92
    np_array[~less] = np.power((np_array[~less] + 0.055) / 1.055, 2.4)


def mut_linear_to_srgb(np_array) -> None:
    less = np_array <= 0.0031308
    np_array[less] = np_array[less] * 12.92
    np_array[~less] = np.power(np_array[~less], 1 / 2.4) * 1.055 - 0.055


def load_exr(file_path: str, srgb: bool) -> ImageMaskTuple:
    image = read_exr(file_path)
    rgb = np.flip(image[:, :, :3], 2).copy()
    if srgb:
        mut_linear_to_srgb(rgb)
        rgb = np.clip(rgb, 0, 1)
    rgb = torch.unsqueeze(torch.from_numpy(rgb), 0)

    mask = torch.zeros((1, image.shape[0], image.shape[1]), dtype=torch.float32)
    if image.shape[2] > 3:
        mask[0] = torch.from_numpy(np.clip(image[:, :, 3], 0, 1))

    return ImageMaskTuple(rgb, mask)


def load_exr_latent(file_path: str) -> Tuple[Tensor]:
    image = read_exr(file_path)
    image = image[:, :, np.array([2, 1, 0, 3])]
    image = torch.unsqueeze(torch.from_numpy(image), 0)
    image = torch.movedim(image, -1, 1)
    return image,


def save_exr(images: Tensor, filepaths_batched: Sequence[str], colorspace="linear"):
    linear = images.detach().clone().cpu().numpy().astype(np.float32)
    if colorspace == "linear":
        mut_srgb_to_linear(linear[:, :, :, :3])  # only convert RGB, not Alpha

    bgr = copy.deepcopy(linear)
    bgr[:, :, :, 0] = linear[:, :, :, 2]  # flip RGB to BGR for opencv
    bgr[:, :, :, 2] = linear[:, :, :, 0]
    if bgr.shape[-1] > 3:
        bgr[:, :, :, 3] = np.clip(1 - linear[:, :, :, 3], 0, 1)  # invert alpha

    for i in range(len(linear.shape[0])):
        cv2.imwrite(filepaths_batched[i], bgr[i]) 
