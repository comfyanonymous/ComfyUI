import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import shutil
import torch
import torch.nn.functional as F
import PIL
import torchvision.transforms.functional as transform
from vfi_utils import load_file_from_github_release
from vfi_models import gmfss_fortuna, ifrnet, ifunet, m2m, rife, sepconv, amt, xvfi, cain, flavr
import numpy as np

frame_0 = torch.from_numpy(np.array(PIL.Image.open("demo_frames/anime0.png").convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
frame_1 = torch.from_numpy(np.array(PIL.Image.open("demo_frames/anime1.png").convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)


if os.path.exists("test_result"):
    shutil.rmtree("test_result")

vfi_node_class = gmfss_fortuna.GMFSS_Fortuna_VFI()
for i, ckpt_name in enumerate(vfi_node_class.INPUT_TYPES()["required"]["ckpt_name"][0][:2]):
    result = vfi_node_class.vfi(ckpt_name, torch.cat([
        frame_0,
        frame_1,
        frame_0,
        frame_1
    ], dim=0).cuda(), multipler=4, batch_size=2)[0]
    print(result.shape)
    print(f"Generated {result.size(0)} frames")
    frames = [PIL.Image.fromarray(np.clip((frame * 255).numpy(), 0, 255).astype(np.uint8)) for frame in result]
    print(result[0].shape)
    os.makedirs(f"test_result/video{i}", exist_ok=True)
    for j, frame in enumerate(frames):
        frame.save(f"test_result/video{i}/{j}.jpg")
    frames[0].save(f"test_result/video{i}.gif", save_all=True, append_images=frames[1:], optimize=True, duration=1/3, loop=0)
    os.startfile(f"test_result{os.path.sep}video{i}.gif")
#torchvision.io.video.write_video("test.mp4", einops.rearrange(result, "n c h w -> n h w c").cpu(), fps=1)