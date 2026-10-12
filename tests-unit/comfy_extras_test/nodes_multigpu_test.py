import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.sd
from comfy_extras import nodes_multigpu


class FakePatcher:
    def __init__(self, model, load_device, dynamic=False):
        self.model = model
        self.load_device = load_device
        self.offload_device = torch.device("cpu")
        self.dynamic = dynamic

    def is_dynamic(self):
        return self.dynamic

    def clone(self, disable_dynamic=False):
        if self.dynamic and disable_dynamic:
            return FakePatcher(torch.nn.Linear(1, 1), self.load_device)
        return FakePatcher(self.model, self.load_device, self.dynamic)

    def deepclone_multigpu(self, new_load_device=None):
        return FakePatcher(torch.nn.Linear(1, 1), new_load_device)


@pytest.mark.parametrize("device,resolved,dynamic", [
    ("gpu:1", torch.device("cuda", 1), False),
    ("cpu", torch.device("cpu"), True),
])
def test_select_clip_device_uses_retargeted_model(monkeypatch, device, resolved, dynamic):
    monkeypatch.setattr(nodes_multigpu.comfy.model_management, "resolve_gpu_device_option", lambda _: resolved)
    model = torch.nn.Linear(1, 1)
    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = model
    clip.patcher = FakePatcher(model, torch.device("cuda", 0), dynamic)
    clip.tokenizer = None
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None

    out = nodes_multigpu.SelectCLIPDeviceNode.execute(clip, device).result[0]

    assert out.patcher.load_device == resolved
    assert out.patcher.model is not model
    assert out.cond_stage_model is out.patcher.model
    assert clip.cond_stage_model is model
