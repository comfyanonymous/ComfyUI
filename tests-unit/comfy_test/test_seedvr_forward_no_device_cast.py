from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import ast  # noqa: E402
import inspect  # noqa: E402

from torch import nn  # noqa: E402

import comfy  # noqa: E402
import comfy.ldm.seedvr.model  # noqa: E402
import comfy.model_management  # noqa: E402
from comfy.ldm.seedvr.model import MMModule  # noqa: E402


def test_no_get_torch_device_in_forward_methods():
    tree = ast.parse(inspect.getsource(comfy.ldm.seedvr.model))
    assert [
        (n.lineno, i.lineno)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "forward"
        for i in ast.walk(n)
        if isinstance(i, ast.Call)
        and isinstance(i.func, ast.Attribute)
        and i.func.attr == "get_torch_device"
    ] == []


def test_mmmodule_forward_succeeds_without_get_torch_device_lookup(monkeypatch):
    call_count = [0]

    def boom():
        call_count[0] += 1
        raise RuntimeError("MMModule.forward called get_torch_device()")

    monkeypatch.setattr(comfy.model_management, "get_torch_device", boom)

    class _IdentityCallable(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    mm = MMModule(_IdentityCallable, shared_weights=False, vid_only=False)

    vid_in = torch.zeros(2, 4)
    txt_in = torch.ones(2, 4)
    vid_out, txt_out = mm.forward(vid_in, txt_in)

    assert call_count[0] == 0
    assert torch.equal(vid_out, vid_in)
    assert torch.equal(txt_out, txt_in)
    assert vid_out.device == vid_in.device
    assert txt_out.device == txt_in.device
