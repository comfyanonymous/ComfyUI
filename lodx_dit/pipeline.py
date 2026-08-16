"""Pipeline-split the MiniMax H3 DiT across GPUs.

For capacity, not speed.  The blocks run in order, so exactly one GPU is busy
at a time and a single generation takes as long as it would on one card -- the
sampler's steps are strictly serial and H3 is batch-size-1
(``model.py:538``), so there is nothing to fill the pipeline with.  What it buys
is that the weights no longer have to fit on one card, which matters once
streaming them from system RAM is the alternative.

Communication is one hidden state per boundary: ``(S, 5376)`` in bf16, 78 MB at
a 2 s clip and 480 MB at 15 s.  That is 1/100th of what tensor parallel would
move, which is the whole reason this split is the cheap one.

It attaches by wrapping ``DiTBlock.forward`` rather than reimplementing
``_forward``: the block loop already threads ``h`` from one block to the next,
so moving the activation on entry is enough to make it migrate.  ``ModelPatcher``
owns a single ``load_device`` and will pull the whole model back there on every
load, so the placement is re-asserted at the start of each forward -- a device
check per block, and a real move only when something has undone it.
"""

from __future__ import annotations

import logging

import torch

import comfy.ldm.minimax.model as h3

_installed = False
_state: dict = {"devices": [], "of_block": {}, "cache": {}}


def plan(n_blocks: int, devices: list) -> dict:
    """Block index -> device, contiguous and as even as the count allows."""
    n = len(devices)
    out = {}
    for i in range(n_blocks):
        out[i] = devices[min(i * n // n_blocks, n - 1)]
    return out


def _resolve(n_gpus: int) -> list:
    import comfy.model_management
    devs = [d for d in comfy.model_management.get_all_torch_devices()
            if not comfy.model_management.is_device_cpu(d)]
    if len(devs) < n_gpus:
        logging.warning("[LoD-PP] asked for %d GPUs, found %d; using what "
                        "there is", n_gpus, len(devs))
    return devs[:max(1, n_gpus)]


def _place(model, devices):
    """Move blocks onto their assigned device.  A no-op once settled."""
    n = len(model.blocks)
    mapping = plan(n, devices)
    moved = 0
    for i, block in enumerate(model.blocks):
        want = mapping[i]
        try:
            have = next(block.parameters()).device
        except StopIteration:
            continue
        if have != want:
            block.to(want)
            moved += 1
    _state["of_block"] = {id(b): mapping[i] for i, b in enumerate(model.blocks)}
    if moved:
        logging.info("[LoD-PP] placed %d/%d blocks across %s", moved, n,
                     [str(d) for d in devices])
    return mapping


def _to(tensor, device):
    """Move with a per-forward memo, so t_emb and the rope table cross once."""
    if tensor is None or tensor.device == device:
        return tensor
    key = (tensor.data_ptr(), str(device))
    got = _state["cache"].get(key)
    if got is None:
        got = _state["cache"][key] = tensor.to(device)
    return got


def install(n_gpus: int):
    global _installed
    if _installed or n_gpus <= 1:
        return
    devices = _resolve(n_gpus)
    if len(devices) <= 1:
        logging.warning("[LoD-PP] need at least two GPUs, staying single")
        return
    _state["devices"] = devices

    real_forward = h3.MiniMaxH3Model._forward
    real_block = h3.DiTBlock.forward
    real_final = h3.FinalLayer.forward

    def forward(self, *args, **kwargs):
        _state["cache"] = {}
        _place(self, devices)
        try:
            return real_forward(self, *args, **kwargs)
        finally:
            _state["cache"] = {}

    def block_forward(self, x, t_emb, mod_segments, rope_freqs, **kwargs):
        dev = _state["of_block"].get(id(self))
        if dev is None:
            return real_block(self, x, t_emb, mod_segments, rope_freqs,
                              **kwargs)
        # comfy-kitchen's HIP ops export buffers through the CURRENT device and
        # refuse a mismatch ("Can't export tensors on a different CUDA device
        # index"), so moving the tensors is not enough -- the context has to
        # follow them.
        with torch.cuda.device(dev):
            # each argument is checked on its own: the loop hands every block
            # the SAME t_emb and rope table, so they are still on the first
            # device for every block past the boundary, not just the first one
            if x.device != dev:
                x = x.to(dev)
            t_emb = _to(t_emb, dev)
            rope_freqs = _to(rope_freqs, dev)
            return real_block(self, x, t_emb, mod_segments, rope_freqs,
                              **kwargs)

    def final_forward(self, x, t_emb, video_seg, audio_seg):
        # the head and everything after it live with the rest of the model
        home = next(self.parameters()).device
        with torch.cuda.device(home):
            if x.device != home:
                x = x.to(home)
                t_emb = _to(t_emb, home)
            return real_final(self, x, t_emb, video_seg, audio_seg)

    h3.MiniMaxH3Model._forward = forward
    h3.DiTBlock.forward = block_forward
    h3.FinalLayer.forward = final_forward
    _installed = True
    logging.info("[LoD-PP] MiniMax H3 pipeline split enabled across %s",
                 [str(d) for d in devices])


__all__ = ["install", "plan"]
