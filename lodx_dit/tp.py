"""Tensor-parallel the MiniMax H3 DiT across GPUs.  Nothing in comfy changes.

``tp_probe.py`` established the arithmetic on one block; this drives all 50 so a
real generation can be compared against the single-GPU one.

Why it patches ``DiTBlock.forward`` and not the two Linears
-----------------------------------------------------------
Splitting only ``Attention`` and ``MLP`` would mean shipping the normed
activation to the other GPU before each one -- two extra crossings per block on
top of the two all-reduces, doubling the traffic that section 8 of the results
doc says is already the binding constraint.  Keeping ``x`` replicated on every
GPU and running the cheap parts (norms, the modulation) redundantly costs 2.9%
of duplicated compute (see the profile in results section 2.3) and leaves
exactly the two all-reduces TP actually requires.

``adaln_proj`` is the exception.  Its weight is 260 MB a block -- 13 GB over the
model, too much to replicate -- but its *output* is six tensors of [3, 5376].
So it runs on GPU 0 only and the outputs are broadcast: 194 KB a block against
the 159 MB an activation crossing would cost.

Where the weights come from
---------------------------
Straight out of the checkpoint file, not out of the loaded model.  Reading the
shards directly means GPU 0 never has to hold the unsplit 34 GB, and it works
whether or not DynamicVRAM is streaming the rest of the model.

Only the four big linears are sharded; everything else (norms, adaln, patchify,
the final layer, the token refiner) stays exactly as comfy loaded it.
"""

from __future__ import annotations

import logging
import time

import torch

import comfy.ldm.minimax.model as h3
import comfy.quant_ops as quant_ops

ck = quant_ops.ck
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

HEAD_DIM = 128
GROUP = 256
CKPT = "minimax_h3_fl2va_int8_convrot.safetensors"

_installed = False
_S: dict = {"devices": [], "shards": {}, "index": {}, "mode": "int8",
            "replica": {}, "loaded": False, "xfer": {}, "exact": False,
            "release": False}


def _to(tensor, device):
    """Move with a per-forward memo: the block loop hands every block the SAME
    rope table, so without this it would cross once per block."""
    if tensor is None or tensor.device == device:
        return tensor
    key = (tensor.data_ptr(), str(device))
    got = _S["xfer"].get(key)
    if got is None:
        got = _S["xfer"][key] = tensor.to(device)
    return got


# --------------------------------------------------------------- sharding

def _rows(qdata, scale, pieces, part, n):
    """Same fractional slice out of each stacked piece of the output axis.

    ``qkv_proj`` holds [q|k|v] and ``fc1`` holds [gate|up]; a GPU's share is one
    slice per piece, not one slice of the whole stack.
    """
    idx = torch.cat([torch.arange(start + part * (size // n),
                                  start + (part + 1) * (size // n))
                     for start, size in pieces])
    return qdata.index_select(0, idx).contiguous(), scale.index_select(0, idx).contiguous()


def _cols(qdata, part, n):
    """Slice the input axis, on a convrot group boundary."""
    step = qdata.size(1) // n
    if step % GROUP:
        raise RuntimeError(f"input split {step} is not a multiple of {GROUP}")
    return qdata[:, part * step:(part + 1) * step].contiguous()


def _release(blk):
    """Drop the block's own copies of the four linears we replaced.

    OFF by default, because it does not work under DynamicVRAM: the weights are
    owned by the aimdo arena rather than the module, so deleting the attribute
    frees nothing and then breaks the streamer's bookkeeping
    ("HostBuffer.truncate failed").  Only useful with --disable-dynamic-vram,
    where the tensors really are the module's.
    """
    for mod in (blk.attn.qkv_proj, blk.attn.out_proj, blk.mlp.fc1, blk.mlp.fc2):
        try:
            if "weight" in mod._parameters:
                del mod._parameters["weight"]
            mod.weight = None
        except Exception as e:                      # keep going; it only costs VRAM
            logging.debug("[LoD-TP] could not release a weight: %s", e)


def _load_shards(model, devices):
    """Read every block's four linears from the checkpoint, split, and place."""
    import folder_paths
    from safetensors import safe_open

    n = len(devices)
    path = folder_paths.get_full_path_or_raise("diffusion_models", CKPT)
    heads = model.blocks[0].attn.heads
    inner = heads * HEAD_DIM
    ffn = model.blocks[0].mlp.fc1.weight.shape[0] // 2   # read before _release
    t0 = time.time()
    total = 0

    with safe_open(path, framework="pt", device="cpu") as f:
        for bi in range(len(model.blocks)):
            per_gpu = []
            qkv = (f.get_tensor(f"blocks.{bi}.attn.qkv_proj.weight"),
                   f.get_tensor(f"blocks.{bi}.attn.qkv_proj.weight_scale"))
            out = (f.get_tensor(f"blocks.{bi}.attn.out_proj.weight"),
                   f.get_tensor(f"blocks.{bi}.attn.out_proj.weight_scale"))
            fc1 = (f.get_tensor(f"blocks.{bi}.mlp.fc1.weight"),
                   f.get_tensor(f"blocks.{bi}.mlp.fc1.weight_scale"))
            fc2 = (f.get_tensor(f"blocks.{bi}.mlp.fc2.weight"),
                   f.get_tensor(f"blocks.{bi}.mlp.fc2.weight_scale"))
            for p, dev in enumerate(devices):
                qq, qs = _rows(*qkv, [(0, inner), (inner, inner), (2 * inner, inner)], p, n)
                fq, fs = _rows(*fc1, [(0, ffn), (ffn, ffn)], p, n)
                with torch.cuda.device(dev):
                    sh = dict(
                        qkv=(qq.to(dev), qs.to(dev)),
                        out=(_cols(out[0], p, n).to(dev), out[1].to(dev)),
                        fc1=(fq.to(dev), fs.to(dev)),
                        fc2=(_cols(fc2[0], p, n).to(dev), fc2[1].to(dev)),
                        heads=heads // n, device=dev)
                total += sum(t.numel() * t.element_size()
                             for pair in (sh["qkv"], sh["out"], sh["fc1"], sh["fc2"])
                             for t in pair)
                per_gpu.append(sh)
            _S["shards"][bi] = per_gpu
            if _S["release"]:
                _release(model.blocks[bi])
            del qkv, out, fc1, fc2
    import gc
    gc.collect()
    for d in devices:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()
    logging.info("[LoD-TP] sharded %d blocks, %.1f GiB total, %.1fs",
                 len(model.blocks), total / 2**30, time.time() - t0)


def _norm_weight(param, device):
    """Materialise a norm weight on ``device``.

    Reading ``.weight`` directly is wrong: with DynamicVRAM the parameter is a
    streaming placeholder until something casts it, so a plain ``.to()`` copies
    whatever happens to be there.  ``cast_to`` is what the model itself uses
    (comfy/ldm/minimax/model.py:166).  These are 10 KB each, so doing it per
    block per forward costs nothing.
    """
    import comfy.model_management
    return comfy.model_management.cast_to(param, device=device)


# --------------------------------------------------------------- all-reduce

def _allreduce(parts, mode):
    """Sum tensors that live on different GPUs, returning the sum on each.

    No P2P here, so every byte goes through host memory; the only lever is how
    many bytes cross.  ``int8`` and ``fp8`` both carry a per-row scale -- raw
    e4m3 without one measured 1.9e-1 relative error, which is unusable.
    """
    n = len(parts)
    if n == 1:
        return parts
    if mode == "bf16":
        wire, scales = parts, [None] * n
    elif mode == "int8":
        wire, scales = [], []
        for p in parts:
            sc = p.abs().amax(dim=-1, keepdim=True).float().clamp_(min=1e-12) / 127.0
            wire.append((p.float() / sc).round_().clamp_(-127, 127).to(torch.int8))
            scales.append(sc)
    elif mode == "fp8":
        wire, scales = [], []
        for p in parts:
            sc = p.abs().amax(dim=-1, keepdim=True).float().clamp_(min=1e-12) / 448.0
            wire.append((p.float() / sc).to(torch.float8_e4m3fn))
            scales.append(sc)
    else:
        raise RuntimeError(f"unknown all-reduce mode {mode}")

    outs = []
    for i, dst in enumerate(parts):
        with torch.cuda.device(dst.device):
            acc = dst.float()
            for j in range(n):
                if i == j:
                    continue
                got = wire[j].to(dst.device, non_blocking=True)
                if scales[j] is None:
                    acc = acc + got.to(torch.float32)
                else:
                    acc = acc + got.to(torch.float32) * scales[j].to(dst.device)
            outs.append(acc.to(dst.dtype))
    return outs


# --------------------------------------------------------------- exact rows

def _row_split(parts_x, qws, ws, devs):
    """Row-split linear that is EXACT, by sharing the activation scale.

    ``int8_linear`` quantizes its input per row over the whole row.  Split the
    input and each GPU computes a scale from its own slice, which is a different
    (finer) quantization -- measured at ~9e-3 a layer.  Sharing one scale makes
    the split reproduce the unsplit result to 0.000e+00.

    The scale has to be the max over every slice, so this costs one extra
    all-reduce -- of ``(S,)`` floats, 62 KB, against the 159 MB one that
    follows.  The rotation is done here in torch rather than in the fused HIP
    quantizer, which is what leaves ~7e-3 against stock; for scale, the
    library's own eager and HIP backends disagree by 8.6e-3 on the same call.
    """
    h = {}
    rot, mx = [], []
    for p, dev in enumerate(devs):
        with torch.cuda.device(dev):
            if dev not in h:
                h[dev] = _build_hadamard(GROUP, dev, torch.float32)
            r = _rotate_activation(parts_x[p].float(), h[dev], GROUP)
            rot.append(r)
            mx.append(r.abs().amax(-1, keepdim=True))
    # shared scale: the max across every slice, gathered onto each GPU
    shared = []
    for p, dev in enumerate(devs):
        with torch.cuda.device(dev):
            m = mx[p]
            for j in range(len(devs)):
                if j != p:
                    m = torch.maximum(m, mx[j].to(dev, non_blocking=True))
            shared.append((m.float() / 127.0).clamp_(min=1e-30))
    outs = []
    for p, dev in enumerate(devs):
        with torch.cuda.device(dev):
            xq = (rot[p] / shared[p]).round_().clamp_(-127, 127).to(torch.int8)
            acc = ck.mm_int8(xq, qws[p].T).float()
            outs.append((acc * shared[p] * ws[p].reshape(1, -1)).to(torch.bfloat16))
    return outs


# --------------------------------------------------------------- block

def _rms(x, weight, eps):
    """Exactly what comfy's RMSNorm does (comfy/ops.py:687).

    A hand-rolled fp32 version differs by only 1.4e-5, but that is a relative
    perturbation entering a 50-block residual stack: measured, it comes out at
    2.3e-2 after one DiT forward and 1.2e-1 after a 2-step trajectory.  Deep
    residual nets amplify, so "close enough" in a norm is not close enough.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def _attn_partial(sh, x, rope_freqs, eps, transformer_options, qn, kn,
                  exact=False):
    heads = sh["heads"]
    s = x.shape[0]
    qkv = ck.int8_linear(x, sh["qkv"][0], sh["qkv"][1], None, x.dtype,
                         convrot=True, convrot_groupsize=GROUP)
    q, k, v = qkv.split(heads * HEAD_DIM, dim=-1)
    v = v.view(s, heads, HEAD_DIM)
    if rope_freqs is not None:
        q = q.view(1, s, heads, HEAD_DIM)
        k = k.view(1, s, heads, HEAD_DIM)
        rot = rope_freqs.shape[-3] * 2
        ck.rms_rope_split_half_(q, k, rope_freqs, qn, kn,
                                epsilon=eps, rot_dim=rot)
        q, k = q[0], k[0]
    else:
        q = _rms(q.view(s, heads, HEAD_DIM), qn, eps)
        k = _rms(k.view(s, heads, HEAD_DIM), kn, eps)
    v = v.clone()
    C = h3.AttentionTensorContainer
    out = h3.optimized_attention(
        C(q.transpose(0, 1).unsqueeze(0)), C(k.transpose(0, 1).unsqueeze(0)),
        C(v.transpose(0, 1).unsqueeze(0)), heads, mask=None, skip_reshape=True,
        transformer_options=transformer_options)
    a = out.squeeze(0)
    if exact:
        return a                      # caller does the shared-scale row split
    return ck.int8_linear(a, sh["out"][0], sh["out"][1], None,
                          x.dtype, convrot=True, convrot_groupsize=GROUP)


def _mlp_partial(sh, x, exact=False):
    hid = ck.int8_linear(x, sh["fc1"][0], sh["fc1"][1], None, x.dtype,
                         convrot=True, convrot_groupsize=GROUP)
    if exact:
        # swiglu has to happen here: the shared-scale path quantizes the
        # ACTIVATED row, which is what the fused input_act does inside
        g, u = hid.chunk(2, dim=-1)
        return torch.nn.functional.silu(g).mul_(u)
    return ck.int8_linear(hid, sh["fc2"][0], sh["fc2"][1], None, x.dtype,
                          convrot=True, convrot_groupsize=GROUP,
                          input_act="swiglu")


def install(n_gpus: int, mode: str = "int8", exact_rows: bool = False,
            release: bool = False):
    """``n_gpus=1`` is the self-check: same code path, one shard, no all-reduce.

    It should reproduce the stock model almost exactly.  If it does not, the
    bug is in this file's re-implementation of the block, not in the split.
    """
    global _installed
    if _installed or n_gpus < 1:
        return
    import comfy.model_management
    devs = [d for d in comfy.model_management.get_all_torch_devices()
            if not comfy.model_management.is_device_cpu(d)][:n_gpus]
    if len(devs) < n_gpus:
        logging.warning("[LoD-TP] asked for %d GPUs, found %d", n_gpus, len(devs))
        return
    _S["devices"], _S["mode"], _S["exact"] = devs, mode, exact_rows
    _S["release"] = release

    real_forward = h3.MiniMaxH3Model._forward
    real_block = h3.DiTBlock.forward

    def forward(self, *args, **kwargs):
        if not _S["loaded"]:
            _load_shards(self, devs)
            _S["index"] = {id(b): i for i, b in enumerate(self.blocks)}
            _S["loaded"] = True
            logging.info("[LoD-TP] %d-way tensor parallel, all-reduce in %s",
                         len(devs), mode)
        _S["replica"].clear(); _S["xfer"].clear()
        try:
            return real_forward(self, *args, **kwargs)
        finally:
            _S["replica"].clear(); _S["xfer"].clear()

    def block_forward(self, x, t_emb, mod_segments, rope_freqs,
                      transformer_options={}):
        bi = _S["index"].get(id(self))
        if bi is None:
            return real_block(self, x, t_emb, mod_segments, rope_freqs,
                              transformer_options)
        shards = _S["shards"][bi]
        eps = self.attn.q_norm.eps

        # adaln runs on GPU 0 only; its outputs are small enough to broadcast
        mods = self.adaln_proj(t_emb)
        xs = [x] + [_S["replica"].get(p) for p in range(1, len(devs))]
        for p in range(1, len(devs)):
            if xs[p] is None:
                xs[p] = x.to(devs[p])
        mods_per = [mods] + [[m.to(devs[p]) for m in mods]
                             for p in range(1, len(devs))]

        hs = []
        for p, dev in enumerate(devs):
            with torch.cuda.device(dev):
                sm, sc = mods_per[p][0], mods_per[p][1]
                hs.append(h3._mod_scale_shift(
                    _rms(xs[p], _norm_weight(self.norm1.weight, dev),
                         self.norm1.eps), sm, sc, mod_segments))
        # comfy-kitchen's HIP ops export buffers through the CURRENT device and
        # refuse a mismatch, so every shard's compute needs its context too --
        # not just the tensors being on the right card
        parts = []
        for p, dev in enumerate(devs):
            with torch.cuda.device(dev):
                parts.append(_attn_partial(
                    shards[p], hs[p], _to(rope_freqs, dev), eps,
                    transformer_options,
                    _norm_weight(self.attn.q_norm.weight, dev),
                    _norm_weight(self.attn.k_norm.weight, dev),
                    exact=_S["exact"]))
        if _S["exact"]:
            parts = _row_split(parts, [shards[p]["out"][0] for p in range(len(devs))],
                               [shards[p]["out"][1] for p in range(len(devs))], devs)
        attn = _allreduce(parts, _S["mode"])
        for p, dev in enumerate(devs):
            with torch.cuda.device(dev):
                xs[p] = h3._mod_gate(xs[p], mods_per[p][2], attn[p], mod_segments)

        hs = []
        for p, dev in enumerate(devs):
            with torch.cuda.device(dev):
                hs.append(h3._mod_scale_shift(
                    _rms(xs[p], _norm_weight(self.norm2.weight, dev),
                         self.norm2.eps),
                    mods_per[p][3], mods_per[p][4], mod_segments))
        parts = []
        for p, dev in enumerate(devs):
            with torch.cuda.device(dev):
                parts.append(_mlp_partial(shards[p], hs[p], exact=_S["exact"]))
        if _S["exact"]:
            parts = _row_split(parts, [shards[p]["fc2"][0] for p in range(len(devs))],
                               [shards[p]["fc2"][1] for p in range(len(devs))], devs)
        mlp = _allreduce(parts, _S["mode"])
        for p, dev in enumerate(devs):
            with torch.cuda.device(dev):
                xs[p] = h3._mod_gate(xs[p], mods_per[p][5], mlp[p], mod_segments)

        for p in range(1, len(devs)):
            _S["replica"][p] = xs[p]
        return xs[0]

    h3.MiniMaxH3Model._forward = forward
    h3.DiTBlock.forward = block_forward
    _installed = True


def free():
    """Drop the shards.  The VAE needs the room, and these are not tracked by
    comfy's model manager so ``unload_all_models`` will not touch them."""
    import gc
    _S["shards"].clear()
    _S["replica"].clear()
    _S["xfer"].clear()
    gc.collect()
    torch.cuda.empty_cache()
    _S["loaded"] = False


__all__ = ["install", "free"]
