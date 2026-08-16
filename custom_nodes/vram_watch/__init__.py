"""Catch VRAM that survives a prompt.  Enable with COMFYUI_VRAM_WATCH=1.

Written to chase an intermittent report: a job finishes, the next one OOMs.
It does not assume a cause -- it records what is actually held.

Three things get watched, because they fail differently:

allocated across prompts
    Live tensors.  If this does not return to its pre-prompt value, something
    still owns them.  ``gc.collect()`` and ``soft_empty_cache()`` (which
    main.py:426 runs between jobs) do NOT free tensors that autograd is holding
    -- those live in C++ nodes, invisible to ``gc.get_objects()``.

reserved minus allocated
    Allocator fragmentation.  Grows when block sizes vary between runs.  This
    one IS recoverable by ``empty_cache`` and is not a leak.

inference mode
    ``execution.py:751`` is the ONLY place ComfyUI disables autograd, and
    ``torch.inference_mode`` is thread-local.  Any model work that reaches a
    different thread runs with autograd on and retains its whole graph -- about
    9 GiB for one MiniMax H3 VAE decode, measured.  This flags it if it happens.

On a prompt that ends heavier than it started, it lists the largest live CUDA
tensors and what holds them, so the next step is reading a name rather than
guessing.
"""

import gc
import logging
import os

_ENABLED = os.environ.get("COMFYUI_VRAM_WATCH", "") not in ("", "0", "false")

if _ENABLED:
    import torch

    import comfy.sd
    import execution

    _GIB = 1024 ** 3
    _state = {"base": None, "n": 0, "no_inference": set()}

    def _mem():
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    def _check_inference(where):
        """Model work outside inference mode keeps its autograd graph alive."""
        if not torch.is_inference_mode_enabled() and torch.is_grad_enabled():
            if where not in _state["no_inference"]:
                _state["no_inference"].add(where)
                logging.warning(
                    "[vram-watch] %s ran with autograd ENABLED. Its activation "
                    "graph will be retained and gc cannot free it.", where)

    def _dump_holders(limit=8):
        big = []
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    n = o.numel() * o.element_size()
                    if n > 32 * 1024 ** 2:
                        big.append((n, o))
            except Exception:
                pass
        big.sort(key=lambda p: -p[0])
        visible = sum(n for n, _ in big)
        logging.warning("[vram-watch] %d live CUDA tensors >32 MiB, %.2f GiB "
                        "visible to gc", len(big), visible / _GIB)
        for n, t in big[:limit]:
            holders = []
            for r in gc.get_referrers(t)[:4]:
                if isinstance(r, dict):
                    owners = gc.get_referrers(r)
                    holders.append("attrs of " +
                                   (type(owners[0]).__name__ if owners else "?"))
                else:
                    holders.append(type(r).__name__)
            logging.warning("[vram-watch]   %7.1f MiB %-22s grad_fn=%-18s %s",
                            n / 1024 ** 2, str(tuple(t.shape)),
                            type(t.grad_fn).__name__ if t.grad_fn else "None",
                            holders)
        return visible

    # --- per-prompt accounting -------------------------------------------
    _real_execute = execution.PromptExecutor.execute

    def execute(self, *args, **kwargs):
        if not torch.cuda.is_available():
            return _real_execute(self, *args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        before = _mem()
        try:
            return _real_execute(self, *args, **kwargs)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            after = _mem()
            _state["n"] += 1
            grew = after[0] - before[0]
            logging.info(
                "[vram-watch] prompt %d: allocated %.2f -> %.2f GiB (%+.2f), "
                "reserved %.2f -> %.2f GiB", _state["n"], before[0] / _GIB,
                after[0] / _GIB, grew / _GIB, before[1] / _GIB, after[1] / _GIB)
            if _state["base"] is None:
                _state["base"] = after[0]
            elif after[0] > _state["base"] + 512 * 1024 ** 2:
                logging.warning(
                    "[vram-watch] %.2f GiB more is held than after the first "
                    "prompt -- this is what the next job will not have.",
                    (after[0] - _state["base"]) / _GIB)
                visible = _dump_holders()
                hidden = after[0] - visible
                if hidden > 512 * 1024 ** 2:
                    logging.warning(
                        "[vram-watch] %.2f GiB is held by something gc cannot "
                        "see -- that signature is a retained autograd graph.",
                        hidden / _GIB)

    execution.PromptExecutor.execute = execute

    # --- did any model work escape inference mode? ------------------------
    for _cls, _name in ((comfy.sd.VAE, "decode"), (comfy.sd.VAE, "encode")):
        _real = getattr(_cls, _name)

        def _wrap(real=_real, label=f"VAE.{_name}"):
            def inner(*a, **k):
                _check_inference(label)
                return real(*a, **k)
            return inner

        setattr(_cls, _name, _wrap())

    logging.info("[vram-watch] active: per-prompt VRAM accounting + "
                 "inference-mode checks")
