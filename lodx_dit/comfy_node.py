"""ComfyUI node: run MiniMax H3's attention through the LoD read.

The point is the ablation.  Insert this node and the DiT reads sparsely; bypass
it and nothing is patched at all, so the same workflow, seed and sampler give
the dense reference.  Everything the node needs it derives from the model
itself -- resolution, frame count, where the video rows start -- so the two
branches differ in exactly one thing.

How it attaches, and why this way:

``comfy/ldm/minimax/model.py:184`` calls ``optimized_attention`` with
``transformer_options`` already threaded through, and AGENTS.md says callers
must treat the selected attention backend as opaque.  So the model is not
edited: a wrapper on ``optimized_attention`` reads its config out of
``transformer_options``, and a wrapper on ``MiniMaxH3Model._forward`` publishes
the one thing the attention call cannot know -- the video grid.  With no config
present both wrappers are pass-throughs, which is what makes bypassing the node
a true dense run rather than a differently-configured LoD run.

Deriving ``prefix``: the packed sequence is ``[text | cond | ref | audio |
video]`` and video is always the last segment (``PackedLayout.__init__``), so
the conditioning rows are simply everything before the video rows.  The token
refiner calls the same attention over the text rows alone; that call is shorter
than the video block and is left dense.
"""

from __future__ import annotations

import logging

import torch

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as h3
from comfy.ldm.modules.attention import AttentionTensorContainer
from comfy_api.latest import ComfyExtension, io

from .kernel import HAVE_TRITON
from .kernel_exp import VARIANTS
from .lod_dit import lod_attention
from .ordering import best_tile, sequence_order

_installed = False
_orders: dict = {}
_run = {"sigma": -1.0}
#: Config for graphs that do not route MODEL through the node.  ComfyUI only
#: runs a node whose output is consumed, so the node is an output node and can
#: sit unconnected; when it does, this is what the wrapper falls back to.  A
#: MODEL routed through the node still wins, because that config travels with
#: the model clone and cannot be stale.
_global_cfg: dict | None = None


def _report(cfg, grid, text_len, sigma):
    """One line per sampling run saying what the read will actually do.

    Printed from the model's own forward rather than from the node, because
    what matters is whether the sparse read is running, not whether the node
    was configured -- and because the sequence length and page count are only
    known here.
    """
    if cfg["mode"] != "lod":
        logging.info("[LoD] mode=dense (stock read)  contiguous_qkv=%s",
                     cfg["contiguous_qkv"])
        return
    if grid is None:
        logging.info("[LoD] mode=lod but INACTIVE at sigma %.3f "
                     "(window %.3f..%.3f) -- dense for this step",
                     sigma, cfg["sigma_end"], cfg["sigma_start"])
        return

    video = grid[0] * grid[1] * grid[2]
    tile, page_size = best_tile(grid, cfg["page_size"])
    if not cfg["tiled_pages"]:
        tile, page_size = None, cfg["page_size"]
    n_pages = video // page_size
    leaves = min(cfg["top_pages"], n_pages) * page_size
    logging.info(
        "[LoD] mode=lod  grid=%s video_rows=%d text_rows=%d  "
        "pages=%d page_size=%d tile=%s", grid, video, text_len, n_pages,
        page_size, tile)
    logging.info(
        "[LoD]   top_pages=%d select_block=%d local_radius=%d tiled_pages=%s "
        "contiguous_qkv=%s kernel=%s window=%.3f..%.3f",
        cfg["top_pages"], cfg["select_block"], cfg["local_radius"],
        cfg["tiled_pages"], cfg["contiguous_qkv"], cfg["kernel_variant"],
        cfg["sigma_end"], cfg["sigma_start"])
    logging.info(
        "[LoD]   leaves %d + summaries %d = %d of %d video rows (%.1f%%) "
        "-- the conditioning rows are always read exactly",
        leaves, n_pages, leaves + n_pages, video,
        100.0 * (leaves + n_pages) / max(video, 1))
    if leaves >= video:
        logging.warning("[LoD]   top_pages covers every page: this is a dense "
                        "read with extra bookkeeping. Lower top_pages.")
    if tile is None:
        logging.warning("[LoD]   grid %s has no exact spatial tile (prime "
                        "sides); falling back to raster pages of %d.",
                        grid, page_size)
    elif max(tile[1], tile[2]) > 4 * min(tile[1], tile[2]):
        logging.warning("[LoD]   tile %s is a thin strip -- the grid admits no "
                        "squarer block. Selection quality may suffer.", tile)
    if cfg["sigma_end"] >= cfg["sigma_start"]:
        logging.warning("[LoD]   window %.3f..%.3f is empty or a single point: "
                        "LoD will be off for almost every step. Check "
                        "start_percent/end_percent (0.0 and 1.0 = always on).",
                        cfg["sigma_end"], cfg["sigma_start"])


def _order_for(grid, tile, seq, prefix, device):
    key = (grid, tile, seq, prefix, str(device))
    order = _orders.get(key)
    if order is None:
        order = sequence_order(seq, prefix, grid, tile, device=device)
        _orders[key] = order
    return order


def _install():
    """Wrap the two functions once, idempotently."""
    global _installed
    if _installed:
        return
    real_attn = h3.optimized_attention
    real_forward = h3.MiniMaxH3Model._forward

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        cfg = transformer_options.get("lod") or _global_cfg
        sigma = float(timestep.flatten()[0] / 1000.0)
        if cfg is not None:
            if cfg["sigma_start"] is None:
                # unconnected node: resolve the window against this model's own
                # flow shift, the same map ModelSamplingDiscreteFlow uses
                shift = float(transformer_options.get(
                    "minimax_h3_sigma_shift_video", self.sigma_shift_video))
                snr = lambda p: shift * (1.0 - p) / (1.0 + (shift - 1.0) * (1.0 - p))
                cfg["sigma_start"] = 1.0 if cfg["start_percent"] <= 0.0 else snr(cfg["start_percent"])
                cfg["sigma_end"] = 0.0 if cfg["end_percent"] >= 1.0 else snr(cfg["end_percent"])
            active = cfg["mode"] == "lod" and cfg["sigma_end"] <= sigma <= cfg["sigma_start"]
            if active:
                video = comfy.ldm.common_dit.pad_to_patch_size(x[0], self.patch_size)
                grid = (video.shape[2], video.shape[3] // 2, video.shape[4] // 2)
                transformer_options["_lod_grid"] = grid
                transformer_options.setdefault("lod", cfg)
            else:
                transformer_options.pop("_lod_grid", None)
            # sigma only ever decreases within a run, so a rise means a new one
            if sigma >= _run["sigma"]:
                _report(cfg, transformer_options.get("_lod_grid"),
                        context.shape[1], sigma)
            _run["sigma"] = sigma
        return real_forward(self, x, timestep, context,
                            transformer_options=transformer_options, **kwargs)

    def attn(q, k, v, heads, *args, transformer_options=None, **kwargs):
        to = transformer_options or {}
        cfg = to.get("lod") or _global_cfg
        if cfg is None:
            return real_attn(q, k, v, heads, *args,
                             transformer_options=transformer_options, **kwargs)

        if cfg["contiguous_qkv"]:
            # The model hands these over as q.transpose(0, 1).unsqueeze(0) from
            # an (s, heads, dim) buffer (model.py:181), which is not
            # contiguous.  Measured at 1344x768/124f on a W7900: 2145 ms as
            # handed over, 765 ms after copying, 723 ms for a natively
            # contiguous tensor.  Applied in BOTH modes so switching mode
            # changes only the read.
            q, k, v = (AttentionTensorContainer(t.take().contiguous())
                       for t in (q, k, v))

        grid = to.get("_lod_grid")
        if cfg["mode"] != "lod" or grid is None:
            return real_attn(q, k, v, heads, *args,
                             transformer_options=transformer_options, **kwargs)
        video_rows = grid[0] * grid[1] * grid[2]
        seq = q.peek().size(2)
        if seq <= video_rows:
            # the token refiner, which only ever sees the text rows
            return real_attn(q, k, v, heads, *args,
                             transformer_options=transformer_options, **kwargs)

        prefix = seq - video_rows
        tile, page_size = best_tile(grid, cfg["page_size"])
        qt, kt, vt = q.take(), k.take(), v.take()
        order = None
        if cfg["tiled_pages"] and tile is not None:
            order = _order_for(grid, tile, seq, prefix, qt.device)
        else:
            # a grid with prime sides may admit no exact tile at all; keep the
            # raster order and the requested page size rather than paging by
            # single rows, which would make every page a summary
            page_size = cfg["page_size"]
        out = lod_attention(qt, kt, vt, order=order, prefix=prefix,
                            page_size=page_size, top_pages=cfg["top_pages"],
                            select_block=cfg["select_block"],
                            local_radius=cfg["local_radius"],
                            variant=cfg["kernel_variant"])
        # (1, heads, seq, dim) -> the (batch, seq, heads*dim) out_proj expects
        return out.transpose(1, 2).reshape(1, seq, -1)

    h3.optimized_attention = attn
    h3.MiniMaxH3Model._forward = forward
    _installed = True


class MiniMaxH3LoDAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoDAttention",
            display_name="MiniMax H3 Attention Mode (dense / LoD)",
            category="model/patch/minimax",
            search_aliases=["lod", "lod attention", "sparse attention",
                            "minimax attention", "dense attention"],
            # An output node runs even with nothing consuming it, so the node
            # works dropped on the canvas -- ComfyUI would otherwise skip it and
            # the settings would silently do nothing.
            is_output_node=True,
            description=(
                "Switch MiniMax H3's attention between the dense read and the "
                "LoD sparse read. Leave the node in place and change 'mode' to "
                "A/B them: everything outside the read stays identical, which "
                "bypassing the node cannot guarantee."),
            inputs=[
                io.Model.Input("model", optional=True,
                               tooltip="Optional. Route MODEL through this node "
                                       "to scope the setting to that model; "
                                       "leave it unconnected and the setting "
                                       "applies to every MiniMax H3 sample."),
                io.Combo.Input("mode", options=["dense", "lod"], default="lod",
                               optional=True,
                               tooltip="dense = the stock read, unchanged. "
                                       "lod = each query block reads top_pages "
                                       "pages of video rows exactly and the "
                                       "rest as one summary term each, in a "
                                       "single softmax."),
                io.Boolean.Input("contiguous_qkv", default=True, optional=True,
                                 tooltip="Copy q/k/v to contiguous memory "
                                         "before attention. The model builds "
                                         "them as a transpose, which on ROCm "
                                         "costs ~3x (2145 ms vs 765 ms at "
                                         "1344x768/124f). Applies to both "
                                         "modes."),
                io.Int.Input("top_pages", default=128, min=1, max=4096,
                             tooltip="Pages read at full detail per query block. "
                                     "Higher is closer to dense and slower; the "
                                     "LLM port needed 128 to match dense retrieval "
                                     "at 32k context."),
                io.Int.Input("select_block", default=64, min=16, max=256, step=16,
                             tooltip="Queries sharing one page set. 64 is ~1.2x "
                                     "faster than 32 but selects more coarsely."),
                io.Int.Input("page_size", default=64, min=16, max=64,
                             tooltip="Target rows per page. The actual size is "
                                     "rounded to a spatial block that divides the "
                                     "frame (64 -> 8x7 = 56 at 1344x768)."),
                io.Int.Input("local_radius", default=0, min=-1, max=8,
                             optional=True,
                             tooltip="Neighbouring pages forced into every query "
                                     "block's selection alongside its own. -1 "
                                     "disables the forcing entirely."),
                io.Combo.Input("kernel_variant", options=list(VARIANTS),
                               default="default", optional=True,
                               tooltip="Experimental kernels from "
                                       "kernel_exp.py. 'default' is the tuned, "
                                       "shipping one; the others trade "
                                       "accumulator precision or launch count "
                                       "for register pressure. Check the "
                                       "report line for which one ran."),
                io.Boolean.Input("tiled_pages", default=True, optional=True,
                                 tooltip="Reorder video rows so a page is a "
                                         "spatial block instead of a 42-wide, "
                                         "2-tall raster strip."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0,
                               step=0.001,
                               tooltip="Sampling fraction to start reading "
                                       "sparsely. Raise it to keep the early, "
                                       "composition-setting steps dense."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0,
                               step=0.001),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, top_pages=128, select_block=64, page_size=64,
                start_percent=0.0, end_percent=1.0, mode="lod",
                contiguous_qkv=True, local_radius=0, kernel_variant="default",
                tiled_pages=True, model=None) -> io.NodeOutput:
        # Every widget carries its default here as well as in the schema, so a
        # workflow saved before an input existed still loads: ComfyUI validates
        # against the current schema and would otherwise reject the old graph.
        global _global_cfg
        if mode == "lod" and not HAVE_TRITON:
            raise RuntimeError("LoD attention needs triton")
        _install()
        cfg = dict(mode=mode, contiguous_qkv=bool(contiguous_qkv),
                   top_pages=int(top_pages),
                   select_block=int(select_block),
                   page_size=int(page_size),
                   local_radius=int(local_radius),
                   tiled_pages=bool(tiled_pages),
                   kernel_variant=str(kernel_variant),
                   start_percent=float(start_percent),
                   end_percent=float(end_percent),
                   sigma_start=None, sigma_end=None)
        if model is not None:
            # percent -> sigma needs the sampler's shift, which only the model
            # knows; unconnected, the window is resolved on the first forward
            sampling = model.get_model_object("model_sampling")
            cfg["sigma_start"] = float(sampling.percent_to_sigma(start_percent))
            cfg["sigma_end"] = float(sampling.percent_to_sigma(end_percent))
        logging.info("[LoD] node configured: mode=%s top_pages=%d "
                     "select_block=%d page_size=%d local_radius=%d "
                     "tiled_pages=%s contiguous_qkv=%s percent=%.3f..%.3f "
                     "scope=%s", mode, top_pages, select_block, page_size,
                     local_radius, tiled_pages, contiguous_qkv, start_percent,
                     end_percent, "model" if model is not None else "global")
        _run["sigma"] = -1.0          # make the next run report itself
        _global_cfg = cfg
        if model is None:
            return io.NodeOutput(None)
        m = model.clone()
        to = m.model_options["transformer_options"] = dict(
            m.model_options.get("transformer_options", {}))
        to["lod"] = cfg
        return io.NodeOutput(m)


def _ssim(a, b, window=11, sigma=1.5):
    """Mean SSIM over a (N, C, H, W) pair, gaussian window, data range 1."""
    coords = torch.arange(window, dtype=torch.float32, device=a.device) - window // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] @ g[None, :]).expand(a.size(1), 1, window, window)
    pad = window // 2

    def blur(x):
        return torch.nn.functional.conv2d(x, kernel, padding=pad, groups=x.size(1))

    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a * mu_a
    sbb = blur(b * b) - mu_b * mu_b
    sab = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = (((2 * mu_a * mu_b + c1) * (2 * sab + c2))
         / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)))
    return s.mean(dim=(1, 2, 3))


class CompareImageBatches(io.ComfyNode):
    """Objective distance between two renders of the same seed.

    Neither number decides anything on its own -- video quality is judged by
    eye -- but the per-frame series is what a single mean hides: a sparse read
    that drifts will show a rising curve even when the first frame looks fine.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CompareImageBatches",
            display_name="Compare Image Batches (PSNR/SSIM)",
            category="image/compare",
            search_aliases=["psnr", "ssim", "compare", "ab", "diff"],
            description="Per-frame PSNR and SSIM between two image batches, for "
                        "A/B runs at a fixed seed.",
            inputs=[
                io.Image.Input("reference", tooltip="The dense run."),
                io.Image.Input("candidate", tooltip="The LoD run."),
                io.Int.Input("worst_frames", default=5, min=0, max=64,
                             tooltip="How many of the worst frames to list."),
            ],
            outputs=[io.String.Output(display_name="report"),
                     io.Image.Output(display_name="abs_diff")],
        )

    @classmethod
    def execute(cls, reference, candidate, worst_frames) -> io.NodeOutput:
        n = min(reference.shape[0], candidate.shape[0])
        a = reference[:n, ..., :3].detach().float()
        b = candidate[:n, ..., :3].detach().float()
        if a.shape != b.shape:
            raise ValueError("frames differ in size: {} vs {}".format(
                tuple(a.shape[1:]), tuple(b.shape[1:])))
        diff = (a - b).abs().detach()
        mse = diff.pow(2).mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        ssim = _ssim(a.movedim(-1, 1), b.movedim(-1, 1))

        lines = [f"frames {n}  {a.shape[2]}x{a.shape[1]}",
                 f"PSNR  mean {psnr.mean():7.2f} dB   min {psnr.min():7.2f}   "
                 f"max {psnr.max():7.2f}",
                 f"SSIM  mean {ssim.mean():7.4f}      min {ssim.min():7.4f}",
                 f"|diff| mean {diff.mean():.5f}   max {diff.max():.5f}"]
        if n > 1:
            half = n // 2
            lines.append(f"drift  first half PSNR {psnr[:half].mean():6.2f} dB -> "
                         f"second half {psnr[half:].mean():6.2f} dB")
        if worst_frames:
            order = psnr.argsort()[:worst_frames]
            lines.append("worst frames: " + ", ".join(
                f"#{int(i)} {float(psnr[i]):.2f} dB" for i in order))
        report = "\n".join(lines)
        logging.info("CompareImageBatches\n%s", report)
        return io.NodeOutput(report, diff.clamp(0.0, 1.0))


class LoDExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LoDAttention, CompareImageBatches]


async def comfy_entrypoint() -> LoDExtension:
    return LoDExtension()
