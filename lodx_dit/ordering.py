"""Token orderings that make a page a compact region of the video grid.

A page is ``ps`` CONSECUTIVE rows of the packed sequence -- that is what makes
the leaf tier a contiguous gather and what the kernel's ``pid*PS + leaf``
indexing assumes.  So the only way to change what a page *covers* is to change
the order the rows arrive in.

MiniMax H3 emits video rows in raster order: ``_frame_grid`` builds its
meshgrid with ``indexing="ij"`` and flattens, so w runs fastest, and
``_video_grid`` puts t outermost (comfy/ldm/minimax/model.py:77-119).  At
1344x768 one row of the DiT patch grid is ``lw/2 = 42`` tokens, so a 64-token
page is 1.5 raster rows: a strip 42 wide and 2 tall.  Its neighbours above and
below land in different pages, and the page sum that selection scores has to
stand for that shape.

Tile-major ordering fixes it without touching the model: sort by
``(t, h//bh, w//bw, h%bh, w%bw)``.  Where the tile divides the grid a page is
exactly one bh x bw block; where it does not, the edge tiles are shorter and
the run boundaries drift, which is still far more compact than a strip.

Nothing here changes the read.  Permuting q, k and v together and inverting the
permutation on the output is an exact identity for attention -- it is a
relabelling of the key set, and softmax does not care about order.  So the
dense anchor is unaffected and any change in output is entirely down to which
pages selection picks.
"""

from __future__ import annotations

import torch

__all__ = ["video_grid_shape", "tile_order", "sequence_order", "apply_order",
           "invert_order", "page_extent", "best_tile"]


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def best_tile(grid, page_size: int, keep: float = 0.75):
    """Pick (tile, page_size) so a page is exactly one spatial block.

    A tile that does not divide the grid is worse than no tiling at all: the
    short edge tiles make the run boundaries drift, so pages straddle both tile
    rows and frames.  Measured at H3's 24x42 patch grid, a (1,8,8) tile takes
    the mean page bounding box from 192 cells to 234 -- the naive choice loses
    to raster order.

    So the tile has to divide the grid, and the page has to be the tile.  That
    fixes ``page_size`` to ``bh*bw`` rather than the other way round: for 24x42
    the best block at or below 64 is 8x7 = 56, which also divides the 1008 rows
    of a frame, so no page ever straddles a frame either.

    Shape is chosen before size.  Every exact tile has a bounding box equal to
    its own area, so the volume metric cannot tell 8x7 from 3x21 -- but a
    3-tall 21-wide strip is a poor thing for a page sum to represent when the
    attention it stands for is roughly isotropic in space.  So take the
    squarest block among those within ``keep`` of the target and only then
    prefer the larger one.

    Returns ``((bt, bh, bw), page_size)``; ``bt`` stays 1 because a DiT frame is
    a much longer stride than a row and mixing frames into one page buys
    nothing.
    """
    _, h, w = grid
    floor = int(page_size * keep)
    best = None
    for bh in _divisors(h):
        for bw in _divisors(w):
            size = bh * bw
            if size > page_size or size < floor:
                continue
            key = (max(bh, bw) / min(bh, bw), -size)
            if best is None or key < best[0]:
                best = (key, (1, bh, bw), size)
    if best is None:                        # nothing divides near the target
        return (1, 1, 1), 1
    return best[1], best[2]


def video_grid_shape(width: int, height: int, frames: int):
    """(latent_t, rows, cols) of the DiT patch grid, from H3's own arithmetic."""
    latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    return latent_t, height // 16 // 2, width // 16 // 2


def tile_order(shape, tile, device=None) -> torch.Tensor:
    """Permutation of a (T, H, W) grid into tile-major order.

    ``out[i]`` is the raster index of the i-th row in the new order.
    """
    t, h, w = shape
    bt, bh, bw = tile
    nt, nh, nw = -(-t // bt), -(-h // bh), -(-w // bw)
    ti = torch.arange(t, device=device).view(t, 1, 1)
    hi = torch.arange(h, device=device).view(1, h, 1)
    wi = torch.arange(w, device=device).view(1, 1, w)
    tile_id = ((ti // bt) * nh + (hi // bh)) * nw + (wi // bw)
    within = ((ti % bt) * bh + (hi % bh)) * bw + (wi % bw)
    key = (tile_id * (bt * bh * bw) + within).reshape(-1)
    return torch.argsort(key)


def sequence_order(seq_len: int, prefix: int, grid, tile, device=None):
    """Order for the whole packed sequence: prefix untouched, video tiled.

    Rows after the video region (H3 has none; other layouts might) keep their
    place too, so this is safe to apply to any packed sequence whose video
    block starts at ``prefix``.
    """
    t, h, w = grid
    n = t * h * w
    if prefix + n > seq_len:
        raise ValueError("video grid does not fit after the prefix")
    order = torch.arange(seq_len, device=device)
    order[prefix:prefix + n] = prefix + tile_order(grid, tile, device=device)
    return order


def apply_order(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Reorder the sequence axis of a (B, H, S, D) tensor."""
    return tensor.index_select(2, order)


def invert_order(order: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return inv


def page_extent(grid, tile, page_size: int, order: torch.Tensor | None = None):
    """Mean bounding-box volume of the rows in each page.

    A page holds ``page_size`` rows however it is ordered; what changes is how
    spread out they are.  This reports the mean (t, h, w) bounding box of a
    page, which is the thing the page sum has to summarise.  Lower is better,
    and ``page_size`` itself is the floor.
    """
    t, h, w = grid
    n = t * h * w
    if order is None:
        order = torch.arange(n)
    tt = order // (h * w)
    hh = (order // w) % h
    ww = order % w
    pages = n // page_size
    vols = []
    for p in range(pages):
        sl = slice(p * page_size, (p + 1) * page_size)
        vols.append((int(tt[sl].max() - tt[sl].min()) + 1)
                    * (int(hh[sl].max() - hh[sl].min()) + 1)
                    * (int(ww[sl].max() - ww[sl].min()) + 1))
    return sum(vols) / len(vols) if vols else 0.0
