"""Atlas packing via bitmap rasterize-and-place."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

import comfy.model_management

try:
    from numba import njit as _njit
    _HAVE_NUMBA_PACK = True
except ImportError:
    _HAVE_NUMBA_PACK = False
    def _njit(*args, **kwargs):
        def deco(fn): return fn
        return deco if not args else args[0]


@dataclass
class ChartPlacement:
    chart_id: int
    offset: Tuple[float, float]       # in texels
    scale: float                      # texels per UV unit
    rotation: float = 0.0             # radians
    swap_xy: bool = False             # extra 90° bitmap rotation chosen at place time
    chart_h: float = 0.0              # unswapped bitmap height in texels (rotation pivot)


@_njit(cache=True, boundscheck=False)
def _best_rotation_jit(uvs_np: np.ndarray, n_angles: int) -> float:
    V = uvs_np.shape[0]
    best_area = 1e30
    best_theta = 0.0
    if V == 0:
        return 0.0
    half_pi = math.pi * 0.5
    for k in range(n_angles):
        theta = half_pi * k / n_angles
        c = math.cos(theta)
        s = math.sin(theta)
        xmin = 1e30
        xmax = -1e30
        ymin = 1e30
        ymax = -1e30
        for i in range(V):
            ux = uvs_np[i, 0]
            uy = uvs_np[i, 1]
            xr = ux * c - uy * s
            yr = ux * s + uy * c
            if xr < xmin:
                xmin = xr
            if xr > xmax:
                xmax = xr
            if yr < ymin:
                ymin = yr
            if yr > ymax:
                ymax = yr
        area = (xmax - xmin) * (ymax - ymin)
        if area < best_area:
            best_area = area
            best_theta = theta
    return best_theta


def _best_rotation(uvs_np: np.ndarray, n_angles: int = 36) -> float:
    return float(_best_rotation_jit(uvs_np.astype(np.float64), n_angles))


def _rotate_xy(uv: np.ndarray, theta: float) -> np.ndarray:
    if theta == 0.0:
        return uv
    c = math.cos(theta)
    s = math.sin(theta)
    return np.stack([uv[:, 0] * c - uv[:, 1] * s, uv[:, 0] * s + uv[:, 1] * c], axis=1)


@_njit(cache=True, boundscheck=False)
def _rasterize_chart_jit(
    uvs_tex: np.ndarray, faces: np.ndarray, w: int, h: int
) -> np.ndarray:
    """JIT-rasterize triangles into an (h, w) bool bitmap via barycentric test."""
    bm = np.zeros((h, w), dtype=np.bool_)
    F = faces.shape[0]
    eps = 1e-7
    for fi in range(F):
        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        x0 = uvs_tex[i0, 0]
        y0 = uvs_tex[i0, 1]
        x1 = uvs_tex[i1, 0]
        y1 = uvs_tex[i1, 1]
        x2 = uvs_tex[i2, 0]
        y2 = uvs_tex[i2, 1]
        xmin_f = x0
        if x1 < xmin_f:
            xmin_f = x1
        if x2 < xmin_f:
            xmin_f = x2
        xmax_f = x0
        if x1 > xmax_f:
            xmax_f = x1
        if x2 > xmax_f:
            xmax_f = x2
        ymin_f = y0
        if y1 < ymin_f:
            ymin_f = y1
        if y2 < ymin_f:
            ymin_f = y2
        ymax_f = y0
        if y1 > ymax_f:
            ymax_f = y1
        if y2 > ymax_f:
            ymax_f = y2
        xmin = int(math.floor(xmin_f))
        if xmin < 0:
            xmin = 0
        xmax = int(math.ceil(xmax_f))
        if xmax > w - 1:
            xmax = w - 1
        ymin = int(math.floor(ymin_f))
        if ymin < 0:
            ymin = 0
        ymax = int(math.ceil(ymax_f))
        if ymax > h - 1:
            ymax = h - 1
        if xmax < xmin or ymax < ymin:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-20:
            continue
        inv_denom = 1.0 / denom
        for py in range(ymin, ymax + 1):
            yc = py + 0.5
            for px in range(xmin, xmax + 1):
                xc = px + 0.5
                a = ((y1 - y2) * (xc - x2) + (x2 - x1) * (yc - y2)) * inv_denom
                b = ((y2 - y0) * (xc - x2) + (x0 - x2) * (yc - y2)) * inv_denom
                c = 1.0 - a - b
                if a >= -eps and b >= -eps and c >= -eps:
                    bm[py, px] = True
    return bm


def _rasterize_chart(
    uvs_tex: np.ndarray, faces: np.ndarray, w: int, h: int, padding: int
) -> np.ndarray:
    """Rasterize chart triangles into (h, w) bool bitmap, dilated by padding texels."""
    if faces.size == 0:
        return np.zeros((h, w), dtype=bool)
    bm = _rasterize_chart_jit(
        uvs_tex.astype(np.float64), faces.astype(np.int64), int(w), int(h)
    )
    if padding > 0:
        bm = _dilate_bitmap(bm, padding)
    return bm


def _dilate_bitmap(bm: np.ndarray, k: int) -> np.ndarray:
    """k-step Manhattan max-filter dilation."""
    out = bm.copy()
    for _ in range(k):
        next_out = out.copy()
        next_out[1:, :] |= out[:-1, :]
        next_out[:-1, :] |= out[1:, :]
        next_out[:, 1:] |= out[:, :-1]
        next_out[:, :-1] |= out[:, 1:]
        out = next_out
    return out


@_njit(cache=True, boundscheck=False)
def _build_candidates_jit(
    skyline: np.ndarray,
    cur_w: int, cur_h: int,
    bw0: int, bh0: int, bw1: int, bh1: int,
    step: int,
) -> np.ndarray:
    """Build per-chart (x, y, swap_flag) candidate positions (skyline-flush + edge-sweep, both orientations)."""
    nx_skyline = (max(cur_w, 1) // step) + 2
    nx_edge = (max(cur_w, 1) // step) + 2
    ny_edge = (max(cur_h, 1) // step) + 2
    per_orient = nx_skyline + 2 * nx_edge + 2 * ny_edge
    out = np.empty((per_orient * 2, 3), dtype=np.int64)
    k = 0
    for swap_flag in range(2):
        cw = bw0 if swap_flag == 0 else bw1
        x = 0
        while x <= cur_w:
            y = 0
            x_end = x + cw
            if x_end > skyline.shape[0]:
                x_end = skyline.shape[0]
            for xs in range(x, x_end):
                if skyline[xs] > y:
                    y = int(skyline[xs])
            out[k, 0] = x
            out[k, 1] = y
            out[k, 2] = swap_flag
            k += 1
            x += step
        for y_fixed in (0, cur_h):
            x = 0
            while x <= cur_w:
                out[k, 0] = x
                out[k, 1] = y_fixed
                out[k, 2] = swap_flag
                k += 1
                x += step
        for x_fixed in (0, cur_w):
            y = 0
            while y <= cur_h:
                out[k, 0] = x_fixed
                out[k, 1] = y
                out[k, 2] = swap_flag
                k += 1
                y += step
    return out[:k]


@_njit(cache=True, boundscheck=False)
def _update_skyline_jit(skyline: np.ndarray, chart: np.ndarray,
                        x: int, y: int) -> None:
    """Lift skyline[x+i] to y + topmost_True_row + 1 per chart column."""
    ch = chart.shape[0]
    cw = chart.shape[1]
    sw = skyline.shape[0]
    for i in range(cw):
        col_x = x + i
        if col_x >= sw or col_x < 0:
            continue
        col_top = -1
        for j in range(ch - 1, -1, -1):
            if chart[j, i]:
                col_top = j
                break
        if col_top < 0:
            continue
        new_h = y + col_top + 1
        if new_h > skyline[col_x]:
            skyline[col_x] = new_h


@_njit(cache=True, boundscheck=False)
def _best_placement_jit(
    atlas: np.ndarray,
    bitmap: np.ndarray,
    bitmap_rot: np.ndarray,
    candidates: np.ndarray,
    cur_w: int,
    cur_h: int,
):
    """Pick lowest-score non-colliding candidate (score = max(new_w,new_h)^2 + new_w*new_h); out-of-atlas treated as free."""
    n = candidates.shape[0]
    best_x = -1
    best_y = -1
    best_score = -1
    best_swap = 0
    bh0 = bitmap.shape[0]
    bw0 = bitmap.shape[1]
    bh1 = bitmap_rot.shape[0]
    bw1 = bitmap_rot.shape[1]
    ah = atlas.shape[0]
    aw = atlas.shape[1]
    for k in range(n):
        x = candidates[k, 0]
        y = candidates[k, 1]
        swap = candidates[k, 2]
        if swap == 0:
            ch = bh0
            cw = bw0
        else:
            ch = bh1
            cw = bw1
        if x < 0 or y < 0:
            continue
        nw = cur_w if cur_w > x + cw else x + cw
        nh = cur_h if cur_h > y + ch else y + ch
        ext = nw if nw > nh else nh
        score = ext * ext + nw * nh
        if best_score >= 0 and score >= best_score:
            continue
        ok = True
        for j in range(ch):
            yy = y + j
            if yy >= ah:
                continue
            for i in range(cw):
                bit = bitmap[j, i] if swap == 0 else bitmap_rot[j, i]
                if not bit:
                    continue
                xx = x + i
                if xx >= aw:
                    continue
                if atlas[yy, xx]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        best_x = x
        best_y = y
        best_score = score
        best_swap = swap
        if x + cw <= cur_w and y + ch <= cur_h:
            break
    return best_x, best_y, best_score, best_swap


def _blit(atlas: np.ndarray, chart: np.ndarray, x: int, y: int) -> None:
    ah, aw = atlas.shape
    ch, cw = chart.shape
    atlas[y: y + ch, x: x + cw] |= chart


@dataclass
class _PreparedChart:
    chart_id: int
    uvs_tex: np.ndarray              # [V, 2] in texel coords (rotated, scaled, origin 0)
    bitmap: np.ndarray               # [h, w] bool, padded
    bitmap_rot: np.ndarray           # 90° rotated bitmap (for swap_xy placement)
    bbox_w: int
    bbox_h: int
    rotation: float                  # radians, applied to UVs
    s_tex: float                     # texels per UV unit
    perimeter: float                 # for chart ordering


@_njit(cache=True, boundscheck=False)
def _chart_perimeter_jit(uvs: np.ndarray, faces: np.ndarray, V: int) -> float:
    """Sum unique-edge lengths via sorted int64 edge keys."""
    F = faces.shape[0]
    keys = np.empty(F * 3, dtype=np.int64)
    for fi in range(F):
        for j in range(3):
            a = faces[fi, j]
            b = faces[fi, (j + 1) % 3]
            if a < b:
                keys[fi * 3 + j] = a * V + b
            else:
                keys[fi * 3 + j] = b * V + a
    keys = np.sort(keys)
    p = 0.0
    for i in range(keys.shape[0]):
        if i > 0 and keys[i] == keys[i - 1]:
            continue
        a = keys[i] // V
        b = keys[i] % V
        dx = uvs[a, 0] - uvs[b, 0]
        dy = uvs[a, 1] - uvs[b, 1]
        p += math.sqrt(dx * dx + dy * dy)
    return p


def _chart_perimeter(uvs: np.ndarray, faces: np.ndarray) -> float:
    V = int(faces.max()) + 1 if faces.size else 0
    return float(_chart_perimeter_jit(uvs.astype(np.float64), faces.astype(np.int64), V))


# Torch fallback (used when numba is unavailable; runs on GPU if present)

def _dilate_local(x: Tensor, p: int) -> Tensor:
    """4-connectivity dilation by p, applied per-image over a batch of (cnt,g,g) bitmaps.
    Matches the old per-chart _dilate_torch; dilation distributes over union so per-triangle
    dilation OR-scattered equals dilating the assembled chart bitmap."""
    for _ in range(p):
        y = x.clone()
        y[:, 1:, :] |= x[:, :-1, :]
        y[:, :-1, :] |= x[:, 1:, :]
        y[:, :, 1:] |= x[:, :, :-1]
        y[:, :, :-1] |= x[:, :, 1:]
        x = y
    return x


def _raster_all_torch(uvs_tex_pad, faces_pad, fmask, bw_t, bh_t, padding, device):
    """Batched rasterize EVERY chart at once into one flat bool buffer, replacing the per-chart
    loop. Returns (buf, cbase) where buf[cbase[i]:cbase[i+1]].view(bh,bw) is chart i's [y,x] bitmap.
    Triangles are bucketed by next-pow2 bbox size so each batch's local grid stays tiny (bounded
    memory) while collapsing ~N chart rasters into a handful of kernels."""
    n = uvs_tex_pad.shape[0]
    fmax = faces_pad.shape[1]
    bwL, bhL = bw_t.long(), bh_t.long()
    cbase = torch.zeros(n + 1, dtype=torch.long, device=device)
    torch.cumsum(bwL * bhL, 0, out=cbase[1:])
    buf = torch.zeros(int(cbase[-1].item()), dtype=torch.bool, device=device)

    # gather all triangle coords, keep only valid faces -> (Ttot,3,2) + chart id per triangle
    fp = faces_pad.reshape(n, fmax * 3)
    tri = torch.gather(uvs_tex_pad, 1, fp[..., None].expand(-1, -1, 2)).reshape(n * fmax, 3, 2)
    fm = fmask.reshape(-1)
    tri_f = tri[fm]
    if tri_f.shape[0] == 0:
        return buf, cbase
    cid = torch.arange(n, device=device).repeat_interleave(fmax)[fm]

    # per-triangle pixel bbox, inflated by padding (origin >= 0); bucket by next-pow2 max-dim
    tmin = tri_f.amin(1)
    tmax = tri_f.amax(1)
    x0 = (tmin[:, 0].floor().long() - padding).clamp_min(0)
    y0 = (tmin[:, 1].floor().long() - padding).clamp_min(0)
    bbw = (tmax[:, 0].ceil().long() + padding) - x0 + 1
    bbh = (tmax[:, 1].ceil().long() + padding) - y0 + 1
    mxd = torch.maximum(bbw, bbh).clamp_min(1)
    bsz = (2 ** torch.ceil(torch.log2(mxd.float())).long()).long()

    a = tri_f[:, 0]
    b = tri_f[:, 1]
    c = tri_f[:, 2]
    v0 = b - a
    v1 = c - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    den = (d00 * d11 - d01 * d01).clamp(min=1e-20)

    for g in sorted(set(bsz.tolist())):                                  # one batch per pow2 grid
        sel = (bsz == g).nonzero(as_tuple=True)[0]
        m = sel.shape[0]
        xs0 = x0[sel].view(m, 1, 1)
        ys0 = y0[sel].view(m, 1, 1)
        cc = cid[sel]
        bwp = bwL[cc].view(m, 1, 1)
        bhp = bhL[cc].view(m, 1, 1)
        gi = torch.arange(g, device=device)
        px = xs0 + gi.view(1, 1, g)
        py = ys0 + gi.view(1, g, 1)                                        # (m,g,g) int
        pxf = px.float() + 0.5
        pyf = py.float() + 0.5
        v2x = pxf - a[sel, 0].view(m, 1, 1)
        v2y = pyf - a[sel, 1].view(m, 1, 1)
        d20 = v2x * v0[sel, 0].view(m, 1, 1) + v2y * v0[sel, 1].view(m, 1, 1)
        d21 = v2x * v1[sel, 0].view(m, 1, 1) + v2y * v1[sel, 1].view(m, 1, 1)
        idn = den[sel].view(m, 1, 1).reciprocal()
        vv = torch.addcmul(d11[sel].view(m, 1, 1) * d20, d01[sel].view(m, 1, 1), d21, value=-1) * idn
        ww = torch.addcmul(d00[sel].view(m, 1, 1) * d21, d01[sel].view(m, 1, 1), d20, value=-1) * idn
        uu = 1.0 - vv - ww
        inside = (uu >= -1e-6) & (vv >= -1e-6) & (ww >= -1e-6)
        if padding > 0:
            inside = _dilate_local(inside, padding)
        valid = inside & (px < bwp) & (py < bhp)
        flat = (cbase[cc].view(m, 1, 1) + py * bwp + px)[valid]
        buf[flat] = True
    return buf, cbase


def _build_candidates_gpu(sky_t, cur_w, cur_h, bw0, bw1, step, rand_n, gen, device):
    """Skyline-flush + edge-sweep + random candidate (x,y) positions per orientation, built on the
    GPU. Returns (cand0, cand1). Random samples find tight pockets the deterministic grid misses."""
    xs = torch.arange(0, max(cur_w, 1) + 1, step, device=device)
    ys = torch.arange(0, max(cur_h, 1) + 1, step, device=device)
    # edge-sweep candidates are orientation-independent: build once, shared by both orientations
    common = [torch.stack([xs, torch.full_like(xs, yf)], 1) for yf in (0, cur_h)]
    common += [torch.stack([torch.full_like(ys, xf), ys], 1) for xf in (0, cur_w)]
    common = torch.cat(common, 0)
    out = []
    for cw in (bw0, bw1):                                               # skyline-flush + random differ
        if cw > 0 and sky_t.shape[0] >= cw:
            wmax = sky_t.unfold(0, cw, 1).amax(1)[xs.clamp(max=max(sky_t.shape[0] - cw, 0))]
        else:
            wmax = torch.zeros_like(xs)
        parts = [torch.stack([xs, wmax], 1), common]
        if rand_n > 0:                                                  # distinct draws keep density
            rx = torch.randint(0, max(cur_w, 1) + 1, (rand_n,), generator=gen, device=device)
            ry = torch.randint(0, max(cur_h, 1) + 1, (rand_n,), generator=gen, device=device)
            parts.append(torch.stack([rx, ry], 1))
        out.append(torch.cat(parts, 0))
    return out[0], out[1]


def _col_top(b: Tensor) -> Tensor:
    """Topmost True row index per column of a bool bitmap (h,w); -1 for empty columns."""
    h = b.shape[0]
    rows = torch.arange(h, device=b.device)[:, None]
    return torch.where(b, rows, torch.full_like(rows.expand_as(b), -1)).amax(0)


def _best_placement_torch(atlas, pix0, dim0, pix1, dim1, cand0, cand1, cur_w, cur_h, device):
    """Lowest-score non-colliding candidate as a (3,) int tensor [x, y, swap] (x=-1 if none).
    Collision tests only each bitmap's True-pixel offsets (pix), not the full window. Fully on-GPU;
    the caller does the single sync (.tolist())."""
    INF = 1 << 60

    def best(cand, pix, dim):                                          # -> (score, x, y) 0-d tensors
        ch, cw = dim
        cx, cy = cand[:, 0], cand[:, 1]
        coll = atlas[cy[:, None] + pix[:, 0][None, :],                 # (M,k) True-pixel gather
                     cx[:, None] + pix[:, 1][None, :]].any(dim=1)
        nw = torch.clamp(cx + cw, min=cur_w)
        nh = torch.clamp(cy + ch, min=cur_h)
        ext = torch.maximum(nw, nh)
        score = torch.where(coll, torch.full_like(nw, INF), ext * ext + nw * nh)
        j = score.argmin()
        return score[j], cx[j], cy[j]

    s0, x0, y0 = best(cand0, pix0, dim0)
    s1, x1, y1 = best(cand1, pix1, dim1)
    take0 = s0 <= s1
    bsc = torch.where(take0, s0, s1)
    pick = torch.stack([torch.where(take0, x0, x1), torch.where(take0, y0, y1),
                        torch.where(take0, x0.new_zeros(()), x0.new_ones(()))])
    return torch.where(bsc < INF, pick, torch.tensor([-1, -1, 0], device=device))


def _pack_bitmap_torch(chart_uvs, chart_3d_areas, chart_uv_areas, chart_faces,
                       texels_per_unit, padding_texels):
    """Torch rasterize-and-place packer (numba-free fallback). Returns (placements, atlas_w, atlas_h)."""
    n = len(chart_uvs)
    if n == 0:
        return [], 1, 1
    device = comfy.model_management.get_torch_device()
    ang = torch.linspace(0.0, math.pi / 2.0, 37, device=device)[:-1]
    cos_a, sin_a = ang.cos(), ang.sin()

    # ---- Prepare pass 1: best-rotation + scale + bbox for ALL charts at once (batched) ----
    vcount = [int(u.shape[0]) for u in chart_uvs]
    fcount = [int(f.shape[0]) for f in chart_faces]
    vmax = max(vcount)
    fmax = max(fcount)
    uvs_pad = torch.zeros(n, vmax, 2, device=device)
    vmask = torch.zeros(n, vmax, dtype=torch.bool, device=device)
    faces_pad = torch.zeros(n, fmax, 3, dtype=torch.long, device=device)
    fmask = torch.zeros(n, fmax, dtype=torch.bool, device=device)
    for i in range(n):
        uvs_pad[i, :vcount[i]] = chart_uvs[i].to(device=device, dtype=torch.float32)
        vmask[i, :vcount[i]] = True
        if fcount[i]:
            faces_pad[i, :fcount[i]] = chart_faces[i].to(device=device, dtype=torch.long)
            fmask[i, :fcount[i]] = True
    u0, u1 = uvs_pad[..., 0], uvs_pad[..., 1]                            # (N,Vmax)
    BIG = 1e30
    mlo = torch.where(vmask, torch.zeros_like(u0), u0.new_full((), BIG))
    mhi = torch.where(vmask, torch.zeros_like(u0), u0.new_full((), -BIG))
    xr = torch.addcmul(u0[:, :, None] * cos_a, u1[:, :, None], sin_a, value=-1)   # (N,Vmax,A)
    yr = torch.addcmul(u0[:, :, None] * sin_a, u1[:, :, None], cos_a)
    xsp = (xr + mhi[:, :, None]).amax(1) - (xr + mlo[:, :, None]).amin(1)         # (N,A) masked span
    ysp = (yr + mhi[:, :, None]).amax(1) - (yr + mlo[:, :, None]).amin(1)
    ti = (xsp * ysp).argmin(1)                                          # (N,) best angle per chart
    cc, ss = cos_a[ti][:, None], sin_a[ti][:, None]                     # (N,1)
    rx = torch.addcmul(u0 * cc, u1, ss, value=-1)                      # (N,Vmax)
    ry = torch.addcmul(u0 * ss, u1, cc)
    rxmin = (rx + mlo).amin(1)                                          # (N,)
    rxmax = (rx + mhi).amax(1)
    rymin = (ry + mlo).amin(1)
    rymax = (ry + mhi).amax(1)
    a3 = torch.tensor([max(a, 1e-12) for a in chart_3d_areas], device=device)
    au = torch.tensor([max(a, 1e-12) for a in chart_uv_areas], device=device)
    base = (a3 / au).sqrt() * texels_per_unit
    maxb = (4.0 * a3.sqrt() * texels_per_unit).clamp_min(8.0)
    bbm = torch.maximum(rxmax - rxmin, rymax - rymin).clamp_min(1e-12)
    scale = torch.minimum(base, maxb / bbm)                            # (N,)
    uvs_tex_pad = torch.stack([(rx - rxmin[:, None]) * scale[:, None],
                               (ry - rymin[:, None]) * scale[:, None]], dim=-1)   # (N,Vmax,2)
    bw_t = ((rxmax - rxmin) * scale).ceil().int() + padding_texels + 1
    bh_t = ((rymax - rymin) * scale).ceil().int() + padding_texels + 1

    # one sync: pull all per-chart scalars
    thetas = ang[ti].cpu().tolist()
    scales = scale.cpu().tolist()
    bws = bw_t.cpu().tolist()
    bhs = bh_t.cpu().tolist()

    # ---- Prepare pass 2: rasterize ALL charts at once, then trim each bitmap to its bounds ----
    buf, cbase = _raster_all_torch(uvs_tex_pad, faces_pad, fmask, bw_t, bh_t, padding_texels, device)
    cb = cbase.cpu().tolist()
    raw, bnd = [], []
    for i in range(n):
        bm = buf[cb[i]:cb[i + 1]].view(bhs[i], bws[i])
        raw.append(bm)
        rr = torch.arange(bm.shape[0], device=device)
        cc = torch.arange(bm.shape[1], device=device)
        rmax = torch.where(bm.any(1), rr, rr.new_full((), -1)).amax()    # last occupied row / col (-1 if empty)
        cmax = torch.where(bm.any(0), cc, cc.new_full((), -1)).amax()
        bnd.append(torch.stack([rmax, cmax]))
    bnd_cpu = torch.stack(bnd).cpu().tolist()                            # one sync for all trim bounds

    # per-chart True-pixel offsets (sparse collision/blit), dims, col-tops (all kept on GPU)
    pix_l, pixr_l, dim_l, dimr_l, bm_h = [], [], [], [], []
    col_tops, col_tops_rot = [], []
    for i in range(n):
        rm, cm = bnd_cpu[i]
        bm = (raw[i][:rm + 1, :cm + 1].contiguous() if rm >= 0 and cm >= 0
              else torch.zeros((1, 1), dtype=torch.bool, device=device))
        bm_rot = torch.flip(bm.t(), dims=[1]).contiguous()
        pix_l.append(bm.nonzero())
        pixr_l.append(bm_rot.nonzero())
        dim_l.append((int(bm.shape[0]), int(bm.shape[1])))
        dimr_l.append((int(bm_rot.shape[0]), int(bm_rot.shape[1])))
        col_tops.append(_col_top(bm))
        col_tops_rot.append(_col_top(bm_rot))
        bm_h.append(int(bm.shape[0]))
    wmax = max(d[1] for d in dim_l + dimr_l)
    ct_pad = torch.full((n, wmax), -1, dtype=torch.long, device=device)
    ctr_pad = torch.full((n, wmax), -1, dtype=torch.long, device=device)
    for i in range(n):
        ct_pad[i, :col_tops[i].shape[0]] = col_tops[i]
        ctr_pad[i, :col_tops_rot[i].shape[0]] = col_tops_rot[i]
    del raw

    # ---- Placement: skyline bin-pack on GPU (1 sync/chart for the chosen position) ----
    order = sorted(range(n), key=lambda i: -(dim_l[i][0] * dim_l[i][1]))   # biggest bitmap first
    max_b = max(max(d) for d in dim_l)
    margin = max_b + 8
    side_guess = int(math.sqrt(sum(d[0] * d[1] for d in dim_l)) * 2) + 16
    cap = side_guess + margin
    atlas = torch.zeros((cap, cap), dtype=torch.bool, device=device)
    sky_t = torch.zeros(cap, dtype=torch.long, device=device)
    cur_w = cur_h = 0
    placements = [None] * n
    gen = torch.Generator(device=device).manual_seed(0)
    rand_n = 512                                                        # random samples per orientation

    for ci in order:
        if cur_h + margin > atlas.shape[0] or cur_w + margin > atlas.shape[1]:
            ns = max(atlas.shape[0], cur_h + margin, cur_w + margin)
            na = torch.zeros((ns, ns), dtype=torch.bool, device=device)
            na[:atlas.shape[0], :atlas.shape[1]] = atlas
            atlas = na
            nsk = torch.zeros(ns, dtype=torch.long, device=device)
            nsk[:sky_t.shape[0]] = sky_t
            sky_t = nsk
        dim, dimr = dim_l[ci], dimr_l[ci]
        step = max(1, min(dim[0], dim[1]) // 8)
        cand0, cand1 = _build_candidates_gpu(sky_t, cur_w, cur_h, dim[1], dimr[1], step, rand_n, gen, device)
        res = _best_placement_torch(atlas, pix_l[ci], dim, pixr_l[ci], dimr, cand0, cand1, cur_w, cur_h, device)
        bx, by, swap = (int(v) for v in res.tolist())                   # the one sync/chart
        if bx < 0:
            bx, by, swap = cur_w, 0, 0
        pix = pixr_l[ci] if swap else pix_l[ci]
        bh_, bw_ = (dimr if swap else dim)
        atlas[by + pix[:, 0], bx + pix[:, 1]] = True                    # sparse blit
        cur_w = max(cur_w, bx + bw_)
        cur_h = max(cur_h, by + bh_)
        ct = (ctr_pad if swap else ct_pad)[ci, :bw_]                    # GPU skyline lift
        ix = torch.arange(bx, bx + bw_, device=device)
        sky_t[ix] = torch.where(ct >= 0, torch.maximum(sky_t[ix], by + ct + 1), sky_t[ix])
        placements[ci] = ChartPlacement(chart_id=ci, offset=(float(bx), float(by)),
                                        scale=scales[ci], rotation=thetas[ci], swap_xy=bool(swap),
                                        chart_h=float(bm_h[ci]))
    return placements, cur_w, cur_h


def pack_bitmap(
    chart_uvs: List[Tensor],
    chart_3d_areas: List[float],
    chart_uv_areas: List[float],
    chart_faces: List[Tensor],
    texels_per_unit: float = 256.0,
    padding_texels: int = 2,
    attempts: int = 4096,
    rng_seed: int = 0,
) -> Tuple[List[ChartPlacement], int, int]:
    """Rasterize-and-place packer. Returns (placements, atlas_w, atlas_h)."""
    n = len(chart_uvs)
    if n == 0:
        return [], 1, 1
    if not _HAVE_NUMBA_PACK:
        return _pack_bitmap_torch(chart_uvs, chart_3d_areas, chart_uv_areas,
                                  chart_faces, texels_per_unit, padding_texels)

    rng = np.random.default_rng(rng_seed)
    prepared: List[_PreparedChart] = []
    skyline_cap = 4096
    skyline = np.zeros(skyline_cap, dtype=np.int64)

    for i, (uvs_t, area_3d, area_uv, faces_t) in enumerate(
        zip(chart_uvs, chart_3d_areas, chart_uv_areas, chart_faces)
    ):
        uvs = uvs_t.detach().cpu().numpy().astype(np.float64)
        faces = faces_t.detach().cpu().numpy()

        theta = _best_rotation(uvs)
        rotated = _rotate_xy(uvs, theta)
        scale = math.sqrt(max(area_3d, 1e-12) / max(area_uv, 1e-12)) * texels_per_unit
        # Cap per-chart bbox to 4x nominal so a degenerate chart can't span the atlas.
        nominal_side = math.sqrt(max(area_3d, 1e-12)) * float(texels_per_unit)
        max_bbox_texels = max(8.0, 4.0 * nominal_side)
        bbox_uv = (rotated.max(axis=0) - rotated.min(axis=0))
        bbox_uv_max = float(max(bbox_uv[0], bbox_uv[1], 1e-12))
        if scale * bbox_uv_max > max_bbox_texels:
            scale = max_bbox_texels / bbox_uv_max
        uvs_tex = rotated * scale
        uvs_tex = uvs_tex - uvs_tex.min(axis=0)
        bbox_w = int(math.ceil(uvs_tex[:, 0].max())) + padding_texels + 1
        bbox_h = int(math.ceil(uvs_tex[:, 1].max())) + padding_texels + 1

        bm = _rasterize_chart(uvs_tex, faces, bbox_w, bbox_h, padding_texels)
        nz_rows = np.where(bm.any(axis=1))[0]
        nz_cols = np.where(bm.any(axis=0))[0]
        if nz_rows.size == 0 or nz_cols.size == 0:
            bm = np.zeros((1, 1), dtype=bool)
            bbox_h, bbox_w = 1, 1
        else:
            bm = bm[: nz_rows[-1] + 1, : nz_cols[-1] + 1]
            bbox_h, bbox_w = bm.shape
        # True 90 deg rotation; plain transpose would mirror and flip winding.
        bm_rot = bm.T[:, ::-1].copy()

        perim = _chart_perimeter(uvs_tex, faces)
        prepared.append(
            _PreparedChart(
                chart_id=i,
                uvs_tex=uvs_tex,
                bitmap=bm,
                bitmap_rot=bm_rot,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                rotation=theta,
                s_tex=scale,
                perimeter=perim,
            )
        )

    order = sorted(range(n), key=lambda i: -prepared[i].perimeter)

    total_area = sum(p.bbox_w * p.bbox_h for p in prepared)
    side_guess = int(math.sqrt(total_area) * 2) + 16
    atlas = np.zeros((side_guess, side_guess), dtype=bool)
    cur_w = 0
    cur_h = 0

    placements: List[ChartPlacement] = [None] * n  # type: ignore

    for ci in order:
        p = prepared[ci]

        step = max(1, min(p.bbox_w, p.bbox_h) // 8)
        det_arr = _build_candidates_jit(
            skyline, cur_w, cur_h,
            p.bitmap.shape[1], p.bitmap.shape[0],
            p.bitmap_rot.shape[1], p.bitmap_rot.shape[0],
            step,
        )

        x_range = max(cur_w + 1, 1)
        y_range = max(cur_h + 1, 1)
        rand_x = rng.integers(0, x_range, size=attempts).astype(np.int64)
        rand_y = rng.integers(0, y_range, size=attempts).astype(np.int64)
        rand_swap = (np.arange(attempts) & 1).astype(np.int64)
        rand_arr = np.stack([rand_x, rand_y, rand_swap], axis=1)
        candidates = np.concatenate([det_arr, rand_arr], axis=0) if det_arr.size else rand_arr

        best_x, best_y, best_score_int, best_swap_int = _best_placement_jit(
            atlas, p.bitmap, p.bitmap_rot, candidates, cur_w, cur_h,
        )
        best_swap = bool(best_swap_int)

        if best_x >= 0:
            bm_b = p.bitmap_rot if best_swap else p.bitmap
            need_h = max(cur_h, best_y + bm_b.shape[0])
            need_w = max(cur_w, best_x + bm_b.shape[1])
            if atlas.shape[0] < need_h or atlas.shape[1] < need_w:
                target_h = max(atlas.shape[0], need_h, side_guess)
                target_w = max(atlas.shape[1], need_w, side_guess)
                new_atlas = np.zeros((target_h, target_w), dtype=bool)
                new_atlas[: atlas.shape[0], : atlas.shape[1]] = atlas
                atlas = new_atlas

        if best_x < 0:
            # Fallback: place at extension corner.
            best_x, best_y = cur_w, 0
            best_swap = False
            bm = p.bitmap
            need_h = max(cur_h, best_y + bm.shape[0])
            need_w = max(cur_w, best_x + bm.shape[1])
            if atlas.shape[0] < need_h or atlas.shape[1] < need_w:
                target_h = max(atlas.shape[0], need_h)
                target_w = max(atlas.shape[1], need_w)
                new_atlas = np.zeros((target_h, target_w), dtype=bool)
                new_atlas[: atlas.shape[0], : atlas.shape[1]] = atlas
                atlas = new_atlas

        bm = p.bitmap_rot if best_swap else p.bitmap
        _blit(atlas, bm, best_x, best_y)
        cur_w = max(cur_w, best_x + bm.shape[1])
        cur_h = max(cur_h, best_y + bm.shape[0])
        if cur_w + 1 > skyline.shape[0]:
            new_sky = np.zeros(max(skyline.shape[0] * 2, cur_w + 1), dtype=np.int64)
            new_sky[: skyline.shape[0]] = skyline
            skyline = new_sky
        _update_skyline_jit(skyline, bm, best_x, best_y)

        placements[ci] = ChartPlacement(
            chart_id=ci,
            offset=(float(best_x), float(best_y)),
            scale=p.s_tex,
            rotation=p.rotation,
            swap_xy=best_swap,
            chart_h=float(p.bitmap.shape[0]),
        )

    return placements, cur_w, cur_h


def apply_placements(
    chart_uvs: List[Tensor], placements: List[ChartPlacement], atlas_w: int, atlas_h: int
) -> List[Tensor]:
    """Apply per-chart (rotation, scale, swap_xy, offset) and normalize by the larger atlas side (shared scale keeps texel density uniform)."""
    out: List[Tensor] = []
    side = float(max(atlas_w, atlas_h, 1))
    for uvs, p in zip(chart_uvs, placements):
        device = uvs.device
        dtype = uvs.dtype
        uvs_np = uvs.detach().cpu().numpy().astype(np.float64)
        if p.rotation != 0.0:
            uvs_np = _rotate_xy(uvs_np, p.rotation)
        uvs_np = uvs_np - uvs_np.min(axis=0)
        uvs_np = uvs_np * p.scale
        if p.swap_xy:
            # 90 deg rotation matching bm.T[:, ::-1]: (u, v) -> (chart_h - v, u).
            u_old = uvs_np[:, 0].copy()
            uvs_np[:, 0] = p.chart_h - uvs_np[:, 1]
            uvs_np[:, 1] = u_old
        uvs_np[:, 0] += p.offset[0]
        uvs_np[:, 1] += p.offset[1]
        uvs_np /= side
        # Clamp into [0,1]; slivers can stick sub-texel past the tracked extent.
        np.clip(uvs_np, 0.0, 1.0, out=uvs_np)
        out.append(torch.from_numpy(uvs_np).to(device=device, dtype=dtype))
    return out
