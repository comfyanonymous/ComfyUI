

import torch
from typing import Tuple, Callable
import math

def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():
        
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        idx_buffer = torch.zeros(1, hsy, wsx, sy*sx, 1, device=metric.device)

        if no_rand:
            rand_idx = torch.zeros(1, hsy, wsx, 1, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(1, hsy, wsx, 1, 1), device=metric.device)
        
        idx_buffer.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=idx_buffer.dtype))
        idx_buffer = idx_buffer.view(1, hsy, wsx, sy, sx, 1).transpose(2, 3).reshape(1, N, 1)
        rand_idx   = idx_buffer.argsort(dim=1)

        num_dst = int((1 / (sx*sy)) * N)
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = x.gather(dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = x.gather(dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=a_idx.expand(B, a_idx.shape[1], 1).gather(dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=a_idx.expand(B, a_idx.shape[1], 1).gather(dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge


def get_functions(x, ratio, original_shape):
    b, c, original_h, original_w = original_shape
    original_tokens = original_h * original_w
    downsample = int(math.sqrt(original_tokens // x.shape[1]))
    stride_x = 2
    stride_y = 2
    max_downsample = 1

    if downsample <= max_downsample:
        w = original_w // downsample
        h = original_h // downsample
        r = int(x.shape[1] * ratio)
        no_rand = True
        m, u = bipartite_soft_matching_random2d(x, w, h, stride_x, stride_y, r, no_rand)
        return m, u

    nothing = lambda y: y
    return nothing, nothing
