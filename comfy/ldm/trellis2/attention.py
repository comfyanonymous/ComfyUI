from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.trellis2.vae import VarLenTensor


def dense_attention(q, k, v, **kwargs):
    """q, k, v: [B, L, H, C]. Permutes for comfy's [B, H, L, C] convention."""
    heads = q.shape[2]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True, **kwargs)
    return out.permute(0, 2, 1, 3)


def _to_rect(t):
    """Fold a VarLenTensor packed as [sum(L_i), H, C] into a dense [B, L, H, C].

    The sparse generation stages run a single object per call (optionally
    CFG-duplicated, which keeps every batch entry the same length), so the
    packed layout is rectangular and attention is ordinary dense attention over
    a batch dim — no variable-length kernel needed. A dense [B, L, H, C] tensor
    (e.g. cross-attention context) passes through unchanged.
    """
    if not isinstance(t, VarLenTensor):
        return t
    B = t.shape[0]
    seqlens = [t.layout[i].stop - t.layout[i].start for i in range(B)]
    if len(set(seqlens)) != 1:
        raise ValueError(
            "trellis2 sparse attention expects equal sequence lengths per batch "
            f"(single object, optionally CFG-duplicated); got {seqlens}. "
            "Multi-object batching is not supported."
        )
    return t.feats.view(B, seqlens[0], *t.feats.shape[1:])


def sparse_attention(q, k, v, **kwargs):
    """Full attention over a SparseTensor's voxels.

    Single object (optionally CFG-duplicated) => the packed layout is
    rectangular, so we fold it into a batch dim and run ordinary dense
    attention. Output type matches q.
    """
    out = dense_attention(_to_rect(q), _to_rect(k), _to_rect(v), **kwargs)  # [B, Lq, H, C]
    if isinstance(q, VarLenTensor):
        return q.replace(out.reshape(-1, *out.shape[2:]))
    return out
