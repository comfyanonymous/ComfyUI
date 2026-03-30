"""User-editable rotary implementation for the baymax-zimage node.

Edit apply_rotary_emb and then run the baymax-zimage node with reload enabled.
The z-Image transformer in this repository calls apply_rope internally, so the
node adapts that call into this apply_rotary_emb interface.
"""

def apply_rotary_emb(xq, xk, freqs_cis, original_apply_rope=None):
    if freqs_cis is None:
        if original_apply_rope is None:
            raise RuntimeError("freqs_cis is None and no original_apply_rope fallback is available")
        return original_apply_rope(xq, xk, freqs_cis)

    # Standalone rotary implementation compatible with z-Image NextDiT paths.
    def _apply_single(x):
        if x is None:
            return None

        x_work = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
        fc = freqs_cis

        # Match the half-dim slice used by this q/k tensor.
        if x_work.shape[2] != 1 and fc.shape[2] != 1 and x_work.shape[2] != fc.shape[2]:
            fc = fc[:, :, :x_work.shape[2]]

        x_out = fc[..., 0] * x_work[..., 0]
        x_out.addcmul_(fc[..., 1], x_work[..., 1])
        return x_out.reshape(*x.shape).type_as(x)

    return _apply_single(xq), _apply_single(xk)