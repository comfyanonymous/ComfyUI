#original code from https://github.com/genmoai/models under apache 2.0 license

# Based on Llama3 Implementation.
import torch


def apply_rotary_emb_qk_real(
    xqk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor without complex numbers.

    Args:
        xqk (torch.Tensor): Query and/or Key tensors to apply rotary embeddings. Shape: (B, S, *, num_heads, D)
                            Can be either just query or just key, or both stacked along some batch or * dim.
        freqs_cos (torch.Tensor): Precomputed cosine frequency tensor.
        freqs_sin (torch.Tensor): Precomputed sine frequency tensor.

    Returns:
        torch.Tensor: The input tensor with rotary embeddings applied.
    """
    # Split the last dimension into even and odd parts
    xqk_even = xqk[..., 0::2]
    xqk_odd = xqk[..., 1::2]

    # Apply rotation
    cos_part = (xqk_even * freqs_cos - xqk_odd * freqs_sin).type_as(xqk)
    sin_part = (xqk_even * freqs_sin + xqk_odd * freqs_cos).type_as(xqk)

    # Interleave the results back into the original shape
    out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2)
    return out
