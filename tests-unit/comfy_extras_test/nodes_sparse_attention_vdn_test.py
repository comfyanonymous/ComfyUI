"""Regression: H3 block-sparse producer must not skip a VDN object-patched forward.

The H3 block API accepts an injected attention callable, which is a full
replacement for block.attn. VDN wraps block.attn.forward with its trained
learned-linear complement; bypassing it produces a different model.
"""
from types import SimpleNamespace

from comfy_extras import nodes_sparse_attention as bsa


class _Attention:
    def __init__(self, *, vdn=False):
        def forward(h, **kwargs):
            return ("vdn" if vdn else "native", h)

        if vdn:
            forward._vdn_forward = True
        self.forward = forward

    def __call__(self, h, **kwargs):
        return self.forward(h, **kwargs)


def _run(monkeypatch, *, vdn, eligible, prior_attention=None):
    monkeypatch.setattr(bsa, "h3_eligible", lambda *args: eligible)
    chunks = []
    monkeypatch.setattr(
        bsa, "h3_sparse_attention",
        lambda *args: chunks.append(True) or ("core_chunked", args[1]),
    )
    attn = _Attention(vdn=vdn)
    patch = SimpleNamespace(log_once=lambda *args: None)
    wrapper = bsa.make_h3_block_patch(SimpleNamespace(attn=attn), 7, patch)

    def original_block(args):
        attention = args.get("attention") or attn
        return attention(args["img"], rope_freqs=args["rope_freqs"],
                         transformer_options=args["transformer_options"])

    args = {"img": "input", "rope_freqs": object(), "transformer_options": {}}
    if prior_attention is not None:
        args["attention"] = prior_attention
    result = wrapper(args, {"original_block": original_block})
    return result, chunks


def test_core_bsa_vdn_forward_wins_over_eligible_chunked_producer(monkeypatch):
    result, chunked = _run(monkeypatch, vdn=True, eligible=True)
    assert result == ("vdn", "input")
    assert chunked == []


def test_core_bsa_vdn_forward_wins_over_prior_block_override(monkeypatch):
    result, chunked = _run(
        monkeypatch, vdn=True, eligible=True,
        prior_attention=lambda *args, **kwargs: ("bypassed_vdn", None),
    )
    assert result == ("vdn", "input")
    assert chunked == []


def test_core_bsa_unowned_forward_keeps_existing_chunked_path(monkeypatch):
    result, chunked = _run(monkeypatch, vdn=False, eligible=True)
    assert result == ("core_chunked", "input")
    assert chunked == [True]


def test_core_bsa_unowned_ineligible_forward_keeps_previous_owner(monkeypatch):
    prior = lambda *args, **kwargs: ("prior_owner", args[0])
    result, chunked = _run(monkeypatch, vdn=False, eligible=False, prior_attention=prior)
    assert result == ("prior_owner", "input")
    assert chunked == []
