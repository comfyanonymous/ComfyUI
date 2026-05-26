"""Regression tests for the SeedVR2-named guard inside
``comfy.ldm.modules.attention.var_attention_pytorch``.

Contract:

  * If ``torch.nested.nested_tensor_from_jagged`` is unavailable on the
    installed PyTorch build, ``var_attention_pytorch`` must raise
    ``RuntimeError`` whose message contains both ``SeedVR2`` and
    ``nested_tensor_from_jagged`` so the operator can identify the
    failing attention path. A bare ``AttributeError`` from the
    ``torch.nested`` lookup is non-conformant. The guard must also
    cover the case where the ``torch.nested`` namespace itself is
    absent (e.g. forks/builds that strip the module) — accessing
    ``torch.nested`` directly would otherwise raise the same opaque
    ``AttributeError`` the guard is meant to translate.
  * If the API is present, the present-API path must produce the
    canonical SeedVR2-inference output shape ``(total_tokens,
    heads * head_dim)``.
  * If the caller passes malformed offsets (off-end / non-monotonic /
    size-mismatched), torch's own per-call ``RuntimeError`` propagates
    unchanged: the SeedVR2-context guard fires only on the missing-API
    path, never on torch's per-call shape errors.

Each cell additionally pins the production guard at the AST level via
``inspect.getsource(var_attention_pytorch)`` so every AC fails
diagnostically on an unguarded base.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import ast  # noqa: E402
import inspect  # noqa: E402
import logging  # noqa: E402
import textwrap  # noqa: E402
import warnings  # noqa: E402

import pytest  # noqa: E402

from comfy.ldm.modules.attention import var_attention_pytorch  # noqa: E402


def _inputs():
    """Canonical 2-D ``(q, k, v, heads, cu_seqlens_q, cu_seqlens_k,
    total_tokens, embed_dim)`` matching the live shape from GPT-3:
    two segments of 3 tokens each, ``embed_dim = heads * head_dim =
    2 * 8 = 16``.
    """
    heads, head_dim, total_tokens = 2, 8, 6
    embed_dim = heads * head_dim
    q = torch.randn(total_tokens, embed_dim)
    k = torch.randn(total_tokens, embed_dim)
    v = torch.randn(total_tokens, embed_dim)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    return q, k, v, heads, cu, cu, total_tokens, embed_dim


def _assert_guard_source_pin():
    """Walk the AST of ``var_attention_pytorch`` and assert that the
    first ``raise RuntimeError(...)`` statement appears strictly
    before any attribute access named ``nested_tensor_from_jagged``.

    Substring-based source pinning (``src.index('raise RuntimeError(')
    < src.index('nested_tensor_from_jagged')``) is fragile: it false-
    positives on docstring or comment text containing the literal,
    and false-negatives on a refactor that splits ``raise
    RuntimeError(`` across lines or replaces it with a helper
    raising ``RuntimeError`` from another scope. AST-walking the
    function body collapses both failure modes onto the only
    invariant we actually require — the guard statement dominates
    the attribute access by line number.
    """
    src = textwrap.dedent(inspect.getsource(var_attention_pytorch))
    tree = ast.parse(src)
    raise_lines = []
    nested_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            if isinstance(func, ast.Name) and func.id == "RuntimeError":
                raise_lines.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == "nested_tensor_from_jagged":
            nested_lines.append(node.lineno)
    assert raise_lines, (
        "var_attention_pytorch has no `raise RuntimeError(...)` AST node; "
        f"the SeedVR2-named guard is missing.\n--- source ---\n{src}"
    )
    assert nested_lines, (
        "var_attention_pytorch source has no `nested_tensor_from_jagged` "
        f"attribute access; cannot pin guard ordering.\n"
        f"--- source ---\n{src}"
    )
    first_raise = min(raise_lines)
    first_nested = min(nested_lines)
    assert first_raise < first_nested, (
        f"`raise RuntimeError(...)` first appears at line {first_raise}, "
        f"but `torch.nested.nested_tensor_from_jagged` is referenced first "
        f"at line {first_nested}; the guard must precede the lookup.\n"
        f"--- source ---\n{src}"
    )


def test_missing_api_raises_seedvr2_runtime_error(monkeypatch):
    monkeypatch.delattr(torch.nested, "nested_tensor_from_jagged", raising=False)
    q, k, v, heads, cu_q, cu_k, _, _ = _inputs()

    with pytest.raises(RuntimeError, match=r"SeedVR2.*nested_tensor_from_jagged"):
        var_attention_pytorch(q, k, v, heads, cu_q, cu_k)

    _assert_guard_source_pin()


def test_missing_namespace_raises_seedvr2_runtime_error(monkeypatch):
    monkeypatch.delattr(torch, "nested", raising=False)
    q, k, v, heads, cu_q, cu_k, _, _ = _inputs()

    with pytest.raises(RuntimeError, match=r"SeedVR2.*nested_tensor_from_jagged"):
        var_attention_pytorch(q, k, v, heads, cu_q, cu_k)

    _assert_guard_source_pin()


def test_present_api_returns_expected_shape():
    q, k, v, heads, cu_q, cu_k, total_tokens, embed_dim = _inputs()

    torch_fx_logger = logging.getLogger("torch.fx._symbolic_trace")
    old_torch_fx_level = torch_fx_logger.level
    torch_fx_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            out = var_attention_pytorch(q, k, v, heads, cu_q, cu_k)
    finally:
        torch_fx_logger.setLevel(old_torch_fx_level)

    assert tuple(out.shape) == (total_tokens, embed_dim), (
        f"expected ({total_tokens}, {embed_dim}); got {tuple(out.shape)}"
    )

    _assert_guard_source_pin()


def test_malformed_offsets_propagates_torch_runtime_error():
    q, k, v, heads, _, _, _, _ = _inputs()
    cu_q_bad = torch.tensor([0, 3, 7], dtype=torch.int32)
    cu_k_ok = torch.tensor([0, 3, 6], dtype=torch.int32)

    with pytest.raises(RuntimeError) as exc_info:
        var_attention_pytorch(q, k, v, heads, cu_q_bad, cu_k_ok)

    msg = str(exc_info.value)
    assert "split_with_sizes" in msg, (
        f"expected torch's `split_with_sizes` error to propagate; got: {msg!r}"
    )
    assert "SeedVR2" not in msg, (
        f"SeedVR2-context substring must not be substituted onto torch's "
        f"per-call shape error; got: {msg!r}"
    )

    _assert_guard_source_pin()
