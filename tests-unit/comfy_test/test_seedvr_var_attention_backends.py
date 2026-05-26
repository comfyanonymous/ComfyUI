import subprocess
import sys
import textwrap
import ast
import inspect

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.modules.attention as attention  # noqa: E402


_VAR_BACKENDS = (
    "var_attention_sage",
    "var_attention_sage3",
    "var_attention_flash",
    "var_attention_flash3",
    "var_attention_sub_quad",
    "var_attention_split",
)


def _inputs():
    heads = 2
    head_dim = 4
    total = 6
    q = torch.randn(total, heads, head_dim)
    k = torch.randn(total, heads, head_dim)
    v = torch.randn(total, heads, head_dim)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    return q, k, v, heads, cu


def _has_dynamo_disable(decorator):
    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "disable"
        and isinstance(decorator.value, ast.Attribute)
        and decorator.value.attr == "_dynamo"
        and isinstance(decorator.value.value, ast.Name)
        and decorator.value.value.id == "torch"
    )


def test_var_attention_backend_functions_are_dynamo_disabled_and_signature_compatible():
    tree = ast.parse(inspect.getsource(attention))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    for name in _VAR_BACKENDS:
        node = functions[name]
        positional = [arg.arg for arg in node.args.args[:6]]
        keyword_only = {arg.arg for arg in node.args.kwonlyargs}
        assert positional == ["q", "k", "v", "heads", "cu_seqlens_q", "cu_seqlens_k"]
        assert node.args.vararg is not None
        assert node.args.kwarg is not None
        assert "skip_reshape" in keyword_only
        assert "skip_output_reshape" in keyword_only
        assert any(_has_dynamo_disable(decorator) for decorator in node.decorator_list)


def test_var_attention_registry_contains_always_available_entries():
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_pytorch"] is attention.var_attention_pytorch
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_sub_quad"] is attention.var_attention_sub_quad
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_split"] is attention.var_attention_split


def _run_attention_import(flag, fake_modules=True, fake_module_code=None):
    argv = ["pytest-subprocess", "--cpu", "--disable-xformers"]
    if flag:
        argv.append(flag)
    if fake_module_code is None:
        fake_module_code = ""
    if fake_modules and not fake_module_code:
        fake_module_code = """
import types

sageattention = types.ModuleType("sageattention")
sageattention.sageattn = lambda *a, **k: a[0]
sageattention.sageattn_varlen = lambda *a, **k: a[0]
sys.modules["sageattention"] = sageattention

sageattn3 = types.ModuleType("sageattn3")
sageattn3.sageattn3_blackwell = lambda *a, **k: a[0]
sys.modules["sageattn3"] = sageattn3

flash_attn = types.ModuleType("flash_attn")
flash_attn.flash_attn_func = lambda q, k, v, **kwargs: q
flash_attn.flash_attn_varlen_func = lambda **kwargs: kwargs["q"]
sys.modules["flash_attn"] = flash_attn

flash_attn_interface = types.ModuleType("flash_attn_interface")
flash_attn_interface.flash_attn_varlen_func = lambda **kwargs: (kwargs["q"], None)
sys.modules["flash_attn_interface"] = flash_attn_interface
"""
    code = (
        "import sys\n"
        "import comfy.options\n"
        "comfy.options.enable_args_parsing()\n"
        f"sys.argv = {argv!r}\n"
        f"{textwrap.dedent(fake_module_code)}\n"
        "import comfy.ldm.modules.attention as attention\n"
        "print(attention.optimized_var_attention.__name__)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_var_attention_rebind_sage_launch_flag():
    result = _run_attention_import("--use-sage-attention")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "var_attention_sage"


def test_var_attention_rebind_flash_launch_flag_uses_pytorch_varlen_in_cpu_mode():
    result = _run_attention_import("--use-flash-attention")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "var_attention_pytorch"


def test_var_attention_rebind_sage_launch_flag_without_varlen_uses_pytorch():
    result = _run_attention_import(
        "--use-sage-attention",
        fake_module_code="""
import types

sageattention = types.ModuleType("sageattention")
sageattention.sageattn = lambda *a, **k: a[0]
sys.modules["sageattention"] = sageattention
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "var_attention_pytorch"


def test_var_attention_rebind_flash_launch_flag_without_varlen_uses_pytorch():
    result = _run_attention_import(
        "--use-flash-attention",
        fake_module_code="""
import types

flash_attn = types.ModuleType("flash_attn")
flash_attn.flash_attn_func = lambda q, k, v, **kwargs: q
sys.modules["flash_attn"] = flash_attn
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "var_attention_pytorch"


def test_var_attention_rebind_pytorch_launch_flag():
    result = _run_attention_import("--use-pytorch-cross-attention")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "var_attention_pytorch"


def test_var_attention_rebind_split_launch_flag():
    result = _run_attention_import("--use-split-cross-attention")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "var_attention_split"


def test_var_attention_rebind_default_launch_flags():
    result = _run_attention_import("")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "var_attention_sub_quad"


def test_var_attention_sage_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def fake_sageattn_varlen(q, k, v, cu_q, cu_k, max_q, max_k, is_causal, sm_scale):
        captured.update(cu_q=cu_q, cu_k=cu_k, max_q=max_q, max_k=max_k, is_causal=is_causal)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "SAGE_ATTENTION_VARLEN_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "sageattn_varlen", fake_sageattn_varlen, raising=False)

    out = attention.var_attention_sage(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["max_q"] == 3
    assert captured["max_k"] == 3
    assert captured["is_causal"] is False


def test_var_attention_sage_runtime_error_preserves_fallback_dtype(monkeypatch):
    q, k, v, heads, cu = _inputs()
    q = q.float()
    k = k.half()
    v = v.half()
    captured = {}

    def failing_sageattn_varlen(*args, **kwargs):
        raise RuntimeError("unsupported")

    def fake_var_attention_pytorch(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, skip_reshape=False, skip_output_reshape=False):
        captured.update(dtype=q.dtype, k_dtype=k.dtype, v_dtype=v.dtype, skip_reshape=skip_reshape)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "SAGE_ATTENTION_VARLEN_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "sageattn_varlen", failing_sageattn_varlen, raising=False)
    monkeypatch.setattr(attention, "var_attention_pytorch", fake_var_attention_pytorch)

    out = attention.var_attention_sage(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert out.dtype == torch.float32
    assert captured["dtype"] == torch.float32
    assert captured["k_dtype"] == torch.float32
    assert captured["v_dtype"] == torch.float32
    assert captured["skip_reshape"] is True


def test_var_attention_sage3_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def fake_sageattn3_blackwell(q, k, v, is_causal=False):
        captured.update(shape=tuple(q.shape), is_causal=is_causal)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "SAGE_ATTENTION3_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "sageattn3_blackwell", fake_sageattn3_blackwell, raising=False)

    out = attention.var_attention_sage3(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert captured["shape"] == (2, heads, 3, 4)
    assert captured["is_causal"] is False


def test_var_attention_sage3_runtime_error_falls_back(monkeypatch):
    q, k, v, heads, cu = _inputs()
    q = q.float()
    k = k.half()
    v = v.half()
    captured = {}

    def failing_sageattn3_blackwell(*args, **kwargs):
        raise RuntimeError("unsupported")

    def fake_var_attention_pytorch(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, skip_reshape=False, skip_output_reshape=False):
        captured.update(cu_q=cu_seqlens_q, dtype=q.dtype, k_dtype=k.dtype, v_dtype=v.dtype, skip_reshape=skip_reshape)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "SAGE_ATTENTION_VARLEN_IS_AVAILABLE", False)
    monkeypatch.setattr(attention, "SAGE_ATTENTION3_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "sageattn3_blackwell", failing_sageattn3_blackwell, raising=False)
    monkeypatch.setattr(attention, "var_attention_pytorch", fake_var_attention_pytorch)

    out = attention.var_attention_sage3(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert captured["dtype"] == torch.float32
    assert captured["k_dtype"] == torch.float32
    assert captured["v_dtype"] == torch.float32
    assert captured["skip_reshape"] is True


def test_var_attention_flash_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(attention, "FLASH_ATTENTION_VARLEN_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "flash_attn_varlen_func", fake_flash_attn_varlen_func, raising=False)

    out = attention.var_attention_flash(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_seqlens_q"], cu)
    assert torch.equal(captured["cu_seqlens_k"], cu)
    assert captured["max_seqlen_q"] == 3
    assert captured["max_seqlen_k"] == 3


def test_var_attention_flash_runtime_error_falls_back(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def failing_flash_attn_varlen_func(**kwargs):
        raise NotImplementedError("cpu")

    def fake_var_attention_pytorch(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, skip_reshape=False, skip_output_reshape=False):
        captured.update(cu_q=cu_seqlens_q, skip_reshape=skip_reshape)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "FLASH_ATTENTION_VARLEN_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "flash_attn_varlen_func", failing_flash_attn_varlen_func, raising=False)
    monkeypatch.setattr(attention, "var_attention_pytorch", fake_var_attention_pytorch)

    out = attention.var_attention_flash(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert captured["skip_reshape"] is True


def test_var_attention_flash3_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def fake_flash_attn3_varlen_func(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["q"]), None

    monkeypatch.setattr(attention, "flash_attn3_varlen_func", fake_flash_attn3_varlen_func, raising=False)
    monkeypatch.setattr(attention, "FLASH_ATTENTION3_IS_AVAILABLE", True)

    out = attention.var_attention_flash3(
        q,
        k,
        v,
        heads,
        cu,
        cu,
        skip_reshape=True,
        skip_output_reshape=True,
        dropout_p=0.25,
        window_size=(16, 16),
    )

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_seqlens_q"], cu)
    assert torch.equal(captured["cu_seqlens_k"], cu)
    assert captured["max_seqlen_q"] == 3
    assert captured["max_seqlen_k"] == 3
    assert captured["seqused_q"] is None
    assert captured["seqused_k"] is None
    assert "dropout_p" not in captured
    assert "window_size" not in captured


def test_var_attention_flash3_accepts_tensor_return(monkeypatch):
    q, k, v, heads, cu = _inputs()

    def fake_flash_attn3_varlen_func(**kwargs):
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(attention, "flash_attn3_varlen_func", fake_flash_attn3_varlen_func, raising=False)
    monkeypatch.setattr(attention, "FLASH_ATTENTION3_IS_AVAILABLE", True)

    out = attention.var_attention_flash3(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)


def test_var_attention_flash3_runtime_error_falls_back(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def failing_flash_attn3_varlen_func(**kwargs):
        raise RuntimeError("unsupported")

    def fake_var_attention_pytorch(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, skip_reshape=False, skip_output_reshape=False):
        captured.update(cu_q=cu_seqlens_q, skip_reshape=skip_reshape)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "FLASH_ATTENTION3_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "flash_attn3_varlen_func", failing_flash_attn3_varlen_func, raising=False)
    monkeypatch.setattr(attention, "var_attention_pytorch", fake_var_attention_pytorch)

    out = attention.var_attention_flash3(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert captured["skip_reshape"] is True


def test_var_attention_sub_quad_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def fake_var_attention_pytorch(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, skip_reshape=False, skip_output_reshape=False):
        captured.update(cu_q=cu_seqlens_q, cu_k=cu_seqlens_k, skip_reshape=skip_reshape)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "var_attention_pytorch", fake_var_attention_pytorch)

    out = attention.var_attention_sub_quad(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["skip_reshape"] is True


def test_var_attention_split_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def fake_var_attention_pytorch_split(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, skip_reshape=False, skip_output_reshape=False):
        captured.update(cu_q=cu_seqlens_q, cu_k=cu_seqlens_k, skip_reshape=skip_reshape)
        return torch.zeros_like(q)

    def fail_var_attention_pytorch(*args, **kwargs):
        raise AssertionError("split backend must not use nested-tensor pytorch var attention")

    monkeypatch.setattr(attention, "var_attention_pytorch", fail_var_attention_pytorch)
    monkeypatch.setattr(attention, "var_attention_pytorch_split", fake_var_attention_pytorch_split)

    out = attention.var_attention_split(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["skip_reshape"] is True


def test_var_attention_pytorch_split_normalizes_split_indices_to_cpu(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured_devices = []
    real_tensor_split = torch.tensor_split

    def capture_tensor_split(input, indices_or_sections, dim=0):
        if isinstance(indices_or_sections, torch.Tensor):
            captured_devices.append(indices_or_sections.device.type)
        return real_tensor_split(input, indices_or_sections, dim=dim)

    monkeypatch.setattr(torch, "tensor_split", capture_tensor_split)

    out = attention.var_attention_pytorch_split(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert captured_devices == ["cpu", "cpu", "cpu"]


def test_missing_sage_package_guard_message_preserved():
    code = textwrap.dedent(
        """
        import builtins
        import sys
        import comfy.options

        comfy.options.enable_args_parsing()

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sageattention":
                raise ImportError("No module named sageattention", name="sageattention")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import
        sys.argv = ["pytest-subprocess", "--cpu", "--disable-xformers", "--use-sage-attention"]
        import comfy.ldm.modules.attention
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "To use the `--use-sage-attention` feature" in result.stderr
    assert "sageattention" in result.stderr
