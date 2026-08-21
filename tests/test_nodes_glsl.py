import sys

import pytest

import comfy.options


def _import_nodes_glsl_on_cpu():
    original_argv = sys.argv[:]
    comfy.options.enable_args_parsing(True)
    sys.argv = [sys.argv[0], "--cpu"]
    try:
        return pytest.importorskip("comfy_extras.nodes_glsl")
    finally:
        sys.argv = original_argv
        comfy.options.enable_args_parsing(False)


nodes_glsl = _import_nodes_glsl_on_cpu()


def test_preload_angle_keeps_libraries_local(monkeypatch):
    loaded = []

    def load_library(name, mode=0):
        loaded.append((name, mode))
        return object()

    monkeypatch.setattr(nodes_glsl.ctypes, "RTLD_GLOBAL", 0x100)
    monkeypatch.setattr(nodes_glsl.ctypes, "RTLD_LOCAL", 0)
    monkeypatch.setattr(nodes_glsl.ctypes, "CDLL", load_library)
    monkeypatch.setattr(nodes_glsl.sys, "platform", "linux")
    monkeypatch.setattr(nodes_glsl.comfy_angle, "get_egl_path", lambda: "libEGL.so")
    monkeypatch.setattr(
        nodes_glsl.comfy_angle, "get_glesv2_path", lambda: "libGLESv2.so"
    )

    nodes_glsl._preload_angle()

    modes = [mode for _, mode in loaded]
    assert modes == [0, 0]
