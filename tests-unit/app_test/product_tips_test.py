import argparse

import pytest

from app import product_tips


def _args(**overrides):
    flags = dict(disable_product_tips=False, dont_print_server=False, windows_standalone_build=False, quick_test_for_ci=False)
    flags.update(overrides)
    return argparse.Namespace(**flags)


@pytest.mark.parametrize("address", ["127.0.0.1:8188", "[::1]:8188", "[2001:db8:85a3::8a2e:370:7334]:8188"])
def test_banner_is_ascii_and_aligned(address):
    lines = product_tips.startup_banner(address).splitlines()
    template = product_tips.COMFY_API_BANNER.splitlines()
    assert len(lines) == len(template)
    for line, original in zip(lines, template):
        assert line.isascii()
        assert len(line) <= 76
        assert line.rfind("|") == original.rfind("|")


def test_banner_shows_address_and_falls_back_to_localhost():
    assert "|  0.0.0.0:8188     |" in product_tips.startup_banner("0.0.0.0:8188")
    assert "|  [::1]:8188       |" in product_tips.startup_banner("[::1]:8188")
    overlong = product_tips.startup_banner("[2001:db8:85a3::8a2e:370:7334]:9000")
    assert "|  localhost:9000   |" in overlong
    assert "2001" not in overlong


def test_banner_default_matches_template():
    assert product_tips.startup_banner("127.0.0.1:8188") == product_tips.COMFY_API_BANNER


def test_tips_enabled_by_default():
    assert product_tips.tips_enabled(_args(), {})
    assert product_tips.tips_enabled(_args(), {"CI": "0", "COMFYUI_DISABLE_PRODUCT_TIPS": "0"})


@pytest.mark.parametrize("flag", ["disable_product_tips", "dont_print_server", "windows_standalone_build", "quick_test_for_ci"])
def test_tips_disabled_by_flag(flag):
    assert not product_tips.tips_enabled(_args(**{flag: True}), {})


@pytest.mark.parametrize("environ", [
    {"COMFYUI_DISABLE_PRODUCT_TIPS": "1"},
    {"COMFYUI_DISABLE_PRODUCT_TIPS": "True"},
    {"COMFYUI_DISABLE_PRODUCT_TIPS": "YES"},
    {"CI": "1"},
    {"CI": "true"},
])
def test_tips_disabled_by_env(environ):
    assert not product_tips.tips_enabled(_args(), environ)


def test_job_hint_once_and_only_for_headless_success(monkeypatch):
    monkeypatch.setattr(product_tips, "_shown_job_hints", set())
    get = product_tips.get_relevant_job_hint
    assert get(_args(disable_product_tips=True), {}, True, {}) is None
    assert get(_args(), {"CI": "1"}, True, {}) is None
    assert get(_args(), {}, False, {}) is None
    assert get(_args(), {}, True, {"client_id": "abc"}) is None
    assert get(_args(), {}, True, {}) == product_tips.COMFY_API_JOB_HINT
    assert get(_args(), {}, True, {}) is None
    assert product_tips.COMFY_API_JOB_HINT.isascii()
