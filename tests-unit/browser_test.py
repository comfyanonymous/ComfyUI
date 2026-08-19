import os

from comfy import browser
from comfy.cli_args import parser


def test_custom_browser_command_is_launched(monkeypatch):
    """A configured browser and optional Chromium profile receive the URL."""
    monkeypatch.setattr(browser.shutil, "which", lambda command: "/resolved/browser" if command == "custom-browser" else None)
    launched = []

    def fake_popen(args, **kwargs):
        launched.append((args, kwargs))
        return object()

    monkeypatch.setattr(browser.subprocess, "Popen", fake_popen)

    assert browser.open_browser(
        "http://127.0.0.1:8188",
        browser_path="custom-browser",
        browser_profile="Profile 1",
    )
    assert launched == [
        (
            ["/resolved/browser", "--profile-directory", "Profile 1", "http://127.0.0.1:8188"],
            {} if os.name == "nt" else {"start_new_session": True},
        )
    ]


def test_missing_custom_browser_falls_back_to_default(monkeypatch):
    """A bad browser command does not prevent the UI from opening."""
    monkeypatch.setattr(browser.shutil, "which", lambda command: None)
    opened = []
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: opened.append(url) or True)

    assert browser.open_browser("http://localhost:8188", browser_path="missing-browser")
    assert opened == ["http://localhost:8188"]


def test_failed_browser_launch_falls_back_to_default(monkeypatch):
    """A launch failure is reported by falling back to the default browser."""
    monkeypatch.setattr(browser.shutil, "which", lambda command: "/resolved/browser")
    opened = []
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: opened.append(url) or True)

    def fail_popen(args, **kwargs):
        raise OSError("launch failed")

    monkeypatch.setattr(browser.subprocess, "Popen", fail_popen)

    assert browser.open_browser("http://localhost:8188", browser_path="custom-browser")
    assert opened == ["http://localhost:8188"]


def test_default_browser_is_used_without_custom_path(monkeypatch):
    """Omitting a browser path retains the existing behavior."""
    opened = []
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: opened.append(url) or True)

    assert browser.open_browser("http://localhost:8188")
    assert opened == ["http://localhost:8188"]


def test_browser_path_cli_option():
    """Both browser options are exposed through the startup parser."""
    parsed = parser.parse_args([
        "--auto-launch",
        "--browser-path",
        "custom-browser",
        "--browser-profile",
        "Profile 1",
    ])
    assert parsed.auto_launch
    assert parsed.browser_path == "custom-browser"
    assert parsed.browser_profile == "Profile 1"
