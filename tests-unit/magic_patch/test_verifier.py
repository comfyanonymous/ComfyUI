from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.magic_patch import cli as magicpatch
from tools.magic_patch import verifier


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return path


def test_auto_mode_is_optional_when_no_verifier_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier, "resolve_executable", lambda configured: None)

    result = verifier.verify(
        mode="auto",
        configured=None,
        pack=Path("pack"),
        source=Path("source"),
        core_root=None,
        python_executable=Path("python"),
        timeout_seconds=10,
    )

    assert result.status == "unavailable"
    assert not result.passed


def test_required_mode_fails_before_an_agent_is_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    monkeypatch.setattr(verifier, "resolve_executable", lambda configured: None)
    config = magicpatch.ConversionConfig(
        source=source,
        output=tmp_path / "output",
        provider="codex",
        source_sha="0123456789abcdef",
        sandbox_verification="required",
    )

    with pytest.raises(
        magicpatch.MagicPatchError, match="needs comfy-secure-verify-pack"
    ):
        magicpatch._preflight(config)


def test_external_verifier_receives_a_versioned_request_and_returns_evidence(
    tmp_path: Path,
) -> None:
    request_copy = tmp_path / "observed-request.json"
    executable = _executable(
        tmp_path / "verifier",
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--request', required=True)\n"
        "parser.add_argument('--output', required=True)\n"
        "args = parser.parse_args()\n"
        "request = json.loads(Path(args.request).read_text())\n"
        f"Path({str(request_copy)!r}).write_text(json.dumps(request))\n"
        "Path(args.output).write_text(json.dumps({\n"
        f"    'format': {verifier.RESULT_FORMAT!r},\n"
        "    'verifier': 'test-seatbelt',\n"
        "    'status': 'passed',\n"
        "    'checks': ['guest import', 'network denied'],\n"
        "    'errors': [],\n"
        "}))\n",
    )
    pack = tmp_path / "pack"
    source = tmp_path / "source"
    pack.mkdir()
    source.mkdir()

    result = verifier.verify(
        mode="auto",
        configured=executable,
        pack=pack,
        source=source,
        core_root=None,
        python_executable=Path("/usr/bin/python3"),
        timeout_seconds=10,
    )

    request = json.loads(request_copy.read_text())
    assert request["format"] == verifier.REQUEST_FORMAT
    assert request["pack"] == str(pack.resolve())
    assert request["source"] == str(source.resolve())
    assert result.status == "passed"
    assert result.checks == ("guest import", "network denied")


def test_an_installed_verifier_failure_is_not_silently_ignored(tmp_path: Path) -> None:
    executable = _executable(
        tmp_path / "verifier",
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--request', required=True)\n"
        "parser.add_argument('--output', required=True)\n"
        "args = parser.parse_args()\n"
        "Path(args.output).write_text(json.dumps({\n"
        f"    'format': {verifier.RESULT_FORMAT!r},\n"
        "    'verifier': 'test-seatbelt',\n"
        "    'status': 'failed',\n"
        "    'checks': ['guest import'],\n"
        "    'errors': ['network access unexpectedly succeeded'],\n"
        "}))\n",
    )
    pack = tmp_path / "pack"
    source = tmp_path / "source"
    pack.mkdir()
    source.mkdir()

    result = verifier.verify(
        mode="auto",
        configured=executable,
        pack=pack,
        source=source,
        core_root=None,
        python_executable=Path("/usr/bin/python3"),
        timeout_seconds=10,
    )

    assert result.status == "failed"
    assert result.errors == ("network access unexpectedly succeeded",)
