from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from comfy import cli_args

PYTHON = sys.executable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli_args(argv: list[str]) -> str:
    script = (
        "import sys; "
        "import comfy.options; comfy.options.enable_args_parsing(); "
        "sys.argv = ['ComfyUI'] + sys.argv[1:]; "
        "from comfy.cli_args import args; "
        "print(args.database_url)"
    )
    result = subprocess.run(
        [PYTHON, "-c", script, *argv],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_default_database_url_is_in_repo_user_directory() -> None:
    url = cli_args.get_default_database_url()
    expected = os.path.abspath(os.path.join(REPO_ROOT, "user", "comfyui.db"))
    assert url == f"sqlite:///{expected}"


def test_default_database_url_respects_base_directory(tmp_path: Path) -> None:
    url = cli_args.get_default_database_url(base_directory=str(tmp_path))
    assert url == f"sqlite:///{tmp_path / 'user' / 'comfyui.db'}"


def test_default_database_url_prefers_user_directory(tmp_path: Path) -> None:
    base = tmp_path / "base"
    user = tmp_path / "custom-user"
    url = cli_args.get_default_database_url(
        base_directory=str(base), user_directory=str(user)
    )
    assert url == f"sqlite:///{user / 'comfyui.db'}"


def test_parsed_base_directory_changes_default_database_url(tmp_path: Path) -> None:
    base = tmp_path / "out-of-tree"
    base.mkdir()
    url = run_cli_args(["--base-directory", str(base)])
    assert url == f"sqlite:///{base / 'user' / 'comfyui.db'}"


def test_parsed_user_directory_changes_default_database_url(tmp_path: Path) -> None:
    base = tmp_path / "out-of-tree"
    user = tmp_path / "custom-user"
    base.mkdir()
    user.mkdir()
    url = run_cli_args(["--base-directory", str(base), "--user-directory", str(user)])
    assert url == f"sqlite:///{user / 'comfyui.db'}"


def test_explicit_database_url_wins(tmp_path: Path) -> None:
    base = tmp_path / "out-of-tree"
    base.mkdir()
    url = run_cli_args(
        ["--base-directory", str(base), "--database-url", "sqlite:///:memory:"]
    )
    assert url == "sqlite:///:memory:"
