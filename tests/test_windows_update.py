import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace


UPDATE_SCRIPT = Path(__file__).parents[1] / ".ci" / "update_windows" / "update.py"


class FakePygit2:
    class GitError(Exception):
        pass


def load_update_function(name):
    """Load one helper without running the standalone updater script."""
    tree = ast.parse(UPDATE_SCRIPT.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"pygit2": FakePygit2, "subprocess": subprocess}
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), str(UPDATE_SCRIPT), "exec"), namespace
    )
    return namespace[name]


def test_ssh_fetch_falls_back_to_system_git():
    fetch_remote = load_update_function("_fetch_remote")
    commands = []

    def fake_run(command, check):
        commands.append((command, check))
        return SimpleNamespace(returncode=0)

    namespace = {"pygit2": FakePygit2, "subprocess": SimpleNamespace(run=fake_run)}
    function = type(fetch_remote)(fetch_remote.__code__, namespace)
    repo = SimpleNamespace(workdir="E:/ComfyUI")
    remote = SimpleNamespace(
        name="origin",
        url="git@github.com:Comfy-Org/ComfyUI.git",
        fetch=lambda: (_ for _ in ()).throw(FakePygit2.GitError("unsupported URL protocol")),
    )

    function(repo, remote, "master")

    assert commands == [
        (
            [
                "git",
                "-C",
                "E:/ComfyUI",
                "fetch",
                "--",
                "origin",
                "+refs/heads/master:refs/remotes/origin/master",
            ],
            True,
        )
    ]


def test_other_pygit2_fetch_errors_are_not_swallowed():
    fetch_remote = load_update_function("_fetch_remote")
    namespace = {"pygit2": FakePygit2, "subprocess": SimpleNamespace(run=lambda *_: None)}
    function = type(fetch_remote)(fetch_remote.__code__, namespace)
    remote = SimpleNamespace(
        name="origin",
        url="git@github.com:Comfy-Org/ComfyUI.git",
        fetch=lambda: (_ for _ in ()).throw(FakePygit2.GitError("authentication failed")),
    )

    try:
        function(SimpleNamespace(workdir="E:/ComfyUI"), remote, "master")
    except FakePygit2.GitError as error:
        assert str(error) == "authentication failed"
    else:
        raise AssertionError("expected the original pygit2 error")
