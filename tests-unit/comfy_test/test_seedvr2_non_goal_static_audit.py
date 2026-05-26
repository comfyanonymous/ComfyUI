import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_FILES = {
    "comfy/ldm/seedvr/model.py",
    "comfy/ldm/modules/attention.py",
    "comfy/sample.py",
    "comfy/samplers.py",
}

pytestmark = pytest.mark.skipif(
    os.environ.get("SEEDVR2_NON_GOAL_STATIC_AUDIT") != "1",
    reason="SEEDVR2_NON_GOAL_STATIC_AUDIT=1 is required for git-index audit execution.",
)


def _git_changed_paths(*args):
    result = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git diff unavailable: {result.stderr.strip()}")
    return set(result.stdout.splitlines())


def test_seedvr2_non_goal_files_are_not_dirty():
    changed = _git_changed_paths()
    changed.update(_git_changed_paths("--cached"))
    changed_forbidden = sorted(FORBIDDEN_FILES.intersection(changed))
    if changed_forbidden:
        pytest.fail(f"forbidden non-goal files changed: {changed_forbidden}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
