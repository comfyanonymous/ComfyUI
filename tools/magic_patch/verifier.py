"""Optional external sandbox verification for Magic Patch artifacts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


REQUEST_FORMAT = "comfy-magic-patch-verifier-request/1"
RESULT_FORMAT = "comfy-magic-patch-verifier-result/1"
DEFAULT_EXECUTABLE = "comfy-secure-verify-pack"
ENVIRONMENT_VARIABLE = "COMFY_MAGIC_PATCH_SANDBOX_VERIFIER"
MODES = frozenset({"auto", "required", "off"})
MAX_RESULT_BYTES = 1_000_000


@dataclass(frozen=True)
class SandboxVerification:
    status: str
    verifier: str | None = None
    checks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "verifier": self.verifier,
            "checks": list(self.checks),
            "errors": list(self.errors),
        }


def resolve_executable(configured: str | Path | None) -> str | None:
    requested = (
        str(configured)
        if configured is not None
        else os.environ.get(ENVIRONMENT_VARIABLE)
    )
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.parent != Path(".") or candidate.is_absolute():
            resolved = candidate.resolve()
            if resolved.is_file() and os.access(resolved, os.X_OK):
                return str(resolved)
            return None
        return shutil.which(requested)
    return shutil.which(DEFAULT_EXECUTABLE)


def availability(mode: str, configured: str | Path | None) -> SandboxVerification:
    if mode not in MODES:
        raise ValueError(f"invalid sandbox verification mode {mode!r}")
    if mode == "off":
        return SandboxVerification(status="skipped")
    executable = resolve_executable(configured)
    if executable is None:
        return SandboxVerification(
            status="unavailable",
            errors=(f"{DEFAULT_EXECUTABLE} is not installed or executable",),
        )
    return SandboxVerification(status="available", verifier=executable)


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"sandbox verifier {field} must be a string list")
    if any(len(item) > 10_000 for item in value):
        raise ValueError(f"sandbox verifier {field} contains an oversized entry")
    return tuple(value)


def _read_result(path: Path, executable: str) -> SandboxVerification:
    if path.is_symlink() or not path.is_file():
        raise ValueError("sandbox verifier did not write a regular result file")
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError("sandbox verifier result exceeds the size limit")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("sandbox verifier result must be an object")
    if value.get("format") != RESULT_FORMAT:
        raise ValueError(f"sandbox verifier result format must be {RESULT_FORMAT!r}")
    status = value.get("status")
    if status not in {"passed", "failed", "unavailable"}:
        raise ValueError(
            "sandbox verifier status must be 'passed', 'failed', or 'unavailable'"
        )
    verifier_name = value.get("verifier")
    if not isinstance(verifier_name, str) or not verifier_name:
        raise ValueError("sandbox verifier name must be a non-empty string")
    checks = _string_list(value.get("checks"), "checks")
    errors = _string_list(value.get("errors"), "errors")
    if status == "passed" and errors:
        raise ValueError("a passing sandbox verifier result cannot contain errors")
    if status in {"failed", "unavailable"} and not errors:
        raise ValueError(
            f"a sandbox verifier result with status {status!r} must contain errors"
        )
    return SandboxVerification(
        status=status,
        verifier=f"{verifier_name} ({executable})",
        checks=checks,
        errors=errors,
    )


def verify(
    *,
    mode: str,
    configured: str | Path | None,
    pack: Path,
    source: Path,
    core_root: Path | None,
    python_executable: Path,
    timeout_seconds: int,
) -> SandboxVerification:
    state = availability(mode, configured)
    if state.status != "available":
        return state
    executable = state.verifier
    assert executable is not None
    with tempfile.TemporaryDirectory(prefix="magic-patch-verifier-") as raw:
        directory = Path(raw)
        request = directory / "request.json"
        result = directory / "result.json"
        request.write_text(
            json.dumps(
                {
                    "format": REQUEST_FORMAT,
                    "pack": str(pack.resolve()),
                    "source": str(source.resolve()),
                    "core_root": str(core_root.resolve()) if core_root else None,
                    "python_executable": str(python_executable.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        try:
            completed = subprocess.run(
                [executable, "--request", str(request), "--output", str(result)],
                cwd=directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return SandboxVerification(
                status="failed",
                verifier=executable,
                errors=(f"sandbox verifier exceeded its {timeout_seconds}s timeout",),
            )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            return SandboxVerification(
                status="failed",
                verifier=executable,
                errors=(
                    f"sandbox verifier exited {completed.returncode}: {detail[-2000:]}",
                ),
            )
        try:
            return _read_result(result, executable)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return SandboxVerification(
                status="failed",
                verifier=executable,
                errors=(f"invalid sandbox verifier result: {error}",),
            )
