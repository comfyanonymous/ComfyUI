from pathlib import Path, PureWindowsPath


def resolve_safe_path(base_dir: str | Path, user_path: str | Path) -> Path | None:
    """Return the resolved path if user_path stays inside base_dir."""
    raw_path = str(user_path)
    if "\0" in raw_path:
        return None

    if ".." in Path(raw_path).parts or ".." in PureWindowsPath(raw_path).parts:
        return None

    windows_path = PureWindowsPath(raw_path)
    if windows_path.is_absolute() or windows_path.drive:
        return None

    try:
        base = Path(base_dir).resolve(strict=False)
        candidate = (base / raw_path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None

    if not candidate.is_relative_to(base):
        return None

    return candidate
