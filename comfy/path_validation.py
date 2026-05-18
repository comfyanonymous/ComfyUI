from pathlib import Path, PureWindowsPath


def resolve_safe_path(base_dir: str | Path, *user_paths: str | Path) -> Path | None:
    """Return the resolved path if user path segments stay inside base_dir."""
    raw_paths = [str(user_path) for user_path in user_paths]
    for raw_path in raw_paths:
        if "\0" in raw_path:
            return None

        if ".." in Path(raw_path).parts or ".." in PureWindowsPath(raw_path).parts:
            return None

        windows_path = PureWindowsPath(raw_path)
        if Path(raw_path).is_absolute() or windows_path.is_absolute() or windows_path.drive:
            return None

    try:
        base = Path(base_dir).resolve(strict=False)
        candidate = base.joinpath(*raw_paths).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None

    if not candidate.is_relative_to(base):
        return None

    return candidate
