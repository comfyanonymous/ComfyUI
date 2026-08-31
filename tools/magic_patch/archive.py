"""Validate and materialize immutable V2 pack ZIP artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MAX_FILES = 200_000
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 8 * 1024 * 1024 * 1024


class PackArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class PackArchive:
    path: Path
    sha256: str
    pack_folder: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _members(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], str]:
    members = archive.infolist()
    if not members or len(members) > MAX_FILES:
        raise PackArchiveError("pack archive has an invalid file count")
    top_levels: set[str] = set()
    expanded = 0
    required: set[str] = set()
    for member in members:
        if "\\" in member.filename:
            raise PackArchiveError(
                f"pack archive entry has an unsafe path: {member.filename!r}"
            )
        relative = PurePosixPath(member.filename)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise PackArchiveError(
                f"pack archive entry escapes its root: {member.filename!r}"
            )
        mode = member.external_attr >> 16
        kind = mode & 0o170000
        if kind not in (0, 0o040000, 0o100000):
            raise PackArchiveError(
                f"pack archive contains an unsupported file: {member.filename!r}"
            )
        if member.file_size > MAX_FILE_BYTES:
            raise PackArchiveError(
                f"pack archive entry is too large: {member.filename!r}"
            )
        expanded += member.file_size
        if expanded > MAX_EXPANDED_BYTES:
            raise PackArchiveError("pack archive expands beyond the allowed size")
        top_levels.add(relative.parts[0])
        if len(relative.parts) == 3 and relative.parts[1] == "v2":
            required.add(relative.parts[2])
    if len(top_levels) != 1:
        raise PackArchiveError("pack archive must contain one top-level folder")
    if not {"pyproject.toml", "secure-nodes.json"}.issubset(required):
        raise PackArchiveError("pack archive has no complete v2 directory")
    return members, next(iter(top_levels))


def inspect(path: Path | str) -> PackArchive:
    archive_path = Path(path).expanduser()
    if archive_path.is_symlink() or not archive_path.is_file():
        raise PackArchiveError(f"pack archive is not a regular file: {archive_path}")
    archive_path = archive_path.resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _, pack_folder = _members(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackArchiveError(f"invalid pack archive: {archive_path}") from exc
    return PackArchive(
        path=archive_path,
        sha256=_sha256(archive_path),
        pack_folder=pack_folder,
    )


def extract(archive: PackArchive, destination: Path | str) -> Path:
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise PackArchiveError(
            f"pack archive destination already exists: {destination}"
        )
    destination.mkdir(parents=True, mode=0o700)
    try:
        with zipfile.ZipFile(archive.path) as opened:
            members, pack_folder = _members(opened)
            if pack_folder != archive.pack_folder:
                raise PackArchiveError("pack archive identity changed")
            for member in members:
                relative = PurePosixPath(member.filename)
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with opened.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                executable = bool((member.external_attr >> 16) & 0o111)
                os.chmod(target, 0o755 if executable else 0o644)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination / archive.pack_folder
