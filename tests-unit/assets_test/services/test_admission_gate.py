import os
from pathlib import Path

from sqlalchemy import select

from app.assets.database.models import AssetContent
from app.assets.scanner import (
    _WATCH_LIST,
    _WatchEntry,
    _should_skip_extension,
    _two_stat_admit,
    tick_watch_list,
)


def test_part_suffix_never_admitted():
    assert _should_skip_extension("model.safetensors.part") is True


def test_size_drift_watch_listed(temp_dir: Path, monkeypatch):
    path = temp_dir / "download.bin"
    path.write_bytes(b"first")
    first_stat = path.stat()

    def changed_stat(_: float) -> None:
        path.write_bytes(b"changed")

    monkeypatch.setattr("app.assets.scanner_admission.time.sleep", changed_stat)
    admitted, watched = _two_stat_admit([(str(path), first_stat)])

    assert admitted == []
    assert watched == [str(path)]


def test_never_stabilizes_dropped_after_cap(session, temp_dir: Path):
    path = temp_dir / "moving.bin"
    path.write_bytes(b"0")
    _WATCH_LIST[:] = [_WatchEntry(str(path), path.stat())]
    for index in range(30):
        path.write_bytes(str(index + 1).encode())
        tick_watch_list(session)

    assert _WATCH_LIST == []
    assert session.scalars(select(AssetContent)).all() == []
