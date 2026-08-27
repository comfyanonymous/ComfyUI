"""The DB file lock must cover the migration block, not just follow it.

A second process arriving while the lock is held has to bail out *before*
inspecting revisions, copying a backup, or running the upgrade — otherwise two
starts against one database race the backup and the restore.
"""
import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config
from filelock import FileLock

from app.database import db as db_module

_PRE_HEAD = "0006_add_loader_path"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _current_revision(db_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT version_num FROM alembic_version").fetchall()
    assert len(rows) == 1
    return rows[0][0]


@pytest.fixture
def stale_db(tmp_path, monkeypatch):
    """A file-backed DB parked one revision behind head, wired into args."""
    db_path = str(tmp_path / "comfyui.db")
    command.upgrade(_make_config(db_path), _PRE_HEAD)

    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    yield db_path
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


def test_init_file_db_migrates_when_lock_is_free(stale_db):
    """Positive control: unblocked, this same fixture really does migrate."""
    db_module._init_file_db(db_module.args.database_url)

    assert _current_revision(stale_db) != _PRE_HEAD
    assert os.path.exists(stale_db + ".bkp")


def test_held_lock_blocks_before_any_migration_work(stale_db):
    # Given: another process already holds the database's lock file
    holder = FileLock(stale_db + ".lock")
    holder.acquire(timeout=0)
    try:
        # When: a second init runs against the same database
        with pytest.raises(RuntimeError, match="Another ComfyUI process may already be using it"):
            db_module._init_file_db(db_module.args.database_url)

        # Then: it bailed out before backing up or upgrading anything
        assert not os.path.exists(stale_db + ".bkp")
        assert _current_revision(stale_db) == _PRE_HEAD
        assert db_module.Session is None
    finally:
        holder.release()
