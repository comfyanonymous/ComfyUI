import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

_REVISION_0008 = "0008_drop_asset_meta"
_REVISION_0009 = "0009_created_at_from_mtime"
_REVISION_0010 = "0010_add_missing_since"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _columns(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(asset_contents)")}


def _indexes(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA index_list(asset_contents)")}


def _insert_missing_row(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO asset_contents (id, size_bytes, path, is_missing, created_at) "
            "VALUES ('c1', 1, '/models/a.safetensors', 1, '2026-10-01 00:00:00')"
        )


def _assert_missing_since_added(db_path: str) -> None:
    assert "missing_since" in _columns(db_path)
    with sqlite3.connect(db_path) as conn:
        (index_sql,) = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'ix_asset_contents_missing_since'"
        ).fetchone()
        assert "WHERE missing_since IS NOT NULL" in index_sql
        assert conn.execute("SELECT is_missing, missing_since FROM asset_contents").fetchall() == [(1, None)]


@pytest.fixture
def db_at_0009_with_a_missing_row(tmp_path):
    db_path = str(tmp_path / "test.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _REVISION_0009)
    _insert_missing_row(db_path)
    yield cfg, db_path


def test_migrations_have_a_single_head():
    assert len(ScriptDirectory(os.path.join(os.path.dirname(__file__), "../../alembic_db")).get_heads()) == 1


def test_0010_adds_missing_since_and_leaves_existing_rows_unstamped(db_at_0009_with_a_missing_row):
    cfg, db_path = db_at_0009_with_a_missing_row

    command.upgrade(cfg, _REVISION_0010)

    _assert_missing_since_added(db_path)


def test_a_0008_database_upgrades_through_0009_to_0010(tmp_path):
    db_path = str(tmp_path / "test.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _REVISION_0008)
    _insert_missing_row(db_path)

    command.upgrade(cfg, _REVISION_0010)

    _assert_missing_since_added(db_path)


def test_0010_downgrade_removes_the_column_and_keeps_the_row(db_at_0009_with_a_missing_row):
    cfg, db_path = db_at_0009_with_a_missing_row
    command.upgrade(cfg, _REVISION_0010)

    command.downgrade(cfg, _REVISION_0009)

    assert "missing_since" not in _columns(db_path)
    assert "ix_asset_contents_missing_since" not in _indexes(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id, is_missing FROM asset_contents").fetchall() == [("c1", 1)]
