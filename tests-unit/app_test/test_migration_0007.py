"""Tests specific to migration 0007 (record/content split)."""
import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config

_BASELINE_0006 = "0006_add_loader_path"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


@pytest.fixture
def db_at_0006(tmp_path):
    db_path = str(tmp_path / "test.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _BASELINE_0006)
    yield cfg, db_path


def test_0007_upgrade_from_0006(db_at_0006):
    """Upgrade from 0006 to head succeeds; new tables present, old gone."""
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "assets" in tables
    assert "asset_contents" in tables
    assert "asset_system_state" in tables
    assert "asset_references" not in tables


def test_0007_schema_has_expected_columns(db_at_0006):
    """After upgrade, key columns exist on new tables."""
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        asset_cols = {r[1] for r in conn.execute("PRAGMA table_info(assets)")}
        content_cols = {r[1] for r in conn.execute("PRAGMA table_info(asset_contents)")}
    assert {"id", "content_id", "name", "loader_path", "updated_at", "last_access_time"} <= asset_cols
    assert {"id", "hash", "path", "is_missing", "mtime_ns"} <= content_cols


def test_0007_downgrade_restores_0006_schema(db_at_0006):
    """Downgrade from head back to 0006 restores asset_references."""
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    command.downgrade(cfg, _BASELINE_0006)
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "asset_references" in tables
    assert "asset_contents" not in tables


def test_0007_invariants_on_migrated_db(db_at_0006):
    """After upgrade, schema invariants hold."""
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        # Insert a content row and a record
        conn.execute(
            "INSERT INTO asset_contents(id, path, is_missing, size_bytes, created_at) "
            "VALUES ('c1', '/tmp/f1', 0, 0, '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO assets(id, content_id, name, created_at, updated_at) "
            "VALUES ('a1', 'c1', 'test', '2024-01-01', '2024-01-01')"
        )
        conn.commit()

        # Duplicate live path should fail (partial unique)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO asset_contents(id, path, is_missing, size_bytes, created_at) "
                "VALUES ('c2', '/tmp/f1', 0, 0, '2024-01-01')"
            )
            conn.commit()
        conn.rollback()

        # Live + missing same path should succeed
        conn.execute(
            "INSERT INTO asset_contents(id, path, is_missing, size_bytes, created_at) "
            "VALUES ('c3', '/tmp/f1', 1, 0, '2024-01-01')"
        )
        conn.commit()

        # Equal-hash rows should coexist (no hash UNIQUE)
        conn.execute("UPDATE asset_contents SET hash='abc123' WHERE id='c1'")
        conn.execute(
            "INSERT INTO asset_contents(id, path, hash, is_missing, size_bytes, created_at) "
            "VALUES ('c4', '/tmp/f2', 'abc123', 0, 0, '2024-01-01')"
        )
        conn.commit()


def test_0007_orm_parity(db_at_0006, tmp_path):
    """Base.metadata.create_all produces same table names as alembic upgrade."""
    from sqlalchemy import create_engine, inspect

    import app.assets.database.models as asset_models
    from app.database.models import Base

    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        alembic_tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'alembic%' AND name NOT LIKE 'sqlite%'"
            )
        }

    alembic_engine = create_engine(f"sqlite:///{db_path}")
    orm_db = str(tmp_path / "orm.db")
    engine = create_engine(f"sqlite:///{orm_db}")
    alembic_inspector = inspect(alembic_engine)
    Base.metadata.create_all(engine)
    orm_inspector = inspect(engine)
    orm_tables = set(orm_inspector.get_table_names())
    assert asset_models.AssetMeta.__tablename__ == "asset_meta"

    alembic_indexes = {
        (index["name"], tuple(index["column_names"]))
        for index in alembic_inspector.get_indexes("asset_meta")
    }
    orm_indexes = {
        (index.name, tuple(index.columns.keys()))
        for index in Base.metadata.tables["asset_meta"].indexes
    }

    assert alembic_tables == orm_tables, f"Mismatch: alembic={alembic_tables}, orm={orm_tables}"
    assert alembic_indexes == orm_indexes, f"Mismatch: alembic={alembic_indexes}, orm={orm_indexes}"


def test_0007_downgrade_chain_past_0003_succeeds(db_at_0006):
    """A multi-step downgrade from head must not crash mid-chain.

    0007's downgrade recreates asset_references (and asset_reference_meta /
    asset_reference_tags). If it omits the indexes those tables had at 0006,
    the older downgrades raise "No such index": 0003 on
    ix_asset_references_preview_id, then 0002 on the remaining asset_references
    and asset_reference_meta/tags indexes. Downgrading to 0001_assets exercises
    the entire chain through 0002's downgrade. (base is not targeted: 0001's own
    downgrade uses a SQLite-incompatible DROP CONSTRAINT that predates and is
    unrelated to this fix.)

    The single-step 0007->0006 test does not catch this because 0003/0002's
    downgrades never run.
    """
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0001_assets")
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # 0002's downgrade recreated the 0001-era schema and removed the split tables.
    assert "assets_info" in tables
    assert "asset_references" not in tables
    assert "asset_contents" not in tables


def test_0007_downgrade_restores_0006_asset_references_indexes(db_at_0006, tmp_path):
    """0007's downgrade must recreate asset_references with the EXACT index set
    present at 0006 — no more, no less.

    Compared against a fresh DB left at 0006 so the expected set is not
    hardcoded: any omitted (or invented) index is a mismatch, and omissions are
    exactly what break the older downgrades mid-chain.
    """
    from sqlalchemy import create_engine, inspect

    cfg, db_path = db_at_0006

    reference_db = str(tmp_path / "reference_0006.db")
    reference_cfg = _make_config(reference_db)
    command.upgrade(reference_cfg, _BASELINE_0006)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, _BASELINE_0006)

    def _asset_reference_indexes(path: str) -> set[tuple[str, tuple[str, ...]]]:
        engine = create_engine(f"sqlite:///{path}")
        try:
            return {
                (index["name"], tuple(index["column_names"]))
                for index in inspect(engine).get_indexes("asset_references")
            }
        finally:
            engine.dispose()

    restored = _asset_reference_indexes(db_path)
    expected = _asset_reference_indexes(reference_db)
    assert restored == expected, f"index drift: restored={restored}, expected={expected}"
