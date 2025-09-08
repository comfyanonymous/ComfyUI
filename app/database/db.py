import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from comfy.cli_args import args
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

LOGGER = logging.getLogger(__name__)
ENGINE: Optional[AsyncEngine] = None
SESSION: Optional[async_sessionmaker] = None


def _root_paths():
    """Resolve alembic.ini and migrations script folder."""
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config_path = os.path.abspath(os.path.join(root_path, "alembic.ini"))
    scripts_path = os.path.abspath(os.path.join(root_path, "alembic_db"))
    return config_path, scripts_path


def _absolutize_sqlite_url(db_url: str) -> str:
    """Make SQLite database path absolute. No-op for non-SQLite URLs."""
    try:
        u = make_url(db_url)
    except Exception:
        return db_url

    if not u.drivername.startswith("sqlite"):
        return db_url

    db_path: str = u.database or ""
    if isinstance(db_path, str) and db_path.startswith("file:"):
        return str(u)  # Do not touch SQLite URI databases like: "file:xxx?mode=memory&cache=shared"
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(os.path.join(os.getcwd(), db_path))
        u = u.set(database=db_path)
    return str(u)


def _normalize_sqlite_memory_url(db_url: str) -> tuple[str, bool]:
    """
    If db_url points at an in-memory SQLite DB (":memory:" or file:... mode=memory),
    rewrite it to a *named* shared in-memory URI and ensure 'uri=true' is present.
    Returns: (normalized_url, is_memory)
    """
    try:
        u = make_url(db_url)
    except Exception:
        return db_url, False
    if not u.drivername.startswith("sqlite"):
        return db_url, False

    db = u.database or ""
    if db == ":memory:":
        u = u.set(database=f"file:comfyui_db_{os.getpid()}?mode=memory&cache=shared&uri=true")
        return str(u), True
    if isinstance(db, str) and db.startswith("file:") and "mode=memory" in db:
        if "uri=true" not in db:
            u = u.set(database=(db + ("&" if "?" in db else "?") + "uri=true"))
        return str(u), True
    return str(u), False


def _to_sync_driver_url(async_url: str) -> str:
    """Convert an async SQLAlchemy URL to a sync URL for Alembic."""
    u = make_url(async_url)
    driver = u.drivername

    if driver.startswith("sqlite+aiosqlite"):
        u = u.set(drivername="sqlite")
    elif driver.startswith("postgresql+asyncpg"):
        u = u.set(drivername="postgresql")
    else:
        # Generic: strip the async driver part if present
        if "+" in driver:
            u = u.set(drivername=driver.split("+", 1)[0])

    return str(u)


def _get_sqlite_file_path(sync_url: str) -> Optional[str]:
    """Return the on-disk path for a SQLite URL, else None."""
    try:
        u = make_url(sync_url)
    except Exception:
        return None

    if not u.drivername.startswith("sqlite"):
        return None
    db_path = u.database
    if isinstance(db_path, str) and db_path.startswith("file:"):
        return None  # Not a real file if it is a URI like "file:...?"
    return db_path


def _get_alembic_config(sync_url: str) -> Config:
    """Prepare Alembic Config with script location and DB URL."""
    config_path, scripts_path = _root_paths()
    cfg = Config(config_path)
    cfg.set_main_option("script_location", scripts_path)
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


async def init_db_engine() -> None:
    """Initialize async engine + sessionmaker and run migrations to head.

    This must be called once on application startup before any DB usage.
    """
    global ENGINE, SESSION

    if ENGINE is not None:
        return

    raw_url = args.database_url
    if not raw_url:
        raise RuntimeError("Database URL is not configured.")

    db_url, is_mem = _normalize_sqlite_memory_url(raw_url)
    db_url = _absolutize_sqlite_url(db_url)

    # Prepare async engine
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args = {
            "check_same_thread": False,
            "timeout": 12,
        }
        if is_mem:
            connect_args["uri"] = True

    ENGINE = create_async_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )

    # Enforce SQLite pragmas on the async engine
    if db_url.startswith("sqlite"):
        async with ENGINE.begin() as conn:
            if not is_mem:
                # WAL for concurrency and durability, Foreign Keys for referential integrity
                current_mode = (await conn.execute(text("PRAGMA journal_mode;"))).scalar()
                if str(current_mode).lower() != "wal":
                    new_mode = (await conn.execute(text("PRAGMA journal_mode=WAL;"))).scalar()
                    if str(new_mode).lower() != "wal":
                        raise RuntimeError("Failed to set SQLite journal mode to WAL.")
                    LOGGER.info("SQLite journal mode set to WAL.")

            await conn.execute(text("PRAGMA foreign_keys = ON;"))
            await conn.execute(text("PRAGMA synchronous = NORMAL;"))

    await _run_migrations(raw_url=db_url, connect_args=connect_args)

    SESSION = async_sessionmaker(
        bind=ENGINE,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def _run_migrations(raw_url: str, connect_args: dict) -> None:
    """
    Run Alembic migrations up to head.

    We deliberately use a synchronous engine for migrations because Alembic's
    programmatic API is synchronous by default and this path is robust.
    """
    # Convert to sync URL and make SQLite URL an absolute one
    sync_url = _to_sync_driver_url(raw_url)
    sync_url, is_mem = _normalize_sqlite_memory_url(sync_url)
    sync_url = _absolutize_sqlite_url(sync_url)

    cfg = _get_alembic_config(sync_url)
    engine = create_engine(sync_url, future=True, connect_args=connect_args)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()

    script = ScriptDirectory.from_config(cfg)
    target_rev = script.get_current_head()

    if target_rev is None:
        LOGGER.warning("Alembic: no target revision found.")
        return

    if current_rev == target_rev:
        LOGGER.debug("Alembic: database already at head %s", target_rev)
        return

    LOGGER.info("Alembic: upgrading database from %s to %s", current_rev, target_rev)

    # Optional backup for SQLite file DBs
    backup_path = None
    sqlite_path = _get_sqlite_file_path(sync_url)
    if sqlite_path and os.path.exists(sqlite_path):
        backup_path = sqlite_path + ".bkp"
        try:
            shutil.copy(sqlite_path, backup_path)
        except Exception as exc:
            LOGGER.warning("Failed to create SQLite backup before migration: %s", exc)

    try:
        command.upgrade(cfg, target_rev)
    except Exception:
        if backup_path and os.path.exists(backup_path):
            LOGGER.exception("Error upgrading database, attempting restore from backup.")
            try:
                shutil.copy(backup_path, sqlite_path)  # restore
                os.remove(backup_path)
            except Exception as re:
                LOGGER.error("Failed to restore SQLite backup: %s", re)
        else:
            LOGGER.exception("Error upgrading database, backup is not available.")
        raise


def get_engine():
    """Return the global async engine (initialized after init_db_engine())."""
    if ENGINE is None:
        raise RuntimeError("Engine is not initialized. Call init_db_engine() first.")
    return ENGINE


def get_session_maker():
    """Return the global async_sessionmaker (initialized after init_db_engine())."""
    if SESSION is None:
        raise RuntimeError("Session maker is not initialized. Call init_db_engine() first.")
    return SESSION


@asynccontextmanager
async def session_scope():
    """Async context manager for a unit of work:

    async with session_scope() as sess:
        ... use sess ...
    """
    maker = get_session_maker()
    async with maker() as sess:
        try:
            yield sess
            await sess.commit()
        except Exception:
            await sess.rollback()
            raise


async def create_session():
    """Convenience helper to acquire a single AsyncSession instance.

    Typical usage:
        async with (await create_session()) as sess:
            ...
    """
    maker = get_session_maker()
    return maker()
