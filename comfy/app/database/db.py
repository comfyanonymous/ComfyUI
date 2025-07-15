import logging
import os
import shutil
from importlib.resources import files

from ...cli_args import args
from ...component_model.files import get_package_as_path

Session = None

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_DB_AVAILABLE = True

logger = logging.getLogger(__name__)

def dependencies_available():
    """
    Temporary function to check if the dependencies are available
    """
    return _DB_AVAILABLE


def can_create_session():
    """
    Temporary function to check if the database is available to create a session
    During initial release there may be environmental issues (or missing dependencies) that prevent the database from being created
    """
    return dependencies_available() and Session is not None


def get_alembic_config():
    config_path = str(files("comfy") / "alembic.ini")
    scripts_path = get_package_as_path("comfy.alembic_db")

    config = Config(config_path)
    config.set_main_option("script_location", scripts_path)
    config.set_main_option("sqlalchemy.url", args.database_url)

    return config


def get_db_path():
    url = args.database_url
    if url.startswith("sqlite:///"):
        return url.split("///")[1]
    else:
        raise ValueError(f"Unsupported database URL '{url}'.")


def init_db():
    db_url = args.database_url
    logger.debug(f"Database URL: {db_url}")
    db_path = get_db_path()
    db_exists = os.path.exists(db_path)

    config = get_alembic_config()

    # Check if we need to upgrade
    engine = create_engine(db_url)
    conn = engine.connect()

    context = MigrationContext.configure(conn)
    current_rev = context.get_current_revision()

    script = ScriptDirectory.from_config(config)
    target_rev = script.get_current_head()

    if target_rev is None:
        logger.debug("No target revision found.")
    elif current_rev != target_rev:
        # Backup the database pre upgrade
        backup_path = db_path + ".bkp"
        if db_exists:
            shutil.copy(db_path, backup_path)
        else:
            backup_path = None

        try:
            command.upgrade(config, target_rev)
            logger.info(f"Database upgraded from {current_rev} to {target_rev}")
        except Exception as e:
            if backup_path:
                # Restore the database from backup if upgrade fails
                shutil.copy(backup_path, db_path)
                os.remove(backup_path)
            logger.exception("Error upgrading database: ")
            raise e

    global Session
    Session = sessionmaker(bind=engine)


def create_session():
    return Session()
