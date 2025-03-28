import logging
import os
import shutil
import sys
from app.database.models import Tag
from comfy.cli_args import args

try:
    import alembic
    import sqlalchemy
except ImportError as e:
    req_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..", "requirements.txt")
    )
    logging.error(
        f"\n\n********** ERROR ***********\n\nRequirements are not installed ({e}). Please install the requirements.txt file by running:\n{sys.executable} -s -m pip install -r {req_path}\n\nIf you are on the portable package you can run: update\\update_comfyui.bat to solve this problem\n********** ERROR **********\n"
    )
    exit(-1)

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Session = None


def get_alembic_config():
    root_path = os.path.join(os.path.dirname(__file__), "../..")
    config_path = os.path.abspath(os.path.join(root_path, "alembic.ini"))
    scripts_path = os.path.abspath(os.path.join(root_path, "alembic_db"))

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
    logging.debug(f"Database URL: {db_url}")

    config = get_alembic_config()

    # Check if we need to upgrade
    engine = create_engine(db_url)
    conn = engine.connect()

    context = MigrationContext.configure(conn)
    current_rev = context.get_current_revision()

    script = ScriptDirectory.from_config(config)
    target_rev = script.get_current_head()

    if current_rev != target_rev:
        # Backup the database pre upgrade
        db_path = get_db_path()
        backup_path = db_path + ".bkp"
        if os.path.exists(db_path):
            shutil.copy(db_path, backup_path)
        else:
            backup_path = None

        try:
            command.upgrade(config, target_rev)
            logging.info(f"Database upgraded from {current_rev} to {target_rev}")
        except Exception as e:
            if backup_path:
                # Restore the database from backup if upgrade fails
                shutil.copy(backup_path, db_path)
                os.remove(backup_path)
            logging.error(f"Error upgrading database: {e}")
            raise e

    global Session
    Session = sessionmaker(bind=engine)

    if not current_rev:
        # Init db, populate models
        from app.model_processor import model_processor

        session = create_session()
        model_processor.populate_models(session)

        # populate tags
        tags = (
            "character",
            "style",
            "concept",
            "clothing",
            "pose",
            "background",
            "vehicle",
            "object",
            "animal",
            "action",
        )
        for tag in tags:
            session.add(Tag(name=tag))

        session.commit()

def can_create_session():
    return Session is not None

def create_session():
    return Session()
