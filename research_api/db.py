"""Research Workbench database session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from research_app.config import get_research_paths

_DB_PATH = get_research_paths().db_path
DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
session_maker = sessionmaker(bind=engine)


def create_session():
    """Create a new database session."""
    return session_maker()


def init_db():
    """Create all research tables."""
    from research_api.models import Base
    Base.metadata.create_all(engine)
