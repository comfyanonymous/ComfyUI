"""Research Workbench database session management."""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

_COMFYUI_ROOT = Path(__file__).parent.parent
_DB_PATH = _COMFYUI_ROOT / "research_workbench.db"

DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
session_maker = sessionmaker(bind=engine)


def create_session():
    """Create a new database session."""
    return session_maker()


def init_db():
    """Create all research tables."""
    from research_api.models import Base
    Base.metadata.create_all(engine)
