from sqlalchemy import (
    Column,
    Text,
    DateTime,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


def to_dict(obj):
    fields = obj.__table__.columns.keys()
    return {
        field: (val.to_dict() if hasattr(val, "to_dict") else val)
        for field in fields
        if (val := getattr(obj, field))
    }


class Model(Base):
    """
    SQLAlchemy model representing a model file in the system.

    This class defines the database schema for storing information about model files,
    including their type, path, hash, and when they were added to the system.

    Attributes:
        type (Text): The type of the model, this is the name of the folder in the models folder (primary key)
        path (Text): The file path of the model relative to the type folder (primary key)
        hash (Text): A sha256 hash of the model file
        date_added (DateTime): Timestamp of when the model was added to the system
    """

    __tablename__ = "model"

    type = Column(Text, primary_key=True)
    path = Column(Text, primary_key=True)
    hash = Column(Text)
    date_added = Column(DateTime, server_default=func.now())

    def to_dict(self):
        """
        Convert the model instance to a dictionary representation.

        Returns:
            dict: A dictionary containing the attributes of the model
        """
        dict = to_dict(self)
        return dict
