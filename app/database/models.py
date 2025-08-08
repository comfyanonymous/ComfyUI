from sqlalchemy import (
    Column,
    Integer,
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
    sqlalchemy model representing a model file in the system.

    This class defines the database schema for storing information about model files,
    including their type, path, hash, and when they were added to the system.

    Attributes:
        type (Text): The type of the model, this is the name of the folder in the models folder (primary key)
        path (Text): The file path of the model relative to the type folder (primary key)
        file_name (Text): The name of the model file
        file_size (Integer): The size of the model file in bytes
        hash (Text): A hash of the model file
        hash_algorithm (Text): The algorithm used to generate the hash
        source_url (Text): The URL of the model file
        date_added (DateTime): Timestamp of when the model was added to the system
    """

    __tablename__ = "model"

    type = Column(Text, primary_key=True)
    path = Column(Text, primary_key=True)
    file_name = Column(Text)
    file_size = Column(Integer)
    hash = Column(Text)
    hash_algorithm = Column(Text)
    source_url = Column(Text)
    date_added = Column(DateTime, server_default=func.now())

    def to_dict(self):
        """
        Convert the model instance to a dictionary representation.

        Returns:
            dict: A dictionary containing the attributes of the model
        """
        dict = to_dict(self)
        return dict
