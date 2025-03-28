from sqlalchemy import (
    Column,
    Integer,
    Text,
    DateTime,
    Table,
    ForeignKeyConstraint,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


def to_dict(obj):
    fields = obj.__table__.columns.keys()
    return {
        field: (val.to_dict() if hasattr(val, "to_dict") else val)
        for field in fields
        if (val := getattr(obj, field))
    }


ModelTag = Table(
    "model_tag",
    Base.metadata,
    Column(
        "model_type",
        Text,
        primary_key=True,
    ),
    Column(
        "model_path",
        Text,
        primary_key=True,
    ),
    Column("tag_id", Integer, primary_key=True),
    ForeignKeyConstraint(
        ["model_type", "model_path"], ["model.type", "model.path"], ondelete="CASCADE"
    ),
    ForeignKeyConstraint(["tag_id"], ["tag.id"], ondelete="CASCADE"),
)


class Model(Base):
    __tablename__ = "model"

    type = Column(Text, primary_key=True)
    path = Column(Text, primary_key=True)
    title = Column(Text)
    description = Column(Text)
    architecture = Column(Text)
    hash = Column(Text)
    source_url = Column(Text)
    date_added = Column(DateTime, server_default=func.now())

    # Relationship with tags
    tags = relationship("Tag", secondary=ModelTag, back_populates="models")

    def to_dict(self):
        dict = to_dict(self)
        dict["tags"] = [tag.to_dict() for tag in self.tags]
        return dict


class Tag(Base):
    __tablename__ = "tag"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, unique=True)

    # Relationship with models
    models = relationship("Model", secondary=ModelTag, back_populates="tags")

    def to_dict(self):
        return to_dict(self)
