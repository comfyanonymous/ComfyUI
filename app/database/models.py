from sqlalchemy.orm import declarative_base

Base = declarative_base()


def to_dict(obj):
    fields = obj.__table__.columns.keys()
    return {
        field: (val.to_dict() if hasattr(val, "to_dict") else val)
        for field in fields
        if (val := getattr(obj, field))
    }

# TODO: Define models here
