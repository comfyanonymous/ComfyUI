from typing import Optional, Any, Callable
from dataclasses import dataclass
from functools import wraps
from aiohttp import web
from app.database.db import db

primitives = (bool, str, int, float, type(None))


def is_primitive(obj):
    return isinstance(obj, primitives)


class EntityError(Exception):
    def __init__(
        self, message: str, field: str = None, value: Any = None, status_code: int = 400
    ):
        self.message = message
        self.field = field
        self.value = value
        self.status_code = status_code
        super().__init__(self.message)

    def to_json(self):
        result = {"message": self.message}
        if self.field is not None:
            result["field"] = self.field
        if self.value is not None:
            result["value"] = self.value
        return result

    def __str__(self) -> str:
        return f"{self.message} {self.field} {self.value}"


class EntityCommon(dict):
    @classmethod
    def _get_route(cls, include_key: bool):
        route = f"/db/{cls._table_name}"
        if include_key:
            route += "".join([f"/{{{k}}}" for k in cls._key_columns])
        return route

    @classmethod
    def _register_route(cls, routes, verb: str, include_key: bool, handler: Callable):
        route = cls._get_route(include_key)

        @getattr(routes, verb)(route)
        async def _(request):
            try:
                data = await handler(request)
                if data is None:
                    return web.json_response(status=204)

                return web.json_response(data)
            except EntityError as e:
                return web.json_response(e.to_json(), status=e.status_code)

    @classmethod
    def _transform(cls, row: list[Any]):
        return {col: value for col, value in zip(cls._columns, row)}

    @classmethod
    def _transform_rows(cls, rows: list[list[Any]]):
        return [cls._transform(row) for row in rows]

    @classmethod
    def _extract_key(cls, request):
        return {key: request.match_info.get(key, None) for key in cls._key_columns}

    @classmethod
    def _validate(cls, fields: list[str], data: dict, allow_missing: bool = False):
        result = {}

        if not isinstance(data, dict):
            raise EntityError("Invalid data")

        # Ensure all required fields are present
        for field in data:
            if field not in fields:
                raise EntityError("Unknown field", field)

        for key in fields:
            col = cls._columns[key]
            if key not in data:
                if col.required and not allow_missing:
                    raise EntityError("Missing field", key)
                else:
                    # e.g. for updates, we allow missing fields
                    continue
            elif data[key] is None and col.required:
                # Dont allow None for required fields
                raise EntityError("Required field", key)

            # Validate data type
            value = data[key]

            if value is not None and not is_primitive(value):
                raise EntityError("Invalid value", key, value)

            try:
                type = col.type
                if value is not None and not isinstance(value, type):
                    value = type(value)
                result[key] = value
            except Exception:
                raise EntityError("Invalid value", key, value)

        return result

    @classmethod
    def _validate_id(cls, id: dict):
        return cls._validate(cls._key_columns, id)

    @classmethod
    def _validate_data(cls, data: dict, allow_missing: bool = False):
        return cls._validate(cls._columns.keys(), data, allow_missing)

    def __setattr__(self, name, value):
        if name in self._columns:
            self[name] = value
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


class GetEntity(EntityCommon):
    @classmethod
    def get(cls, top: Optional[int] = None, where: Optional[str] = None):
        limit = ""
        if top is not None and isinstance(top, int):
            limit = f" LIMIT {top}"
        result = db.execute(
            f"SELECT * FROM {cls._table_name}{limit}{f' WHERE {where}' if where else ''}",
        )

        # Map each row in result to an instance of the class
        return cls._transform_rows(result)

    @classmethod
    def register_route(cls, routes):
        async def get_handler(request):
            top = request.rel_url.query.get("top", None)
            if top is not None:
                try:
                    top = int(top)
                except Exception:
                    raise EntityError("Invalid top parameter", "top", top)
            return cls.get(top)

        cls._register_route(routes, "get", False, get_handler)


class GetEntityById(EntityCommon):
    @classmethod
    def get_by_id(cls, id: dict):
        id = cls._validate_id(id)

        result = db.execute(
            f"SELECT * FROM {cls._table_name} WHERE {cls._where_clause}",
            *[id[key] for key in cls._key_columns],
        )

        return cls._transform_rows(result)

    @classmethod
    def register_route(cls, routes):
        async def get_by_id_handler(request):
            id = cls._extract_key(request)
            return cls.get_by_id(id)

        cls._register_route(routes, "get", True, get_by_id_handler)


class CreateEntity(EntityCommon):
    @classmethod
    def create(cls, data: dict, allow_upsert: bool = False):
        data = cls._validate_data(data)
        values = ", ".join(["?"] * len(data))
        on_conflict = ""

        data_keys = ", ".join(list(data.keys()))
        if allow_upsert:
            # Remove key columns from data
            upsert_keys = [key for key in data if key not in cls._key_columns]

            set_clause = ", ".join([f"{k} = excluded.{k}" for k in upsert_keys])
            on_conflict = f" ON CONFLICT ({', '.join(cls._key_columns)}) DO UPDATE SET {set_clause}"
        sql = f"INSERT INTO {cls._table_name} ({data_keys}) VALUES ({values}){on_conflict} RETURNING *"
        result = db.execute(
            sql,
            *[data[key] for key in data],
        )

        if len(result) == 0:
            raise EntityError("Failed to create entity", status_code=500)

        return cls._transform_rows(result)[0]

    @classmethod
    def register_route(cls, routes):
        async def create_handler(request):
            data = await request.json()
            return cls.create(data)

        cls._register_route(routes, "post", False, create_handler)


class UpdateEntity(EntityCommon):
    @classmethod
    def update(cls, id: list, data: dict):
        id = cls._validate_id(id)
        data = cls._validate_data(data, allow_missing=True)

        sql = f"UPDATE {cls._table_name} SET {', '.join([f'{k} = ?' for k in data])} WHERE {cls._where_clause} RETURNING *"
        result = db.execute(
            sql,
            *[data[key] for key in data],
            *[id[key] for key in cls._key_columns],
        )

        if len(result) == 0:
            raise EntityError("Failed to update entity", status_code=404)

        return cls._transform_rows(result)[0]

    @classmethod
    def register_route(cls, routes):
        async def update_handler(request):
            id = cls._extract_key(request)
            data = await request.json()
            return cls.update(id, data)

        cls._register_route(routes, "patch", True, update_handler)


class UpsertEntity(CreateEntity):
    @classmethod
    def upsert(cls, data: dict):
        return cls.create(data, allow_upsert=True)

    @classmethod
    def register_route(cls, routes):
        async def upsert_handler(request):
            data = await request.json()
            return cls.upsert(data)

        cls._register_route(routes, "put", False, upsert_handler)


class DeleteEntity(EntityCommon):
    @classmethod
    def delete(cls, id: list):
        id = cls._validate_id(id)
        db.execute(
            f"DELETE FROM {cls._table_name} WHERE {cls._where_clause}",
            *[id[key] for key in cls._key_columns],
        )

    @classmethod
    def register_route(cls, routes):
        async def delete_handler(request):
            id = cls._extract_key(request)
            cls.delete(id)

        cls._register_route(routes, "delete", True, delete_handler)


class BaseEntity(GetEntity, CreateEntity, UpdateEntity, DeleteEntity, GetEntityById):
    pass


@dataclass
class Column:
    type: Any
    required: bool = False
    key: bool = False
    default: Any = None


def column(type_: Any, required: bool = False, key: bool = False, default: Any = None):
    return Column(type_, required, key, default)


def table(table_name: str):
    def decorator(cls):
        # Store table name
        cls._table_name = table_name

        # Process column definitions
        columns: dict[str, Column] = {}
        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, Column):
                columns[attr_name] = attr_value

        # Store columns metadata
        cls._columns = columns
        cls._key_columns = [col for col in columns if columns[col].key]
        cls._column_csv = ", ".join([col for col in columns])
        cls._where_clause = " AND ".join([f"{col} = ?" for col in cls._key_columns])

        # Add initialization
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Initialize columns with default values
            for col_name, col_def in cls._columns.items():
                setattr(self, col_name, col_def.default)
            # Call original init
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


def test():
    @table("models")
    class Model(BaseEntity):
        id: int = column(int, required=True, key=True)
        path: str = column(str, required=True)
        name: str = column(str, required=True)
        description: Optional[str] = column(str)
        architecture: Optional[str] = column(str)
        type: str = column(str, required=True)
        hash: Optional[str] = column(str)
        source_url: Optional[str] = column(str)

    return Model


@table("test")
class Test(GetEntity, CreateEntity):
    id: int = column(int, required=True, key=True)
    test: str = column(str, required=True)


Model = test()
