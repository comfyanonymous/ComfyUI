from comfy.cli_args import args

args.memory_database = True  # force in-memory database for testing

from typing import Callable, Optional
import pytest
import pytest_asyncio
from unittest.mock import patch
from aiohttp import web
from app.database.entities import (
    DeleteEntity,
    column,
    table,
    Column,
    GetEntity,
    GetEntityById,
    CreateEntity,
    UpsertEntity,
    UpdateEntity,
)
from app.database.db import db

pytestmark = pytest.mark.asyncio


def create_table(entity):
    # reset db
    db.close()

    cols: list[Column] = entity._columns
    # Create tables as temporary so when we close the db, the tables are dropped for next test
    sql = f"CREATE TEMPORARY TABLE {entity._table_name} (  "
    for col_name, col in cols.items():
        type = None
        if col.type == int:
            type = "INTEGER"
        elif col.type == str:
            type = "TEXT"

        sql += f"{col_name} {type}"
        if col.required:
            sql += " NOT NULL"
        sql += ", "

    sql += f"PRIMARY KEY ({', '.join(entity._key_columns)})"
    sql += ")"
    db.execute(sql)


async def wrap_db(method: Callable, expected_sql: str, expected_args: list):
    with patch.object(db, "execute", wraps=db.execute) as mock:
        response = await method()
        assert mock.call_count == 1
        assert mock.call_args[0][0] == expected_sql
        assert mock.call_args[0][1:] == expected_args
        return response


@pytest.fixture
def getable_entity():
    @table("getable_entity")
    class GetableEntity(GetEntity):
        id: int = column(int, required=True, key=True)
        test: str = column(str, required=True)
        nullable: Optional[str] = column(str)

    return GetableEntity


@pytest.fixture
def getable_by_id_entity():
    @table("getable_by_id_entity")
    class GetableByIdEntity(GetEntityById):
        id: int = column(int, required=True, key=True)
        test: str = column(str, required=True)

    return GetableByIdEntity


@pytest.fixture
def getable_by_id_composite_entity():
    @table("getable_by_id_composite_entity")
    class GetableByIdCompositeEntity(GetEntityById):
        id1: str = column(str, required=True, key=True)
        id2: int = column(int, required=True, key=True)
        test: str = column(str, required=True)

    return GetableByIdCompositeEntity


@pytest.fixture
def creatable_entity():
    @table("creatable_entity")
    class CreatableEntity(CreateEntity):
        id: int = column(int, required=True, key=True)
        test: str = column(str, required=True)
        reqd: str = column(str, required=True)
        nullable: Optional[str] = column(str)

    return CreatableEntity


@pytest.fixture
def upsertable_entity():
    @table("upsertable_entity")
    class UpsertableEntity(UpsertEntity):
        id: int = column(int, required=True, key=True)
        test: str = column(str, required=True)
        reqd: str = column(str, required=True)
        nullable: Optional[str] = column(str)

    return UpsertableEntity


@pytest.fixture
def updateable_entity():
    @table("updateable_entity")
    class UpdateableEntity(UpdateEntity):
        id: int = column(int, required=True, key=True)
        reqd: str = column(str, required=True)

    return UpdateableEntity


@pytest.fixture
def deletable_entity():
    @table("deletable_entity")
    class DeletableEntity(DeleteEntity):
        id: int = column(int, required=True, key=True)

    return DeletableEntity


@pytest.fixture
def deletable_composite_entity():
    @table("deletable_composite_entity")
    class DeletableCompositeEntity(DeleteEntity):
        id1: str = column(str, required=True, key=True)
        id2: int = column(int, required=True, key=True)

    return DeletableCompositeEntity


@pytest.fixture()
def entity(request):
    value = request.getfixturevalue(request.param)
    create_table(value)
    return value


@pytest_asyncio.fixture
async def client(aiohttp_client, app):
    return await aiohttp_client(app)


@pytest.fixture
def app(entity):
    app = web.Application()
    routes = web.RouteTableDef()
    entity.register_route(routes)
    app.add_routes(routes)
    return app


@pytest.mark.parametrize("entity", ["getable_entity"], indirect=True)
async def test_get_model_empty_response(client):
    expected_sql = "SELECT * FROM getable_entity"
    expected_args = ()
    response = await wrap_db(
        lambda: client.get("/db/getable_entity"), expected_sql, expected_args
    )

    assert response.status == 200
    assert await response.json() == []


@pytest.mark.parametrize("entity", ["getable_entity"], indirect=True)
async def test_get_model_with_data(client):
    # seed db
    db.execute(
        "INSERT INTO getable_entity (id, test, nullable) VALUES (1, 'test1', NULL), (2, 'test2', 'test2')"
    )

    expected_sql = "SELECT * FROM getable_entity"
    expected_args = ()
    response = await wrap_db(
        lambda: client.get("/db/getable_entity"), expected_sql, expected_args
    )

    assert response.status == 200
    assert await response.json() == [
        {"id": 1, "test": "test1", "nullable": None},
        {"id": 2, "test": "test2", "nullable": "test2"},
    ]


@pytest.mark.parametrize("entity", ["getable_entity"], indirect=True)
async def test_get_model_with_top_parameter(client):
    # seed with 3 rows
    db.execute(
        "INSERT INTO getable_entity (id, test, nullable) VALUES (1, 'test1', NULL), (2, 'test2', 'test2'), (3, 'test3', 'test3')"
    )

    expected_sql = "SELECT * FROM getable_entity LIMIT 2"
    expected_args = ()
    response = await wrap_db(
        lambda: client.get("/db/getable_entity?top=2"),
        expected_sql,
        expected_args,
    )

    assert response.status == 200
    assert await response.json() == [
        {"id": 1, "test": "test1", "nullable": None},
        {"id": 2, "test": "test2", "nullable": "test2"},
    ]


@pytest.mark.parametrize("entity", ["getable_entity"], indirect=True)
async def test_get_model_with_invalid_top_parameter(client):
    response = await client.get("/db/getable_entity?top=hello")
    assert response.status == 400
    assert await response.json() == {
        "message": "Invalid top parameter",
        "field": "top",
        "value": "hello",
    }


@pytest.mark.parametrize("entity", ["getable_by_id_entity"], indirect=True)
async def test_get_model_by_id_empty_response(client):
    # seed db
    db.execute("INSERT INTO getable_by_id_entity (id, test) VALUES (1, 'test1')")

    expected_sql = "SELECT * FROM getable_by_id_entity WHERE id = ?"
    expected_args = (1,)
    response = await wrap_db(
        lambda: client.get("/db/getable_by_id_entity/1"),
        expected_sql,
        expected_args,
    )

    assert response.status == 200
    assert await response.json() == [
        {"id": 1, "test": "test1"},
    ]


@pytest.mark.parametrize("entity", ["getable_by_id_entity"], indirect=True)
async def test_get_model_by_id_with_invalid_id(client):
    response = await client.get("/db/getable_by_id_entity/hello")
    assert response.status == 400
    assert await response.json() == {
        "message": "Invalid value",
        "field": "id",
        "value": "hello",
    }


@pytest.mark.parametrize("entity", ["getable_by_id_composite_entity"], indirect=True)
async def test_get_model_by_id_composite(client):
    # seed db
    db.execute(
        "INSERT INTO getable_by_id_composite_entity (id1, id2, test) VALUES ('one', 2, 'test')"
    )

    expected_sql = (
        "SELECT * FROM getable_by_id_composite_entity WHERE id1 = ? AND id2 = ?"
    )
    expected_args = ("one", 2)
    response = await wrap_db(
        lambda: client.get("/db/getable_by_id_composite_entity/one/2"),
        expected_sql,
        expected_args,
    )

    assert response.status == 200
    assert await response.json() == [
        {"id1": "one", "id2": 2, "test": "test"},
    ]


@pytest.mark.parametrize("entity", ["getable_by_id_composite_entity"], indirect=True)
async def test_get_model_by_id_composite_with_invalid_id(client):
    response = await client.get("/db/getable_by_id_composite_entity/hello/hello")
    assert response.status == 400
    assert await response.json() == {
        "message": "Invalid value",
        "field": "id2",
        "value": "hello",
    }


@pytest.mark.parametrize("entity", ["creatable_entity"], indirect=True)
async def test_create_model(client):
    expected_sql = (
        "INSERT INTO creatable_entity (id, test, reqd) VALUES (?, ?, ?) RETURNING *"
    )
    expected_args = (1, "test1", "reqd1")
    response = await wrap_db(
        lambda: client.post(
            "/db/creatable_entity", json={"id": 1, "test": "test1", "reqd": "reqd1"}
        ),
        expected_sql,
        expected_args,
    )

    assert response.status == 200
    assert await response.json() == {
        "id": 1,
        "test": "test1",
        "reqd": "reqd1",
        "nullable": None,
    }


@pytest.mark.parametrize("entity", ["creatable_entity"], indirect=True)
async def test_create_model_missing_required_field(client):
    response = await client.post(
        "/db/creatable_entity", json={"id": 1, "test": "test1"}
    )

    assert response.status == 400
    assert await response.json() == {
        "message": "Missing field",
        "field": "reqd",
    }


@pytest.mark.parametrize("entity", ["creatable_entity"], indirect=True)
async def test_create_model_missing_key_field(client):
    response = await client.post(
        "/db/creatable_entity",
        json={"test": "test1", "reqd": "reqd1"},  # Missing 'id' which is a key
    )

    assert response.status == 400
    assert await response.json() == {
        "message": "Missing field",
        "field": "id",
    }


@pytest.mark.parametrize("entity", ["creatable_entity"], indirect=True)
async def test_create_model_invalid_key_data(client):
    response = await client.post(
        "/db/creatable_entity",
        json={
            "id": "not_an_integer",
            "test": "test1",
            "reqd": "reqd1",
        },  # id should be int
    )

    assert response.status == 400
    assert await response.json() == {
        "message": "Invalid value",
        "field": "id",
        "value": "not_an_integer",
    }


@pytest.mark.parametrize("entity", ["creatable_entity"], indirect=True)
async def test_create_model_invalid_field_data(client):
    response = await client.post(
        "/db/creatable_entity",
        json={"id": "aaa", "test": "123", "reqd": "reqd1"},  # id should be int
    )

    assert response.status == 400
    assert await response.json() == {
        "message": "Invalid value",
        "field": "id",
        "value": "aaa",
    }


@pytest.mark.parametrize("entity", ["creatable_entity"], indirect=True)
async def test_create_model_invalid_field_type(client):
    response = await client.post(
        "/db/creatable_entity",
        json={
            "id": 1,
            "test": ["invalid_array"],
            "reqd": "reqd1",
        },  # test should be string
    )

    assert response.status == 400
    assert await response.json() == {
        "message": "Invalid value",
        "field": "test",
        "value": ["invalid_array"],
    }


@pytest.mark.parametrize("entity", ["creatable_entity"], indirect=True)
async def test_create_model_invalid_field_name(client):
    response = await client.post(
        "/db/creatable_entity",
        json={"id": 1, "test": "test1", "reqd": "reqd1", "nonexistent_field": "value"},
    )

    assert response.status == 400
    assert await response.json() == {
        "message": "Unknown field",
        "field": "nonexistent_field",
    }


@pytest.mark.parametrize("entity", ["upsertable_entity"], indirect=True)
async def test_upsert_model(client):
    expected_sql = (
        "INSERT INTO upsertable_entity (id, test, reqd) VALUES (?, ?, ?) "
        "ON CONFLICT (id) DO UPDATE SET test = excluded.test, reqd = excluded.reqd "
        "RETURNING *"
    )
    expected_args = (1, "test1", "reqd1")
    response = await wrap_db(
        lambda: client.put(
            "/db/upsertable_entity", json={"id": 1, "test": "test1", "reqd": "reqd1"}
        ),
        expected_sql,
        expected_args,
    )

    assert response.status == 200
    assert await response.json() == {
        "id": 1,
        "test": "test1",
        "reqd": "reqd1",
        "nullable": None,
    }


@pytest.mark.parametrize("entity", ["updateable_entity"], indirect=True)
async def test_update_model(client):
    # seed db
    db.execute("INSERT INTO updateable_entity (id, reqd) VALUES (1, 'test1')")

    expected_sql = "UPDATE updateable_entity SET reqd = ? WHERE id = ? RETURNING *"
    expected_args = ("updated_test", 1)
    response = await wrap_db(
        lambda: client.patch("/db/updateable_entity/1", json={"reqd": "updated_test"}),
        expected_sql,
        expected_args,
    )

    assert response.status == 200
    assert await response.json() == {
        "id": 1,
        "reqd": "updated_test",
    }


@pytest.mark.parametrize("entity", ["updateable_entity"], indirect=True)
async def test_update_model_reject_null_required_field(client):
    response = await client.patch("/db/updateable_entity/1", json={"reqd": None})

    assert response.status == 400
    assert await response.json() == {
        "message": "Required field",
        "field": "reqd",
    }


@pytest.mark.parametrize("entity", ["updateable_entity"], indirect=True)
async def test_update_model_reject_invalid_field(client):
    response = await client.patch("/db/updateable_entity/1", json={"hello": "world"})

    assert response.status == 400
    assert await response.json() == {
        "message": "Unknown field",
        "field": "hello",
    }


@pytest.mark.parametrize("entity", ["updateable_entity"], indirect=True)
async def test_update_model_reject_missing_record(client):
    response = await client.patch(
        "/db/updateable_entity/1", json={"reqd": "updated_test"}
    )

    assert response.status == 404
    assert await response.json() == {
        "message": "Failed to update entity",
    }


@pytest.mark.parametrize("entity", ["deletable_entity"], indirect=True)
async def test_delete_model(client):
    expected_sql = "DELETE FROM deletable_entity WHERE id = ?"
    expected_args = (1,)
    response = await wrap_db(
        lambda: client.delete("/db/deletable_entity/1"),
        expected_sql,
        expected_args,
    )

    assert response.status == 204


@pytest.mark.parametrize("entity", ["deletable_composite_entity"], indirect=True)
async def test_delete_model_composite_key(client):
    expected_sql = "DELETE FROM deletable_composite_entity WHERE id1 = ? AND id2 = ?"
    expected_args = ("one", 2)
    response = await wrap_db(
        lambda: client.delete("/db/deletable_composite_entity/one/2"),
        expected_sql,
        expected_args,
    )

    assert response.status == 204
