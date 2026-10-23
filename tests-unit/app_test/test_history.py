import copy
import json
import logging
import sqlite3
import threading
import time
from enum import IntEnum
from types import SimpleNamespace
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from app.history import HistoryStore, project_record, HistoryStorageError
from app.database import db
from comfy.cli_args import args

args.cpu = True
from execution import PromptQueue


def record(id="one"):
    return {
        "prompt": [
            0,
            id,
            {"1": {"inputs": {"api_key_comfy_org": "secret", "text": "hello"}}},
            {
                "extra_pnginfo": {
                    "workflow": {"id": "flow", "auth_token_comfy_org": "hidden"}
                },
                "create_time": 12,
            },
            ["1"],
        ],
        "outputs": {
            "1": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}
        },
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                ["execution_start", {"timestamp": 1}],
                [
                    "execution_error",
                    {
                        "timestamp": 2,
                        "current_inputs": "private",
                        "traceback": "trace",
                        "exception_message": "secret",
                    },
                ],
            ],
        },
    }


def test_projection():
    value = record()
    before = copy.deepcopy(value)
    canonical, encoded = project_record("one", value)
    assert value == before
    assert all(
        s not in encoded
        for s in ("secret", "hidden", "private", "traceback", "current_inputs")
    )
    assert canonical["prompt"][3]["extra_pnginfo"]["workflow"]["id"] == "flow"
    assert canonical["outputs"] == value["outputs"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_unsupported(bad):
    value = record()
    value["outputs"]["bad"] = bad
    with pytest.raises(HistoryStorageError):
        project_record("one", value)


@pytest.mark.parametrize(
    "bad",
    [
        ["execution_error", None],
        ["execution_start", []],
        ["execution_start"],
        3,
        ["execution_start", {"timestamp": "bad"}],
    ],
)
def test_bad_messages(bad):
    value = record()
    value["status"]["messages"] = [bad]
    with pytest.raises(HistoryStorageError):
        project_record("one", value)


def test_size(monkeypatch):
    monkeypatch.setattr("app.history.MAX_RECORD_BYTES", 10)
    with pytest.raises(HistoryStorageError):
        project_record("one", record())


def test_order_and_remove(file_db):
    store = HistoryStore(2)
    for id in ["one", "two", "one"]:
        store.save(id, record(id))
    assert list(store.load()) == ["one", "two"]
    assert store.save("three", record("three"))[1] == ["one"]
    assert list(HistoryStore(2).load()) == ["two", "three"]
    store.remove(["unknown", "two"])
    assert list(store.load()) == ["three"]
    store.remove([], clear=True)
    assert store.load() == {}


def test_prune_rollback(file_db, monkeypatch):
    store = HistoryStore(1)
    store.save("one", record())

    def fail(session):
        raise OperationalError("test", {}, Exception("authored"))

    with monkeypatch.context() as m:
        m.setattr(store, "_prune", fail)
        with pytest.raises(HistoryStorageError):
            store.save("two", record("two"))
    assert list(store.load()) == ["one"]


@pytest.mark.parametrize(
    "version,payload", [(99, None), (1, '{"bad":1}'), (1, "{"), (1, None)]
)
def test_bad_restore(file_db, version, payload):
    store = HistoryStore(2)
    store.save("one", record())
    invalid = record()
    invalid["status"]["messages"] = [["execution_error", None]]
    with db.create_write_session() as session, session.begin():
        session.execute(
            text("UPDATE execution_history SET format_version=:v,record_json=:p"),
            {"v": version, "p": payload or json.dumps(invalid)},
        )
    with pytest.raises(HistoryStorageError):
        store.load()


def complete(q, id):
    r = record(id)
    q.put(r["prompt"])
    _, item = q.get()
    q.task_done(
        item, {"outputs": r["outputs"]}, PromptQueue.ExecutionStatus(**r["status"])
    )


def test_sticky_failure_and_delete_atomicity(file_db, monkeypatch):
    store = HistoryStore(2)
    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    q.initialize_history(store)
    complete(q, "one")

    def fail(*a, **kw):
        raise HistoryStorageError("authored")

    with monkeypatch.context() as m:
        m.setattr(store, "save", fail)
        complete(q, "two")
    complete(q, "three")
    assert q.history_persistence_status() == {
        "enabled": True,
        "healthy": False,
        "failure_count": 1,
    }
    before = copy.deepcopy(q.history)
    with monkeypatch.context() as m:
        m.setattr(store, "remove", fail)
        with pytest.raises(HistoryStorageError):
            q.remove_history(["three"], clear=True)
    assert q.history == before
    q.remove_history(["three"])
    assert "three" not in store.load()


def test_disabled_cap():
    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    q.history = {str(i): {} for i in range(10001)}
    complete(q, "new")
    assert len(q.history) == 10001


@pytest.mark.parametrize(
    "url",
    [
        "sqlite://",
        "sqlite:///:memory:",
        "sqlite:///:memory:?cache=shared",
        "sqlite:///file:memdb?mode=memory&cache=shared&uri=true",
        "sqlite:///file::memory:?cache=shared&uri=true",
    ],
)
def test_reject_memory_urls(url):
    assert not db.is_persistent_sqlite_url(url)


def test_file_url():
    assert db.is_persistent_sqlite_url("sqlite:////tmp/authored.db")


@pytest.mark.parametrize("change", ["overflow", "workflow"])
def test_corrupt_nested_restore(file_db, change):
    store = HistoryStore(1)
    store.save("one", record())
    value = record()
    if change == "workflow":
        value["prompt"][3]["extra_pnginfo"] = None
    else:
        value["outputs"]["value"] = 1
    payload = json.dumps(value)
    if change == "overflow":
        payload = payload.replace('"value": 1', '"value": 1e999')
    with db.create_write_session() as session, session.begin():
        session.execute(
            text("UPDATE execution_history SET record_json=:p"), {"p": payload}
        )
    with pytest.raises(HistoryStorageError):
        store.load()


def test_real_write_wait_does_not_hold_queue_mutex(file_db):

    store = HistoryStore(2)
    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    q.initialize_history(store)
    complete(q, "one")
    connection = sqlite3.connect(file_db, isolation_level=None)
    connection.execute("BEGIN IMMEDIATE")
    started = threading.Event()
    errors = []

    def remove():
        started.set()
        try:
            q.remove_history(clear=True)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=remove)
    worker.start()
    assert started.wait(1)
    try:
        time.sleep(0.1)
        assert worker.is_alive()
        start = time.monotonic()
        assert "one" in q.get_history()
        assert q.history_persistence_status()["healthy"]
        assert time.monotonic() - start < 0.5
    finally:
        connection.rollback()
        connection.close()
        worker.join(10)
    assert not worker.is_alive() and not errors
    assert q.get_history() == {} and store.load() == {}


@pytest.mark.parametrize(
    "assets,history", [(False, False), (True, False), (False, True), (True, True)]
)
def test_startup_matrix(monkeypatch, assets, history):
    import main

    calls = []
    monkeypatch.setattr(main.args, "enable_history", history)
    monkeypatch.setattr(main, "dependencies_available", lambda: True)
    monkeypatch.setattr(main, "get_database_url", lambda: "sqlite:////tmp/authored.db")
    monkeypatch.setattr(main, "init_db", lambda: calls.append("db"))
    monkeypatch.setattr(main, "warn_if_database_in_use", lambda: calls.append("warn"))
    manager = SimpleNamespace(enabled=assets, startup=lambda: calls.append("startup"))
    main.setup_database(manager)
    assert calls == (["db", "startup"] if assets or history else ["warn", "startup"])


@pytest.mark.parametrize(
    "dependency,url",
    [(False, "sqlite:////tmp/a.db"), (True, "sqlite:///:memory:?cache=shared")],
)
def test_requested_history_startup_failure(monkeypatch, dependency, url):
    import main

    monkeypatch.setattr(main.args, "enable_history", True)
    monkeypatch.setattr(main, "dependencies_available", lambda: dependency)
    monkeypatch.setattr(main, "get_database_url", lambda: url)
    with pytest.raises(SystemExit):
        main.setup_database(SimpleNamespace(enabled=False, startup=lambda: None))


def test_completion_stays_visible_while_storage_waits(file_db):

    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    q.initialize_history(HistoryStore(2))
    r = record()
    q.put(r["prompt"])
    _, item = q.get()
    connection = sqlite3.connect(file_db, isolation_level=None)
    connection.execute("BEGIN IMMEDIATE")
    worker = threading.Thread(
        target=lambda: q.task_done(
            item, {"outputs": r["outputs"]}, PromptQueue.ExecutionStatus(**r["status"])
        )
    )
    worker.start()
    try:
        time.sleep(0.1)
        assert worker.is_alive()
        running, _ = q.get_current_queue()
        assert running[0][1] == "one"
        assert q.get_history() == {}
    finally:
        connection.rollback()
        connection.close()
        worker.join(10)
    assert not worker.is_alive()
    assert q.get_current_queue()[0] == [] and "one" in q.get_history()


def test_delete_batches_rollback(file_db):

    store = HistoryStore(600)
    for id in ["one", "two"]:
        store.save(id, record(id))
    with sqlite3.connect(file_db) as connection:
        connection.execute(
            "CREATE TRIGGER fail_two BEFORE DELETE ON execution_history WHEN OLD.prompt_id='two' BEGIN SELECT RAISE(ABORT, 'authored'); END"
        )
    with pytest.raises(HistoryStorageError):
        store.remove(["one"] + [f"absent{i}" for i in range(499)] + ["two"])
    assert list(store.load()) == ["one", "two"]


def test_persisted_projection_retained_on_database_failure(file_db, monkeypatch):
    store = HistoryStore(1)

    def fail(session):
        raise OperationalError("authored", {}, Exception("test"))

    monkeypatch.setattr(store, "_prune", fail)
    with pytest.raises(HistoryStorageError) as error:
        store.save("one", record())
    assert "secret" not in json.dumps(error.value.record)


@pytest.mark.parametrize("bad", [object(), float("nan")])
def test_queue_projection_failure_then_success(file_db, bad):
    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    q.initialize_history(HistoryStore(2))
    r = record("bad")
    q.put(r["prompt"])
    _, item = q.get()
    q.task_done(
        item, {"outputs": {"custom": bad}}, PromptQueue.ExecutionStatus(**r["status"])
    )
    assert "bad" in q.history
    assert q.history_persistence_status()["failure_count"] == 1
    complete(q, "good")
    assert "good" in q.history_store.load()
    assert not q.history_persistence_status()["healthy"]


def test_large_integer_timestamp_is_finite():
    r = record()
    r["status"]["messages"][0][1]["timestamp"] = 10**400
    canonical, _ = project_record("one", r)
    assert canonical["status"]["messages"][0][1]["timestamp"] == 10**400


class JsonInt(IntEnum):
    VALUE = 7


class JsonString(str):
    pass


class JsonFloat(float):
    pass


@pytest.mark.parametrize("value", [JsonInt.VALUE, JsonString("text"), JsonFloat(1.25)])
def test_json_scalar_subclass_roundtrip(file_db, value):
    expected = json.loads(json.dumps(value, allow_nan=False))
    r = record()
    r["outputs"]["custom"] = value
    store = HistoryStore(2)
    store.save("one", r)
    assert store.load()["one"]["outputs"]["custom"] == expected


@pytest.mark.parametrize(
    "value", [JsonFloat("nan"), JsonFloat("inf"), JsonFloat("-inf")]
)
def test_nonfinite_scalar_subclass_rejected(file_db, value):
    r = record()
    r["outputs"]["custom"] = value
    store = HistoryStore(2)
    with pytest.raises(HistoryStorageError):
        store.save("one", r)
    assert store.load() == {}


def test_failure_reason_never_logs_exception_payload(file_db, monkeypatch, caplog):
    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    store = HistoryStore(2)
    q.initialize_history(store)
    complete(q, "one")

    def fail(*a, **kw):
        raise HistoryStorageError(
            "PRIVATE exception SQL token", reason="PRIVATE reason"
        ) from ValueError("PRIVATE cause")

    with caplog.at_level(logging.DEBUG), monkeypatch.context() as m:
        m.setattr(store, "save", fail)
        complete(q, "two")
        m.setattr(store, "remove", fail)
        with pytest.raises(HistoryStorageError):
            q.remove_history(clear=True)
    assert "reason=storage_error" in caplog.text
    assert "PRIVATE" not in caplog.text
    assert all(entry.exc_info is None for entry in caplog.records)
    assert q.history_persistence_status()["failure_count"] == 2
    assert "one" in store.load()


def test_actual_projection_reason_logged(file_db, caplog, monkeypatch):
    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    q.initialize_history(HistoryStore(2))
    with caplog.at_level(logging.ERROR):
        with monkeypatch.context() as m:
            m.setattr("app.history.MAX_RECORD_BYTES", 10)
            complete(q, "too-large")
        r = record("unsupported")
        q.put(r["prompt"])
        _, item = q.get()
        q.task_done(item, {"outputs": {"custom": object()}}, None)
    assert "reason=record_too_large" in caplog.text
    assert "reason=unsupported_json" in caplog.text
    assert q.history_persistence_status()["failure_count"] == 2
    complete(q, "good")
    assert "good" in q.history_store.load()


def test_database_reason_omits_sql_and_parameters(file_db, monkeypatch, caplog):
    q = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    store = HistoryStore(2)
    q.initialize_history(store)

    def fail(session):
        raise OperationalError(
            "PRIVATE SQL",
            {"token": "PRIVATE token"},
            sqlite3.OperationalError("database is locked PRIVATE driver"),
        )

    with caplog.at_level(logging.DEBUG), monkeypatch.context() as m:
        m.setattr(store, "_prune", fail)
        complete(q, "one")
    assert "reason=database_locked" in caplog.text
    assert "PRIVATE" not in caplog.text
    assert all(entry.exc_info is None for entry in caplog.records)


def test_history_startup_reason_omits_exception_payload(monkeypatch, caplog):
    import main

    monkeypatch.setattr(main.args, "enable_history", True)
    monkeypatch.setattr(main, "dependencies_available", lambda: True)
    monkeypatch.setattr(main, "get_database_url", lambda: "sqlite:///history.db")

    def fail():
        raise OperationalError(
            "PRIVATE SQL",
            {"token": "PRIVATE token"},
            sqlite3.OperationalError("database is locked PRIVATE driver"),
        )

    monkeypatch.setattr(main, "init_db", fail)
    with caplog.at_level(logging.DEBUG), pytest.raises(SystemExit):
        main.setup_database(SimpleNamespace(enabled=False, startup=lambda: None))
    assert "reason=database_locked" in caplog.text
    assert "PRIVATE" not in caplog.text
    assert all(entry.exc_info is None for entry in caplog.records)
