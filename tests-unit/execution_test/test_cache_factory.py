import pytest


class _Server:
    def __init__(self) -> None:
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.sockets_metadata: dict[str, dict[str, object]] = {}

    def send_sync(self, event: str | int, data: object, sid: str | None = None) -> None:
        pass

    def queue_updated(self) -> None:
        pass


@pytest.fixture
def execution_module(monkeypatch):
    try:
        from comfy.cli_args import args
        monkeypatch.setattr(args, "cpu", True, raising=False)
        import execution
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"execution module could not be imported in CPU mode: {exc!r}")

    return execution


def test_cache_factory_supplies_caches_on_construction(execution_module) -> None:
    sentinel = execution_module.CacheSet(cache_type=execution_module.CacheType.NONE)

    executor = execution_module.PromptExecutor(_Server(), cache_factory=lambda: sentinel)

    assert executor.caches is sentinel


def test_default_cache_factory_uses_cache_type_and_args(execution_module) -> None:
    executor = execution_module.PromptExecutor(
        _Server(), cache_type=execution_module.CacheType.LRU, cache_args={"lru": 7}
    )

    assert isinstance(executor.caches, execution_module.CacheSet)
    assert executor.caches.outputs.max_size == 7


def test_cache_factory_exceptions_propagate_from_construction(execution_module) -> None:
    def raise_boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        execution_module.PromptExecutor(_Server(), cache_factory=raise_boom)
