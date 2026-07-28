import asyncio

from comfy_execution.graph_utils import GraphBuilder


def test_default_prefix_is_isolated_between_async_tasks():
    async def run():
        ready = 0
        both_ready = asyncio.Event()

        async def build(root, call_index, graph_index):
            nonlocal ready
            GraphBuilder.set_default_prefix(root, call_index, graph_index)
            ready += 1
            if ready == 2:
                both_ready.set()
            await both_ready.wait()
            return GraphBuilder().prefix, GraphBuilder().prefix

        return await asyncio.gather(build("first", 10, 20), build("second", 30, 40))

    assert asyncio.run(run()) == [
        ("first.10.20.", "first.10.21."),
        ("second.30.40.", "second.30.41."),
    ]


def test_explicit_prefix_still_advances_default_graph_index():
    try:
        GraphBuilder.set_default_prefix("default", 1, 2)
        assert GraphBuilder.alloc_prefix("explicit", 3, 4) == "explicit.3.4."
        assert GraphBuilder.alloc_prefix() == "default.1.3."
    finally:
        GraphBuilder.set_default_prefix("", 0, 0)
