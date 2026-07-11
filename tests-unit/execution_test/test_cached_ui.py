from types import SimpleNamespace

from execution import _send_cached_ui


class FakeServer:
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.sent = []

    def send_sync(self, event, data, client_id):
        self.sent.append((event, data, client_id))


def test_cached_ui_populates_outputs_without_client_id():
    cached_ui = {
        "meta": {"node_id": "1"},
        "output": {"images": [{"filename": "cached.png"}]},
    }
    cached = SimpleNamespace(ui=cached_ui)
    server = FakeServer(client_id=None)
    ui_outputs = {}

    _send_cached_ui(server, "1", "1", cached, "prompt-id", ui_outputs)

    assert ui_outputs == {"1": cached_ui}
    assert server.sent == []


def test_cached_ui_sends_executed_event_with_client_id():
    cached_ui = {
        "meta": {"node_id": "1"},
        "output": {"images": [{"filename": "cached.png"}]},
    }
    cached = SimpleNamespace(ui=cached_ui)
    server = FakeServer(client_id="client-id")
    ui_outputs = {}

    _send_cached_ui(server, "1", "display-1", cached, "prompt-id", ui_outputs)

    assert ui_outputs == {"1": cached_ui}
    assert server.sent == [
        (
            "executed",
            {
                "node": "1",
                "display_node": "display-1",
                "output": cached_ui["output"],
                "prompt_id": "prompt-id",
            },
            "client-id",
        )
    ]
