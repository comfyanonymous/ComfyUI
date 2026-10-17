"""Contract test for app/agent_host.py against a fake v1 host."""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import signal
import threading
import time
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import WSMsgType, web

import folder_paths
from app import agent_host
from comfy_api import feature_flags


class FakeHost:
    def __init__(self, upstream: str = ""):
        self.state = {"state": "starting", "url": None, "reason": None, "restarts": 0, "max_restarts": 3}
        self.upstream = upstream
        self.started = None
        self.on_change = None
        self.ready = True
        self.stopped = 0

    def start(self, *, comfy_url, comfy_path, data_dir, logger, on_change=None):
        self.started = {"comfy_url": comfy_url, "data_dir": data_dir, "logger": logger}
        self.on_change = on_change

    def stop(self, timeout=10.0):
        self.stopped += 1

    def status(self):
        return dict(self.state)

    def route(self, path):
        if not self.ready and path.startswith("/api/agent/"):
            return None, {}
        if path.startswith("/api/agent/"):
            return self.upstream + path[len("/api"):], {"X-Comfy-Agent-Session": "tok"}
        return None


installed_on = []
# The real one, before the fixture below stubs it out for every test.
_install_termination_handlers = agent_host._handle_termination_signals


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(folder_paths, "get_system_user_directory", lambda name="system": str(tmp_path / f"__{name}"))
    monkeypatch.setattr(agent_host, "_logged_proxy_errors", set())
    monkeypatch.setattr(agent_host, "_host", None)
    monkeypatch.setattr(agent_host, "_failure", None)
    installed_on.clear()
    monkeypatch.setattr(agent_host, "_handle_termination_signals", installed_on.append)


def _entry_points(monkeypatch, eps: dict, installed: bool = False):
    agent = SimpleNamespace(name="comfy-agent")
    points = [SimpleNamespace(name=name, load=lambda f=factory: f, dist=agent) for name, factory in eps.items()]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: points if group == "comfyui.agent" else [])

    def distribution(name):
        if installed:
            return object()
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


def _raise(*args, **kwargs):
    raise RuntimeError("broken install")


def _exit(*args, **kwargs):
    raise SystemExit("wrong-architecture wheel")


def _agent_records(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[agent-event]")]


@pytest.mark.parametrize(
    "eps, installed, loaded, record",
    [
        ({"v1": FakeHost}, True, True, None),
        ({}, False, False, "[agent-event] package_missing"),
        ({}, True, False, "[agent-event] agent_error reason=not_found"),  # an install from before the entry point
        ({"v9": FakeHost}, False, False, "[agent-event] agent_error reason=not_found"),
        ({"v1": _raise}, True, False, "[agent-event] agent_error reason=unknown"),
        ({"v1": _exit}, True, False, "[agent-event] agent_error reason=unknown"),
    ],
)
def test_load_finds_v1_or_reports_why_not(monkeypatch, caplog, eps, installed, loaded, record):
    _entry_points(monkeypatch, eps, installed)
    with caplog.at_level(logging.INFO):
        assert agent_host.load() is loaded
    assert _agent_records(caplog) == ([record] if record else [])


def test_broken_entry_point_metadata_does_not_stop_comfyui(monkeypatch, caplog):
    monkeypatch.setattr(importlib.metadata, "entry_points", _raise)
    with caplog.at_level(logging.INFO):
        assert agent_host.load() is False
    assert _agent_records(caplog) == ["[agent-event] agent_error reason=unknown"]


def test_status_reaches_the_flag_and_the_websocket(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    assert agent_host.feature_flag() is None
    agent_host.load()
    sent = []
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append((event, data))), "http", "0.0.0.0", 8188)
    # Records must leave through ComfyUI's own logger, and the agent's data (its
    # session token) must stay in a system user dir the userdata API refuses.
    assert agent_host._host.started == {"comfy_url": "http://127.0.0.1:8188", "data_dir": folder_paths.get_system_user_directory("agent"), "logger": logging.getLogger()}
    assert agent_host.feature_flag() == {"state": "starting", "url": None, "reason": None}
    agent_host._host.state = {"state": "ready", "url": None, "reason": None}
    agent_host._host.on_change()
    assert sent == [("agent_status", {"state": "ready", "url": "/api/agent", "reason": None})]
    assert installed_on == [None], "start() hands ComfyUI's loop to the termination handlers"
    assert feature_flags.get_server_features()["agent"] == sent[0][1]


@pytest.mark.parametrize(
    ("url", "published"),
    [
        ("http://127.0.0.1:5555", "http://127.0.0.1:5555"),  # direct: the agent serves the page itself
        ("/agent", "/api/agent"),  # anything else: ComfyUI proxies
        (None, "/api/agent"),
    ],
)
def test_an_absolute_status_url_is_published_as_is(monkeypatch, url, published):
    monkeypatch.setattr(agent_host, "_host", FakeHost())
    agent_host._host.state = {"state": "ready", "url": url, "reason": None}
    assert agent_host.feature_flag() == {"state": "ready", "url": published, "reason": None}
    agent_host._host.state = {"state": "starting", "url": url, "reason": None}
    assert agent_host.feature_flag()["url"] is None


@pytest.mark.parametrize(("eps", "installed", "reason"), [({}, False, "not_found"), ({}, True, "not_found"), ({"v1": _raise}, True, "unknown")])
def test_a_package_that_does_not_load_publishes_failed(monkeypatch, eps, installed, reason):
    _entry_points(monkeypatch, eps, installed)
    agent_host.load()
    assert feature_flags.get_server_features()["agent"] == {"state": "failed", "url": None, "reason": reason}


def test_status_changes_the_flag_does_not_carry_send_nothing(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    sent = []
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append(data)), "http", "127.0.0.1", 8188)
    agent_host._host.on_change()
    agent_host._host.state = {**agent_host._host.state, "restarts": 1}
    agent_host._host.on_change()  # only the restart count moved
    assert sent == [{"state": "starting", "url": None, "reason": None}]


def test_a_raising_status_is_logged_once(monkeypatch, caplog):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    monkeypatch.setattr(agent_host._host, "status", _raise)
    for _ in range(3):
        assert agent_host.feature_flag()["state"] == "failed"
    assert sum("status could not be read" in r.getMessage() for r in caplog.records) == 1




@pytest.mark.parametrize("error", [SystemExit])
def test_any_exception_from_the_package_fails_the_agent_not_comfyui(monkeypatch, error):
    def create():
        raise error()

    _entry_points(monkeypatch, {"v1": create})
    assert agent_host.load() is False


def test_a_factory_that_returns_no_host_fails_the_agent(monkeypatch):
    _entry_points(monkeypatch, {"v1": object})
    assert agent_host.load() is False
    assert agent_host.feature_flag() == {"state": "failed", "url": None, "reason": "unknown"}










def test_a_start_that_exits_fails_the_agent_not_comfyui(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()

    def start(**kwargs):
        raise SystemExit(3)  # what a corrupt or wrong-architecture wheel does

    monkeypatch.setattr(agent_host._host, "start", start)
    sent = []
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append(data)), "http", "127.0.0.1", 8188)
    assert sent == [{"state": "failed", "url": None, "reason": "spawn_failed"}]








@pytest.mark.parametrize(("address", "url"), [("0.0.0.0", "http://127.0.0.1:8188"), ("", "http://127.0.0.1:8188"), ("::", "http://[::1]:8188"), ("::1", "http://[::1]:8188"), ("192.168.1.5", "http://192.168.1.5:8188")])
def test_comfyui_tells_the_agent_a_url_it_can_reach(monkeypatch, address, url):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: None), "http", address, 8188)
    assert agent_host._host.started["comfy_url"] == url






@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX permissions")
def test_an_unwritable_user_dir_fails_only_the_claim(monkeypatch):
    base = folder_paths.get_system_user_directory("agent")
    os.makedirs(base)
    os.chmod(base, 0o500)
    try:
        assert agent_host._claim_data_dir(8189) == base + "-8189"
    finally:
        os.chmod(base, 0o700)








async def _until(predicate):
    while not predicate():
        await asyncio.sleep(0.02)








@pytest.mark.asyncio
async def test_nothing_is_proxied_after_the_agent_failed_to_start(proxied, monkeypatch):
    monkeypatch.setattr(agent_host, "_failure", {"state": "failed", "url": None, "reason": "spawn_failed"})
    resp = await proxied.get("/api/agent/threads")
    assert (resp.status, resp.headers.get("Retry-After"), await resp.text()) == (503, None, "the local agent is not available: spawn_failed")
    assert await (await proxied.get("/api/prompt")).json() == {"comfy": True}


@pytest.mark.asyncio
async def test_a_load_failure_answers_503_but_not_to_other_machines(proxied, monkeypatch):
    monkeypatch.setattr(agent_host, "_failure", {"state": "failed", "url": None, "reason": "not_found"})
    assert (await proxied.get("/api/agent/threads")).status == 503
    assert (await proxied.get("/api/agent/threads", headers={"X-Test-Remote": "192.168.1.5"})).status == 403


@pytest.mark.asyncio
async def test_an_agent_that_failed_for_good_is_a_503_without_retry(proxied):
    agent_host._host.state = {"state": "failed", "url": None, "reason": "crashed", "restarts": 3, "max_restarts": 3}
    resp = await proxied.get("/api/agent/threads")
    assert (resp.status, resp.headers.get("Retry-After")) == (503, None)
    assert "crashed" in await resp.text()






def test_a_host_that_fails_to_start_reports_failed(monkeypatch, caplog):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    monkeypatch.setattr(agent_host._host, "start", _raise)
    sent = []
    with caplog.at_level(logging.INFO):
        agent_host.start(SimpleNamespace(loop=None, send_sync=lambda event, data: sent.append((event, data))), "http", "127.0.0.1", 8188)
    failed = {"state": "failed", "url": None, "reason": "spawn_failed"}
    assert agent_host.feature_flag() == failed
    assert sent == [("agent_status", failed)]
    assert _agent_records(caplog) == ["[agent-event] agent_error reason=spawn_failed"]


@pytest_asyncio.fixture
async def proxied(monkeypatch, aiohttp_server, aiohttp_client):
    async def echo(request):
        body = await request.text()
        return web.json_response({"method": request.method, "path": request.path_qs, "session": request.headers.get("X-Comfy-Agent-Session"), "origin": request.headers.get("Origin"), "body": body}, status=201 if body else 200)

    async def socket(request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        async for msg in ws:
            if msg.data == "bye":
                await ws.close(code=4001)
            elif msg.data == "crash":
                request.transport.close()
            else:
                await ws.send_str("echo:" + msg.data)
        return ws

    async def cut(request):  # an agent that dies mid-response
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"partial")
        request.transport.close()
        return response

    upstream_app = web.Application()
    upstream_app.router.add_get("/agent/events", socket)
    upstream_app.router.add_get("/agent/cut", cut)
    upstream_app.router.add_route("*", "/agent/{tail:.*}", echo)
    upstream = await aiohttp_server(upstream_app)
    monkeypatch.setattr(agent_host, "_host", FakeHost(f"http://127.0.0.1:{upstream.port}"))

    async def prompt(request):
        return web.json_response({"comfy": True})

    @web.middleware
    async def peer(request, handler):  # lets a test stand in for a client on the network
        remote = request.headers.get("X-Test-Remote")
        return await handler(request.clone(remote=remote) if remote else request)

    app = web.Application(middlewares=[peer, agent_host.proxy_middleware])
    app.router.add_get("/api/prompt", prompt)
    return await aiohttp_client(app)


@pytest.mark.asyncio
async def test_proxy_forwards_claimed_paths_with_the_session(proxied):
    resp = await proxied.get("/api/agent/threads?limit=5", headers={"Origin": f"http://{proxied.host}:{proxied.port}"})
    assert await resp.json() == {"method": "GET", "path": "/agent/threads?limit=5", "session": "tok", "origin": None, "body": ""}
    resp = await proxied.post("/api/agent/threads", data="hello", headers={"Origin": f"http://{proxied.host}:{proxied.port}"})
    assert (resp.status, (await resp.json())["body"]) == (201, "hello")
    assert await (await proxied.get("/api/prompt")).json() == {"comfy": True}

    big = "x" * (6 * 1024 * 1024)  # above aiohttp's 4 MiB default
    async with proxied.ws_connect("/api/agent/events", max_msg_size=0, origin=f"http://{proxied.host}:{proxied.port}") as ws:
        await ws.send_str(big)
        msg = await ws.receive(timeout=10)
        assert (msg.type, msg.data) == (WSMsgType.TEXT, "echo:" + big)  # both directions, above 4 MiB
        await ws.send_str("bye")  # the agent closing reaches the page, with its own code
        assert (await ws.receive(timeout=10)).type in (WSMsgType.CLOSE, WSMsgType.CLOSED)
        assert ws.close_code == 4001

    async with proxied.ws_connect("/api/agent/events", origin=f"http://{proxied.host}:{proxied.port}") as ws:
        await ws.send_str("crash")
        await ws.receive(timeout=10)
        assert ws.close_code == 1011, "an agent that dies without a close frame is an error, not a normal close"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "attacker.example"},  # a site that rebinds its name to 127.0.0.1
        {"Origin": "http://attacker.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"X-Test-Remote": "192.168.1.5"},  # a client on the network, with a loopback Host
        {"X-Forwarded-For": "203.0.113.9"},  # a reverse proxy that says so
    ],
)
async def test_proxy_refuses_other_sites(proxied, headers):
    resp = await proxied.get("/api/agent/threads", headers=headers)
    assert resp.status == 403


@pytest.mark.asyncio
async def test_another_local_port_is_another_site(proxied):
    # e.g. a dev server on this machine: same host, different port, so not ComfyUI's own page
    other = f"http://{proxied.host}:{proxied.port + 1}"
    assert (await proxied.post("/api/agent/threads", data="x", headers={"Origin": other})).status == 403


@pytest.mark.asyncio
async def test_writes_and_sockets_need_comfyuis_own_page(proxied):
    # A same-host reverse proxy can present a loopback peer and Host without saying so.
    assert (await proxied.post("/api/agent/threads", data="x")).status == 403
    resp = await proxied.get("/api/agent/events", headers={"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="})
    assert resp.status == 403
    assert (await proxied.get("/api/agent/threads")).status == 200  # reads without an Origin still work


@pytest.mark.asyncio
async def test_proxied_sockets_are_capped(proxied, monkeypatch):
    monkeypatch.setattr(agent_host, "MAX_PROXIED_WEBSOCKETS", 1)
    origin = f"http://{proxied.host}:{proxied.port}"
    async with proxied.ws_connect("/api/agent/events", origin=origin):
        with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
            await proxied.ws_connect("/api/agent/events", origin=origin)
        assert refused.value.status == 503
    for _ in range(100):  # the proxy frees the slot once its relay winds down
        if agent_host._open_websockets == 0:
            break
        await asyncio.sleep(0.02)
    async with proxied.ws_connect("/api/agent/events", origin=origin):
        pass


@pytest.mark.asyncio
async def test_proxied_requests_are_capped(proxied, monkeypatch):
    monkeypatch.setattr(agent_host, "_open_requests", agent_host.MAX_PROXIED_REQUESTS)
    resp = await proxied.get("/api/agent/threads")
    assert (resp.status, resp.headers.get("Retry-After")) == (503, "1")




@pytest.mark.asyncio
async def test_a_broken_route_leaves_comfyui_serving(proxied, monkeypatch, caplog):
    monkeypatch.setattr(agent_host._host, "route", _raise)
    for path in ("/api/prompt", "/api/prompt?a=1"):  # a different path per request, as a browser makes them
        resp = await proxied.get(path)
        assert (resp.status, await resp.json()) == (200, {"comfy": True}), "ComfyUI keeps serving"
    assert sum("could not route" in r.getMessage() for r in caplog.records) == 1, "logged once per error, not per path"


@pytest.mark.asyncio
async def test_an_agent_that_dies_mid_response_ends_it(proxied, caplog):
    resp = await proxied.get("/api/agent/cut")
    assert resp.status == 200
    with pytest.raises(Exception):
        await resp.read()
    assert any("was cut off" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_restarting_agent_is_a_503(proxied):
    agent_host._host.ready = False
    agent_host._host.state = {**agent_host._host.state, "reason": "crashed"}
    resp = await proxied.get("/api/agent/threads")
    assert (resp.status, resp.headers.get("Retry-After")) == (503, "2")




@pytest.mark.asyncio
async def test_a_host_that_claims_no_paths_is_never_proxied(proxied, monkeypatch):
    monkeypatch.setattr(agent_host._host, "route", lambda path: None)  # direct mode
    agent_host._host.state = {"state": "ready", "url": "http://127.0.0.1:5555", "reason": None}
    assert (await proxied.get("/api/agent/threads")).status == 404  # ComfyUI's own routes answer
    assert await (await proxied.get("/api/prompt")).json() == {"comfy": True}


@pytest.mark.asyncio
async def test_an_unreachable_agent_is_a_502(proxied, monkeypatch):
    monkeypatch.setattr(agent_host._host, "upstream", "http://127.0.0.1:1")
    assert (await proxied.get("/api/agent/threads")).status == 502


def test_only_this_machine_reaches_the_agent():
    assert [agent_host._is_loopback(r) for r in ("127.0.0.1", "::1", "localhost", "192.168.1.5", None)] == [True, True, True, False, False]


def test_termination_stops_the_agent_off_the_signal_path_then_re_delivers(monkeypatch):
    calls = []
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(loop, "remove_signal_handler", lambda signum: calls.append(("default", signum)))
    monkeypatch.setattr(agent_host, "stop", lambda: calls.append(("stop", threading.current_thread() is threading.main_thread())))
    monkeypatch.setattr(agent_host.os, "kill", lambda pid, signum: calls.append(("kill", signum)))
    agent_host._on_termination(loop, signal.SIGTERM, signal.SIG_DFL)
    loop.run_until_complete(asyncio.wait_for(_until(lambda: any(c[0] == "kill" for c in calls)), 5))
    loop.close()
    assert calls == [("default", signal.SIGTERM), ("stop", False), ("kill", signal.SIGTERM)]


def test_termination_chains_to_a_python_handler_after_stopping_the_agent(monkeypatch):
    calls = []
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(loop, "remove_signal_handler", lambda signum: None)
    monkeypatch.setattr(agent_host.signal, "signal", lambda signum, handler: None)
    monkeypatch.setattr(agent_host, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(agent_host.os, "kill", lambda pid, signum: calls.append(("default action", signum)))
    agent_host._on_termination(loop, signal.SIGTERM, lambda signum, frame: calls.append(("theirs", signum)))
    loop.run_until_complete(asyncio.wait_for(_until(lambda: len(calls) == 2), 5))
    loop.close()
    assert calls == ["stop", ("theirs", signal.SIGTERM)]


def test_termination_handlers_skip_an_ignored_signal(monkeypatch):
    installed = []
    loop = SimpleNamespace(add_signal_handler=lambda signum, *args: installed.append(signum))
    monkeypatch.setattr(agent_host.signal, "getsignal", lambda signum: signal.SIG_IGN if signum == signal.SIGHUP else (lambda *a: None))
    _install_termination_handlers(loop)
    assert installed == [signal.SIGTERM], "a Python handler is wrapped; an ignored signal is left alone"


async def _until(predicate):
    while not predicate():
        await asyncio.sleep(0.02)


def _lock_elsewhere(base):
    """Holds owner.lock as another ComfyUI would (a separate open file description)."""
    import fcntl

    os.makedirs(base, exist_ok=True)
    fd = os.open(os.path.join(base, "owner.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


@pytest.mark.skipif(os.name == "nt", reason="flock")
def test_the_first_comfyui_keeps_the_shared_dir_until_it_stops(caplog):
    base = folder_paths.get_system_user_directory("agent")
    assert agent_host._claim_data_dir(8188) == base
    other = os.open(os.path.join(base, "owner.lock"), os.O_RDWR)
    import fcntl

    with pytest.raises(OSError):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # held while this ComfyUI runs
    agent_host._release_data_dir()
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free once it stops
    os.close(other)


@pytest.mark.skipif(os.name == "nt", reason="flock")
def test_a_second_comfyui_gets_a_sibling_dir_without_waiting(caplog):
    base = folder_paths.get_system_user_directory("agent")
    holder = _lock_elsewhere(base)
    try:
        began = time.monotonic()
        assert agent_host._claim_data_dir(8189) == base + "-8189"
        assert time.monotonic() - began < 0.5
        held = [r for r in caplog.records if "holds" in r.getMessage()]
        assert [(r.levelname, r.exc_info) for r in held] == [("WARNING", None)]
    finally:
        os.close(holder)


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
@pytest.mark.parametrize("odd", ["directory", "dangling link"])
def test_whatever_is_at_owner_lock_costs_only_the_shared_dir(odd):
    base = folder_paths.get_system_user_directory("agent")
    os.makedirs(base)
    outside = os.path.join(os.path.dirname(base), "created-outside")
    if odd == "directory":
        os.mkdir(os.path.join(base, "owner.lock"))
    else:
        os.symlink(outside, os.path.join(base, "owner.lock"))
    assert agent_host._claim_data_dir(8188) == base + "-8188"
    assert not os.path.exists(outside)


@pytest.mark.skipif(os.name == "nt", reason="flock")
def test_the_claim_waits_off_the_event_loop(monkeypatch):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    claimed = threading.Event()
    monkeypatch.setattr(agent_host, "_claim_data_dir", lambda port: claimed.wait(5) and "dir")
    loop = asyncio.new_event_loop()
    try:
        began = time.monotonic()
        agent_host.start(SimpleNamespace(loop=loop, send_sync=lambda event, data: None), "http", "127.0.0.1", 8188)
        assert time.monotonic() - began < 0.5, "start() doesn't run the claim on the event loop"
        claimed.set()
        loop.run_until_complete(asyncio.wait_for(_until(lambda: agent_host._host.started), 5))
    finally:
        loop.close()


@pytest.mark.parametrize("raises", [None, RuntimeError, SystemExit])
def test_stop_stops_the_host_and_frees_the_shared_dir(monkeypatch, raises):
    _entry_points(monkeypatch, {"v1": FakeHost})
    agent_host.load()
    base = folder_paths.get_system_user_directory("agent")
    assert agent_host._claim_data_dir(8188) == base
    if raises:
        def stop(timeout=10.0):
            agent_host._host.stopped += 1
            raise raises()

        monkeypatch.setattr(agent_host._host, "stop", stop)
    agent_host.stop()
    assert agent_host._host.stopped == 1
    assert agent_host._claim_data_dir(8189) == base, "the next ComfyUI gets the shared dir"
    agent_host._release_data_dir()

