"""Loads the local agent (the optional comfy-agent package) and fronts it.

ComfyUI owns the v1 contract below; the package provides an object that meets
it through the `comfyui.agent` entry point named "v1", and owns everything else
(spawning, readiness, restarts, its output and `[agent-event]` lifecycle
records, where it listens). ComfyUI holds the one instance, publishes its status
in the feature flags plus an `agent_status` websocket message on every change,
and forwards the paths the agent claims (`route()`) so the browser reaches it on
ComfyUI's own origin.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import ipaddress
import logging
import os
import signal
import urllib.parse
from typing import Callable, Literal, Protocol, TypedDict

import aiohttp
from aiohttp import web

import folder_paths
from comfy_api import feature_flags

ENTRY_POINT_GROUP = "comfyui.agent"
PROTOCOL = "v1"
# What the frontend calls; the agent serves it behind ComfyUI's origin.
FRONTEND_URL = "/api/agent"
# Above the agent's own frame limit (12 MiB for a document catch-up).
WS_MAX_MSG_SIZE = 16 * 1024 * 1024
# Each proxied socket costs two of ComfyUI's file descriptors.
MAX_PROXIED_WEBSOCKETS = 64
MAX_PROXIED_REQUESTS = 256
_FORWARDED_HEADERS = ("Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Real-IP")


class AgentStatus(TypedDict):
    state: Literal["starting", "ready", "failed"]
    url: str | None
    reason: str | None


class AgentHostV1(Protocol):
    def start(self, *, comfy_url: str, comfy_path: str, data_dir: str, logger: logging.Logger, on_change: Callable[[], None] | None = None) -> None: ...
    def stop(self, timeout: float = ...) -> None: ...
    # status() and route() run on ComfyUI's event loop and must not block.
    def status(self) -> AgentStatus: ...
    def route(self, path: str) -> tuple[str | None, dict[str, str]] | None: ...


_host: AgentHostV1 | None = None
# Set when loading or starting the host failed, so the flag says "failed"
# rather than leaving the agent looking switched off.
_failure: AgentStatus | None = None
_open_websockets = 0
_owner_lock: int | None = None  # held while this ComfyUI owns user/__agent
_open_requests = 0
_logged_proxy_errors: set[str] = set()


def emit_event(event: str, /, **fields) -> None:
    logging.info(" ".join([f"[agent-event] {event}", *(f"{k}={v}" for k, v in fields.items() if v is not None)]))


def load() -> bool:
    """Find and create the agent host. Logs and reports why when it can't;
    nothing about the installed package can stop ComfyUI from starting."""
    global _host, _failure
    feature_flags.register_runtime_flag("agent", feature_flag)
    try:
        entry_points = {ep.name: ep for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
        if PROTOCOL not in entry_points:
            if entry_points or _distribution_installed():
                logging.warning(f"The installed comfy-agent offers agent protocols {sorted(entry_points)}; this ComfyUI needs {PROTOCOL}. Update comfy-agent to use the agent.")
                emit_event("agent_error", reason="not_found")
            else:
                emit_event("package_missing")
                logging.warning(f"`--enable-agent` was passed but the comfy-agent package is not installed, so the agent is disabled. Its requirements are in {os.path.join(os.path.dirname(os.path.abspath(folder_paths.__file__)), 'agent_requirements.txt')}.")
            _failure = {"state": "failed", "url": None, "reason": "not_found"}
            return False
        host = entry_points[PROTOCOL].load()()
        if not all(callable(getattr(host, name, None)) for name in ("start", "stop", "status", "route")):
            raise TypeError(f"the {PROTOCOL} entry point returned {type(host).__name__}, not an agent host")
        _host = host
    except (Exception, SystemExit):  # a broken wheel may exit; that must not end ComfyUI's startup
        logging.exception("Could not load the local agent; it is disabled.")
        emit_event("agent_error", reason="unknown")
        _failure = {"state": "failed", "url": None, "reason": "unknown"}
        return False
    return True


def _distribution_installed() -> bool:
    try:
        importlib.metadata.distribution("comfy-agent")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def start(server, scheme: str, address: str, port: int) -> None:
    """Start the loaded host once the server is listening. Main thread only."""
    if _host is None:
        return
    host = "127.0.0.1" if address in ("", "0.0.0.0") else "::1" if address == "::" else address
    comfy_url = f"{scheme}://[{host}]:{port}" if ":" in host else f"{scheme}://{host}:{port}"

    last_sent = None

    def on_change():
        nonlocal last_sent
        flag = feature_flag()
        if flag != last_sent:  # the host also reports changes the flag doesn't carry
            last_sent = flag
            server.send_sync("agent_status", flag)

    _handle_termination_signals(server.loop)

    def run():
        global _failure
        data_dir = _claim_data_dir(port)
        try:
            _host.start(comfy_url=comfy_url, comfy_path=os.path.dirname(os.path.abspath(folder_paths.__file__)), data_dir=data_dir, logger=logging.getLogger(), on_change=on_change)
        except (Exception, SystemExit):
            logging.exception("The local agent could not be started.")
            emit_event("agent_error", reason="spawn_failed")
            _failure = {"state": "failed", "url": None, "reason": "spawn_failed"}
            on_change()

    # The data-dir claim may wait up to 2 s on owner.lock, so it and start() run off the event loop.
    if server.loop is not None:
        server.loop.run_in_executor(None, run)
    else:
        run()


def _handle_termination_signals(loop) -> None:
    """Stop the agent on SIGTERM/SIGHUP, then let the signal end ComfyUI as before.

    Python's default action ends the process without its shutdown path, and
    nothing else stops the agent. The work runs on the event loop and a worker
    thread, never inside a signal handler; a Python handler already installed is
    called afterwards. POSIX only: Desktop tree-kills ComfyUI on Windows.
    """
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        previous = signal.getsignal(signum) if signum is not None else None
        if previous is not signal.SIG_DFL and not callable(previous):
            continue  # ignored, or installed outside Python
        try:
            loop.add_signal_handler(signum, _on_termination, loop, signum, previous)
        except (NotImplementedError, RuntimeError, ValueError):
            pass


def _on_termination(loop, signum, previous) -> None:
    loop.remove_signal_handler(signum)  # back to the default action, so a second signal ends ComfyUI at once
    logging.info("Stopping the local agent before exiting.")

    def then(_):
        if callable(previous):
            signal.signal(signum, previous)
            previous(signum, None)
        else:
            os.kill(os.getpid(), signum)

    loop.run_in_executor(None, stop).add_done_callback(then)


def stop() -> None:
    if _host is not None:
        try:
            _host.stop()
        except (Exception, SystemExit):
            logging.exception("The local agent could not be stopped.")
    _release_data_dir()


def _claim_data_dir(port: int) -> str:
    """The agent's data dir: user/__agent for the first ComfyUI on this install,
    so its history stays put, and user/__agent-<port>, a sibling, for a second
    ComfyUI running at the same time, so neither reaps the other's agent.

    The first one holds an OS lock on owner.lock for as long as it runs; the OS
    drops it when the process ends however it ends, so there is no stale claim
    to clean up. Anything odd at that path only costs the agent the shared dir."""
    global _owner_lock
    base = folder_paths.get_system_user_directory("agent")
    try:
        os.makedirs(base, exist_ok=True)
        fd = os.open(os.path.join(base, "owner.lock"), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            logging.warning(f"Another ComfyUI on this install holds {base}; this one's agent uses {base}-{port}.")
            return f"{base}-{port}"
        _owner_lock = fd
        return base
    except Exception:
        logging.exception("The local agent's data dir could not be claimed; using a separate one.")
    return f"{base}-{port}"


def _release_data_dir() -> None:
    global _owner_lock
    if _owner_lock is not None:
        with contextlib.suppress(OSError):
            os.close(_owner_lock)
        _owner_lock = None


def feature_flag() -> AgentStatus | None:
    """The `agent` feature flag: the host's state, with the URL the frontend uses."""
    if _failure is not None or _host is None:
        return _failure
    try:
        status = _host.status()
        state = status["state"]
        # An absolute URL means the agent serves the page itself (direct mode,
        # so it handles CORS and its own credential); otherwise ComfyUI proxies it.
        url = status.get("url") or ""
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            url = FRONTEND_URL
        return {"state": state, "url": url if state == "ready" else None, "reason": status.get("reason")}
    except Exception as e:
        _log_once(f"status:{type(e).__name__}:{e}", "The local agent's status could not be read.", e)
        return {"state": "failed", "url": None, "reason": "unknown"}


@web.middleware
async def proxy_middleware(request: web.Request, handler):
    """Forward a request the agent claims to it, HTTP or WebSocket."""
    global _open_websockets, _open_requests
    target = None
    if _failure is None and _host is not None:
        try:
            target = _host.route(request.path_qs)
        except Exception as e:
            _log_once(f"route:{type(e).__name__}:{e}", f"The local agent could not route {request.path}.", e)
    # When the package didn't load or start, ComfyUI still answers for the agent's own prefix.
    claimed = target is not None or (_failure is not None and (request.path == FRONTEND_URL or request.path.startswith(FRONTEND_URL + "/")))
    if not claimed:
        return await handler(request)
    # The session added here is the agent's own credential: only this machine,
    # by a loopback name, and no other site's page may use it.
    if not _from_this_machine(request):
        return web.Response(status=403, text="the local agent is only reachable from ComfyUI's own page on this machine")
    url, headers = target or (None, {})
    flag = feature_flag()
    if flag is None or flag["state"] == "failed":  # for good: retrying won't help
        return web.Response(status=503, text=f"the local agent is not available: {(flag or {}).get('reason')}")
    if url is None:  # starting or restarting
        return web.Response(status=503, headers={"Retry-After": "2"}, text="the local agent is not ready")
    # Origin was checked above; the agent would refuse an https one.
    forward = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS and k.lower() != "origin"}
    forward.update(headers)
    if request.headers.get("Upgrade", "").lower() == "websocket":
        if _open_websockets >= MAX_PROXIED_WEBSOCKETS:
            return web.Response(status=503, headers={"Retry-After": "5"}, text="too many open connections to the local agent")
        _open_websockets += 1  # reserved before the handshake, so concurrent ones can't overshoot
        try:
            return await _proxy_websocket(request, url, forward)
        finally:
            _open_websockets -= 1
    if _open_requests >= MAX_PROXIED_REQUESTS:
        return web.Response(status=503, headers={"Retry-After": "1"}, text="too many open requests to the local agent")
    _open_requests += 1
    try:
        return await _proxy_http(request, url, forward)
    finally:
        _open_requests -= 1


async def _proxy_http(request: web.Request, url: str, forward: dict[str, str]):
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        try:
            upstream = await session.request(request.method, url, headers=forward, data=request.content if request.body_exists else None, allow_redirects=False)
        except aiohttp.ClientError as e:
            return web.Response(status=502, text=f"the local agent is not reachable: {e}")
        async with upstream:
            response = web.StreamResponse(status=upstream.status, headers={k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS})
            await response.prepare(request)
            try:
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
            except ConnectionResetError:  # the page went away: nothing to report
                pass
            except aiohttp.ClientError as e:
                logging.warning(f"The local agent's response to {request.path} was cut off: {e}")
                if request.transport is not None:
                    request.transport.close()  # so the page sees a broken response, not a short one
            return response


_HOP_HEADERS = frozenset({"host", "connection", "keep-alive", "upgrade", "transfer-encoding", "content-length", "te", "trailer", "proxy-authorization", "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions", "sec-websocket-accept"})


async def _proxy_websocket(request: web.Request, url: str, headers: dict[str, str]):
    async with aiohttp.ClientSession() as session:
        try:
            upstream = await session.ws_connect(url, headers=headers, max_msg_size=WS_MAX_MSG_SIZE)
        except aiohttp.ClientError as e:
            return web.Response(status=502, text=f"the local agent is not reachable: {e}")
        async with upstream:
            return await _relay_websocket(request, upstream)


async def _relay_websocket(request: web.Request, upstream):
    client = web.WebSocketResponse(max_msg_size=WS_MAX_MSG_SIZE)
    await client.prepare(request)

    async def pump(source, sink, close_code):
        async for msg in source:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await sink.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await sink.send_bytes(msg.data)
            else:
                break
        await sink.close(code=close_code())

    def agent_close_code():
        # The agent's own close code, or 1011 if it went away without one (a crash).
        return upstream.close_code if upstream.close_code not in (None, 1006) else 1011

    await asyncio.gather(pump(client, upstream, lambda: 1000), pump(upstream, client, agent_close_code), return_exceptions=True)
    return client


def _log_once(key: str, message: str, error: BaseException | None = None) -> None:
    """Log a proxy problem once per kind (key excludes the request path), not on every request."""
    if key not in _logged_proxy_errors and len(_logged_proxy_errors) < 100:
        _logged_proxy_errors.add(key)
        logging.error(message, exc_info=error)


def _from_this_machine(request: web.Request) -> bool:
    """A loopback peer, not behind a proxy that says so, addressing ComfyUI by a
    loopback name (which a site that rebinds its own name to 127.0.0.1 can't),
    from no page or ComfyUI's own; and from ComfyUI's own page for anything that
    changes state or opens a socket, since a same-host reverse proxy can satisfy
    the rest without saying so."""
    if not _is_loopback(request.remote) or not _is_loopback(urllib.parse.urlsplit("//" + request.host).hostname):
        return False
    if request.headers.get("Sec-Fetch-Site") == "cross-site" or any(h in request.headers for h in _FORWARDED_HEADERS):
        return False
    origin = request.headers.get("Origin")
    same_origin = origin is not None and urllib.parse.urlsplit(origin).netloc.lower() == request.host.lower()
    if request.method not in ("GET", "HEAD", "OPTIONS") or request.headers.get("Upgrade", "").lower() == "websocket":
        return same_origin
    return origin is None or same_origin


def _is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
