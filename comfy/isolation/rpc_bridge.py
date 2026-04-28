import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


class RpcBridge:
    """Minimal helper to run coroutines synchronously inside isolated processes.

    If an event loop is already running, the coroutine is executed on a fresh
    thread with its own loop to avoid nested run_until_complete errors.
    """

    def run_sync(self, maybe_coro):
        if not asyncio.iscoroutine(maybe_coro):
            return maybe_coro

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result_container = {}
            exc_container = {}

            def _runner():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result_container["value"] = new_loop.run_until_complete(maybe_coro)
                except Exception as exc:  # pragma: no cover
                    exc_container["error"] = exc
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join()

            if "error" in exc_container:
                raise exc_container["error"]
            return result_container.get("value")

        return asyncio.run(maybe_coro)
