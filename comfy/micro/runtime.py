import json
import logging
import urllib.error
import urllib.request

from comfy.cli_args import args

from . import envelope


class MicroExecutionError(RuntimeError):
    pass


def call_micro_node(node_id: str, inputs: dict) -> dict:
    """Invoke a registered Micro node body via the configured worker."""

    request_body = envelope.build_request(node_id, inputs)
    logging.info("[micro] POST %s node_id=%s", args.micro_worker_url, node_id)
    response_body = _post_json(args.micro_worker_url, request_body)

    if response_body.get("ok") is False:
        error = response_body.get("error", {})
        kind = error.get("kind", "micro_error")
        message = error.get("message", "micro worker returned an error")
        raise MicroExecutionError(f"{kind}: {message}")

    return envelope.parse_response(response_body)


def _post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise MicroExecutionError(f"micro worker HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise MicroExecutionError(f"micro worker request failed: {exc}") from exc
