import logging
import traceback

from aiohttp import web

from . import envelope


_MICRO_TO_NATIVE = {}
_REGISTERED_ROUTE_TABLES = set()


def register_micro_nodes(mapping: dict[str, str]) -> None:
    _MICRO_TO_NATIVE.update(mapping)


def register_routes(prompt_server=None) -> bool:
    if prompt_server is None:
        import server

        prompt_server = getattr(server.PromptServer, "instance", None)
        if prompt_server is None:
            return False

    route_table_id = id(prompt_server.routes)
    if route_table_id in _REGISTERED_ROUTE_TABLES:
        return False

    prompt_server.routes.post("/micro/execute")(micro_execute)
    _REGISTERED_ROUTE_TABLES.add(route_table_id)
    logging.info("[micro] registered POST /micro/execute")
    return True


async def micro_execute(request):
    try:
        body = await request.json()
    except Exception as exc:
        return _error_response("protocol_error", f"invalid JSON body: {exc}")

    try:
        inputs = envelope.parse_request(body, materialize=True)
        node_id = _parse_node_id(body)
    except envelope.ProtocolError as exc:
        return _error_response("protocol_error", str(exc))

    logging.info("[micro] handling /micro/execute node_id=%s", node_id)
    node_cls = _get_node_class(node_id)
    if node_cls is None:
        return _error_response("unknown_node", f"no registered node for {node_id}")

    try:
        instance = node_cls()
        result = getattr(instance, node_cls.FUNCTION)(**inputs)
        response = _build_success_response(node_id, node_cls, result)
        return web.json_response(response)
    except Exception as exc:
        return _error_response("node_error", str(exc), traceback.format_exc())


def _parse_node_id(body) -> str:
    node_id = body.get("node_id")
    if not isinstance(node_id, str) or not node_id:
        raise envelope.ProtocolError("node_id must be a non-empty string")
    return node_id


def _get_node_class(node_id: str):
    import nodes

    native_node_id = _MICRO_TO_NATIVE.get(node_id, node_id)
    return nodes.NODE_CLASS_MAPPINGS.get(native_node_id)


def _build_success_response(node_id: str, node_cls, result) -> dict:
    import nodes

    wire_node_cls = nodes.NODE_CLASS_MAPPINGS.get(node_id)
    output_names = getattr(wire_node_cls, "RETURN_NAMES", None) if wire_node_cls is not None else None
    if output_names is None:
        output_names = getattr(node_cls, "RETURN_NAMES", node_cls.RETURN_TYPES)

    return envelope.build_response(output_names, result, output_types=node_cls.RETURN_TYPES)


def _error_response(kind: str, message: str, tb: str = ""):
    return web.json_response({
        "ok": False,
        "error": {
            "kind": kind,
            "message": message,
            "traceback": tb,
        },
    })
