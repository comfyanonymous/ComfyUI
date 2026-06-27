import base64
from collections.abc import Mapping

from . import codec
from .wire import BytesPayload, MicroValue


PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


def build_micro_payload(value: MicroValue) -> dict:
    payload_bytes = value.payload.as_bytes()
    return {
        "kind": "bytes",
        "compression": "none",
        "data_b64": base64.b64encode(payload_bytes).decode("ascii"),
    }


def parse_micro_payload(d: Mapping) -> BytesPayload:
    if not isinstance(d, Mapping):
        raise ProtocolError("payload must be an object")

    kind = d.get("kind")
    if kind == "bytes":
        return _parse_bytes_payload(d)
    raise ProtocolError(f"unknown payload.kind {kind!r}")


def _parse_bytes_payload(d: Mapping) -> BytesPayload:
    compression = d.get("compression")
    if compression == "none":
        data_b64 = d.get("data_b64")
        if not isinstance(data_b64, str):
            raise ProtocolError("payload.data_b64 must be a string")
        try:
            return BytesPayload(base64.b64decode(data_b64.encode("ascii"), validate=True))
        except Exception as exc:
            raise ProtocolError("payload.data_b64 is not valid base64") from exc
    raise ProtocolError(f"unknown payload.compression {compression!r}")


def build_micro_value(value: MicroValue) -> dict:
    return {
        "_micro": True,
        "type": value.type_name,
        "payload": build_micro_payload(value),
    }


def parse_micro_value(d: Mapping) -> MicroValue:
    if not isinstance(d, Mapping):
        raise ProtocolError("micro value must be an object")
    if d.get("_micro") is not True:
        raise ProtocolError("micro value missing _micro discriminator")

    type_name = d.get("type")
    if not isinstance(type_name, str):
        raise ProtocolError("micro value type must be a string")

    payload = parse_micro_payload(d.get("payload"))
    return MicroValue(type_name, payload)


def materialize_micro_value(value: MicroValue):
    try:
        return codec.decode(value.type_name, value.payload.as_bytes())
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc


def build_request(node_id: str, inputs: dict) -> dict:
    if not isinstance(node_id, str) or not node_id:
        raise ProtocolError("node_id must be a non-empty string")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "node_id": node_id,
        "inputs": {name: _build_value(value) for name, value in inputs.items()},
    }


def parse_request(body: Mapping, *, materialize: bool = False) -> dict:
    _check_protocol_version(body)

    inputs = body.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ProtocolError("inputs must be an object")

    return {name: _parse_value(value, materialize=materialize) for name, value in inputs.items()}


def build_response(output_names, result, output_types=None) -> dict:
    values = _normalize_result(result)
    if len(output_names) != len(values):
        raise ProtocolError("output name count does not match result count")
    if output_types is not None and len(output_types) != len(values):
        raise ProtocolError("output type count does not match result count")

    outputs = {}
    for index, name in enumerate(output_names):
        output_type = None if output_types is None else output_types[index]
        outputs[name] = _build_output_value(values[index], output_type)

    return {"ok": True, "outputs": outputs}


def parse_response(body: Mapping) -> dict:
    if not isinstance(body, Mapping):
        raise ProtocolError("response must be an object")
    if body.get("ok") is not True:
        raise ProtocolError("response is not an ok micro response")

    outputs = body.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ProtocolError("outputs must be an object")

    return {name: _parse_value(value, materialize=False) for name, value in outputs.items()}


def _check_protocol_version(body: Mapping) -> None:
    if not isinstance(body, Mapping):
        raise ProtocolError("request must be an object")
    version = body.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol_version {version}")


def _build_value(value):
    if isinstance(value, MicroValue):
        return build_micro_value(value)
    return value


def _parse_value(value, *, materialize: bool):
    if isinstance(value, Mapping):
        if value.get("_micro") is True:
            micro_value = parse_micro_value(value)
            if materialize:
                return materialize_micro_value(micro_value)
            return micro_value
        raise ProtocolError("objects must use the _micro discriminator")
    return value


def _build_output_value(value, output_type):
    if isinstance(value, MicroValue):
        return build_micro_value(value)
    if output_type is not None and codec.supports_type(output_type):
        return build_micro_value(MicroValue(output_type, BytesPayload(codec.encode(output_type, value))))
    return value


def _normalize_result(result) -> tuple:
    if isinstance(result, dict):
        if "result" in result:
            result = result["result"]
        else:
            raise ProtocolError("node result dict missing result key")
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)
