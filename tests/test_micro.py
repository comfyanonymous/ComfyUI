import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import nodes  # noqa: E402
from comfy.micro import codec, envelope, server as micro_server  # noqa: E402
from comfy.micro.wire import BytesPayload, MicroValue  # noqa: E402


def _load_micro_nodes():
    module_path = Path(__file__).resolve().parents[1] / "comfy_extras" / "nodes_micro.py"
    spec = importlib.util.spec_from_file_location("nodes_micro_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    nodes.NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    return module


def _load_run_demo():
    module_path = Path(__file__).with_name("run_demo.py")
    spec = importlib.util.spec_from_file_location("micro_run_demo_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MICRO_NODES = _load_micro_nodes()


class DummyRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def _json_response_body(response):
    return json.loads(response.text)


def _micro_image(value: torch.Tensor) -> MicroValue:
    return MicroValue("IMAGE", BytesPayload(codec.encode_image(value)))


def test_image_codec_round_trips_bit_identically():
    image = torch.rand((1, 64, 64, 3), dtype=torch.float32).to(torch.float16)

    decoded = codec.decode_image(codec.encode_image(image))

    assert torch.equal(decoded, image)


def test_mask_codec_round_trips_bit_identically():
    mask = torch.rand((1, 64, 64), dtype=torch.float32).to(torch.float16)

    decoded = codec.decode_mask(codec.encode_mask(mask))

    assert torch.equal(decoded, mask)


def test_codec_unknown_type_raises():
    with pytest.raises(ValueError):
        codec.encode("LATENT", torch.zeros((1,)))


def test_envelope_build_parse_request_round_trip():
    value = MicroValue("IMAGE", BytesPayload(b"abc"))
    request = envelope.build_request("Micro_ScaleImage", {"image": value, "width": 32})
    parsed = envelope.parse_request(json.loads(json.dumps(request)))

    assert parsed["image"].type_name == "IMAGE"
    assert parsed["image"].payload.as_bytes() == b"abc"
    assert parsed["width"] == 32


def test_envelope_unknown_payload_kind_raises_protocol_error():
    body = {
        "protocol_version": envelope.PROTOCOL_VERSION,
        "node_id": "Micro_ScaleImage",
        "inputs": {
            "image": {
                "_micro": True,
                "type": "IMAGE",
                "payload": {"kind": "shmem"},
            }
        },
    }

    with pytest.raises(envelope.ProtocolError):
        envelope.parse_request(body)


def test_envelope_unknown_payload_compression_raises_protocol_error():
    body = envelope.build_request("Micro_ScaleImage", {"image": MicroValue("IMAGE", BytesPayload(b"abc"))})
    body["inputs"]["image"]["payload"]["compression"] = "zstd"

    with pytest.raises(envelope.ProtocolError):
        envelope.parse_request(body)


def test_envelope_unsupported_protocol_version_raises_protocol_error():
    body = {
        "protocol_version": 999,
        "node_id": "Micro_ScaleImage",
        "inputs": {},
    }

    with pytest.raises(envelope.ProtocolError):
        envelope.parse_request(body)


def test_micro_execute_happy_path_matches_image_scale():
    image = torch.linspace(0, 1, 1 * 7 * 5 * 3, dtype=torch.float32).reshape((1, 7, 5, 3))
    body = envelope.build_request("Micro_ScaleImage", {
        "image": _micro_image(image),
        "upscale_method": "nearest-exact",
        "width": 10,
        "height": 14,
        "crop": "disabled",
    })

    response = asyncio.run(micro_server.micro_execute(DummyRequest(body)))
    response_body = _json_response_body(response)
    parsed = envelope.parse_response(response_body)
    decoded = codec.decode_image(parsed["image"].payload.as_bytes())
    expected = nodes.ImageScale().upscale(image, "nearest-exact", 10, 14, "disabled")[0]

    assert response_body["ok"] is True
    assert torch.equal(decoded, expected)


def test_micro_execute_unknown_node_returns_error():
    body = {
        "protocol_version": envelope.PROTOCOL_VERSION,
        "node_id": "Micro_NonexistentNode",
        "inputs": {},
    }

    response = asyncio.run(micro_server.micro_execute(DummyRequest(body)))
    response_body = _json_response_body(response)

    assert response_body["ok"] is False
    assert response_body["error"]["kind"] == "unknown_node"


def test_micro_execute_node_exception_returns_error():
    image = torch.rand((1, 4, 4, 3), dtype=torch.float32)
    body = envelope.build_request("Micro_ScaleImage", {
        "image": _micro_image(image),
        "upscale_method": "not-a-method",
        "width": 8,
        "height": 8,
        "crop": "disabled",
    })

    response = asyncio.run(micro_server.micro_execute(DummyRequest(body)))
    response_body = _json_response_body(response)

    assert response_body["ok"] is False
    assert response_body["error"]["kind"] == "node_error"
    assert response_body["error"]["traceback"]


def test_single_instance_demo_workflow(tmp_path):
    run_demo = _load_run_demo()

    digest, output = run_demo.run_single(tmp_path)

    assert len(output) > 0
    assert len(digest) == 64
