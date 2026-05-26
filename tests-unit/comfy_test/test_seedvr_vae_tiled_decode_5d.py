from copy import deepcopy

def _valid_probe_payload():
    sha = "0" * 64
    return {
        "torch_equal": True,
        "non_tiled_sha256": sha,
        "tiled_sha256": sha,
        "dtype": "torch.float16",
        "source_frames": 32,
        "temporal_tile_size": 16,
        "temporal_overlap": 4,
        "generic_fallback_used": False,
    }


def _assert_real_probe_json_contract(payload):
    required = {
        "torch_equal",
        "non_tiled_sha256",
        "tiled_sha256",
        "dtype",
        "source_frames",
        "temporal_tile_size",
        "temporal_overlap",
        "generic_fallback_used",
    }
    missing = required.difference(payload)
    if missing:
        raise AssertionError(f"missing keys: {sorted(missing)}")
    if payload["torch_equal"] is not True:
        raise AssertionError("torch_equal must be true")
    if payload["non_tiled_sha256"] != payload["tiled_sha256"]:
        raise AssertionError("tensor sha256 values must match")
    if payload["dtype"] != "torch.float16":
        raise AssertionError("dtype must be torch.float16")
    if payload["source_frames"] != 32:
        raise AssertionError("source_frames must be 32")
    if payload["temporal_tile_size"] != 16:
        raise AssertionError("temporal_tile_size must be 16")
    if payload["temporal_overlap"] != 4:
        raise AssertionError("temporal_overlap must be 4")
    if payload["generic_fallback_used"] is not False:
        raise AssertionError("generic_fallback_used must be false")


def test_real_probe_json_contract():
    valid = _valid_probe_payload()
    _assert_real_probe_json_contract(valid)

    for key in valid:
        missing = deepcopy(valid)
        missing.pop(key)
        try:
            _assert_real_probe_json_contract(missing)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted payload missing {key}")

    invalid_values = {
        "torch_equal": False,
        "tiled_sha256": "1" * 64,
        "dtype": "torch.float32",
        "source_frames": 31,
        "temporal_tile_size": 8,
        "temporal_overlap": 0,
        "generic_fallback_used": True,
    }
    for key, value in invalid_values.items():
        invalid = deepcopy(valid)
        invalid[key] = value
        try:
            _assert_real_probe_json_contract(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted payload with invalid {key}")
