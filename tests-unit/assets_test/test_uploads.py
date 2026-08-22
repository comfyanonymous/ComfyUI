import pytest


@pytest.mark.hashing_on
def test_hash_on_upload_dedup_returns_same_entity(asset_factory, make_asset_bytes):
    data = make_asset_bytes("hash-on-dedup")

    first = asset_factory("dedup.png", ["output", "unit-tests"], {}, data)
    second = asset_factory("dedup.png", ["output", "unit-tests"], {}, data)

    assert second["id"] == first["id"]
    assert second["created_new"] is False


def test_upload_returns_a_record_identifier(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("upload.png", ["output", "unit-tests"], {}, make_asset_bytes("upload"))

    assert record["id"]
    assert http.get(f"{api_base}/api/assets/{record['id']}").status_code == 200
