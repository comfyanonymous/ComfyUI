def test_upload_returns_a_record_identifier(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("upload.png", ["output", "unit-tests"], {}, make_asset_bytes("upload"))

    assert record["id"]
    assert http.get(f"{api_base}/api/assets/{record['id']}").status_code == 200
