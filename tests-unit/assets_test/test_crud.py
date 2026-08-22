def test_record_crud_hard_deletes_the_record(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("crud.png", ["output", "unit-tests"], {}, make_asset_bytes("crud"))

    assert http.get(f"{api_base}/api/assets/{record['id']}").status_code == 200
    assert http.delete(f"{api_base}/api/assets/{record['id']}").status_code == 204
    assert http.get(f"{api_base}/api/assets/{record['id']}").status_code == 404
