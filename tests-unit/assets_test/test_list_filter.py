def test_record_list_filters_record_tags(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("filtered.png", ["output", "unit-tests", "chosen"], {}, make_asset_bytes("filtered"))

    response = http.get(f"{api_base}/api/assets", params={"include_tags": "chosen"})

    assert response.status_code == 200
    assert [asset["id"] for asset in response.json()["assets"]] == [record["id"]]


def test_record_list_rejects_metadata_filter(http, api_base):
    response = http.get(
        f"{api_base}/api/assets", params={"metadata_filter": '{"k":"v"}'}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "UNSUPPORTED_PARAM",
        "message": "metadata_filter is no longer supported",
    }
