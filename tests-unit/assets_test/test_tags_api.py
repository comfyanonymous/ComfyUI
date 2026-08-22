def test_tag_listing_includes_record_tag(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("tagged.png", ["output", "unit-tests", "visible"], {}, make_asset_bytes("tagged"))

    response = http.get(f"{api_base}/api/assets", params={"include_tags": "visible"})

    assert response.status_code == 200
    assert [asset["id"] for asset in response.json()["assets"]] == [record["id"]]
