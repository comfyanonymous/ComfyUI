from aiohttp.multipart import parse_content_disposition

from api_server.utils.http_headers import content_disposition_for_file


def test_content_disposition_for_file_adds_inline_type_by_default():
    header = content_disposition_for_file("vace14b_00003.mp4")

    disposition_type, params = parse_content_disposition(header)

    assert header == 'inline; filename="vace14b_00003.mp4"'
    assert disposition_type == "inline"
    assert params == {"filename": "vace14b_00003.mp4"}


def test_content_disposition_for_file_supports_attachment():
    header = content_disposition_for_file("download me.png", "attachment")

    disposition_type, params = parse_content_disposition(header)

    assert disposition_type == "attachment"
    assert params == {"filename": "download me.png"}


def test_content_disposition_for_file_adds_utf8_filename_parameter():
    header = content_disposition_for_file("八月のスーベニア.MP3")

    disposition_type, params = parse_content_disposition(header)

    assert disposition_type == "inline"
    assert params["filename"] == "download.MP3"
    assert params["filename*"] == "八月のスーベニア.MP3"
