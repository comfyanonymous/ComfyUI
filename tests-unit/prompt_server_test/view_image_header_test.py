from email.message import Message

from utils.http_headers import content_disposition_header


def test_view_content_disposition_is_valid_inline_filename():
    header = content_disposition_header("example image.png", "inline")

    message = Message()
    message["Content-Disposition"] = header

    assert message.get_content_disposition() == "inline"
    assert message.get_filename() == "example image.png"
    assert "filename*=UTF-8''example%20image.png" in header


def test_view_content_disposition_sanitizes_control_characters():
    header = content_disposition_header('bad"\r\nname.png', "inline")

    message = Message()
    message["Content-Disposition"] = header

    assert message.get_content_disposition() == "inline"
    assert message.get_filename() == "bad__name.png"


def test_view_content_disposition_adds_utf8_filename_parameter():
    header = content_disposition_header("café.png", "inline")

    assert 'filename="caf_.png"' in header
    assert "filename*=UTF-8''caf%C3%A9.png" in header
