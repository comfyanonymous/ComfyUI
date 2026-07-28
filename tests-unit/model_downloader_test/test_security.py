"""Unit tests for the security layer: allowlist, SSRF checks, path safety."""

from __future__ import annotations

import pytest

from app.model_downloader.security import allowlist, paths
from app.model_downloader.security.ssrf import (
    SSRFError,
    check_redirect_hop,
    is_blocked_ip,
)


# ----- allowlist -----


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("https://huggingface.co/org/repo/resolve/main/model.safetensors", True),
        ("https://civitai.com/api/download/x/model.safetensors", True),
        ("http://localhost/model.safetensors", True),
        # off-list host
        ("https://evil.example.com/model.safetensors", False),
        # http to a non-loopback allowlisted host is not permitted (https only)
        ("http://huggingface.co/org/repo/resolve/main/model.safetensors", False),
        # bad extension on an allowed host
        ("https://huggingface.co/org/repo/resolve/main/config.json", False),
        # userinfo trick: real host is the metadata IP, not 127.0.0.1
        ("http://127.0.0.1@169.254.169.254/x.safetensors", False),
    ],
)
def test_is_url_allowed(url, allowed):
    assert allowlist.is_url_allowed(url) is allowed


def test_allow_any_extension_relaxes_extension_only():
    url = "https://huggingface.co/org/repo/resolve/main/weights.bin"
    assert allowlist.is_url_allowed(url) is True  # .bin is in the known set
    odd = "https://huggingface.co/org/repo/resolve/main/weights.zip"
    assert allowlist.is_url_allowed(odd) is False
    assert allowlist.is_url_allowed(odd, allow_any_extension=True) is True


@pytest.mark.parametrize(
    "url,downloadable",
    [
        # known model extension in the path -> allowed
        ("https://civitai.com/x/model.safetensors", True),
        # no extension in the path (Civitai download API) -> allowed, resolved later
        ("https://civitai.com/api/download/models/3031464?fileId=2910346", True),
        ("https://civitai.com/api/download/models/3031464", True),
        # explicit non-model extension -> rejected even on an allowed host
        ("https://civitai.com/api/download/models/thing.zip", False),
        ("https://huggingface.co/org/repo/resolve/main/config.json", False),
        # off-list host is never downloadable
        ("https://evil.example.com/api/download/models/1", False),
        # http to a non-loopback allowlisted host is not permitted
        ("http://civitai.com/api/download/models/1", False),
    ],
)
def test_is_url_downloadable(url, downloadable):
    assert allowlist.is_url_downloadable(url) is downloadable


@pytest.mark.parametrize(
    "name,ext",
    [
        ("model.safetensors", ".safetensors"),
        ("model.SAFETENSORS", ".safetensors"),
        ("archive.tar.gz", ".gz"),
        ("noext", ""),
        (".safetensors", ""),  # leading-dot dotfile -> no extension
        ("a/b/c/model.ckpt", ".ckpt"),
    ],
)
def test_filename_extension(name, ext):
    assert allowlist.filename_extension(name) == ext


# ----- SSRF: blocked IPs -----


@pytest.mark.parametrize(
    "ip,blocked",
    [
        ("169.254.169.254", True),  # cloud metadata / link-local
        ("127.0.0.1", True),
        ("10.0.0.5", True),
        ("192.168.1.1", True),
        ("172.16.0.1", True),
        ("::1", True),
        ("0.0.0.0", True),
        # IPv4-mapped IPv6: must see through the mapping even on CPython
        # versions predating the gh-113171 is_* property fix.
        ("::ffff:169.254.169.254", True),  # mapped cloud metadata
        ("::ffff:127.0.0.1", True),  # mapped loopback
        ("::ffff:10.0.0.1", True),  # mapped RFC1918
        ("::ffff:8.8.8.8", False),  # mapped public address stays allowed
        ("8.8.8.8", False),
        ("1.1.1.1", False),
        ("not-an-ip", True),  # unparseable -> refuse
    ],
)
def test_is_blocked_ip(ip, blocked):
    assert is_blocked_ip(ip) is blocked


# ----- SSRF: redirect hop validation -----


def test_check_redirect_hop_rejects_bad_scheme_and_userinfo():
    with pytest.raises(SSRFError):
        check_redirect_hop("ftp://huggingface.co/x.safetensors")
    with pytest.raises(SSRFError):
        check_redirect_hop("https://user:pass@cdn.example.com/x")
    # A CDN host that is NOT on the allowlist is allowed as a redirect target
    # (private-IP protection is the resolver's job; credential leak is prevented
    # by exact host matching).
    assert check_redirect_hop("https://cdn-lfs.huggingface.co/abc") is not None


def test_check_redirect_hop_http_only_for_loopback():
    # Plain http to an external host is rejected (no plaintext downgrade).
    with pytest.raises(SSRFError):
        check_redirect_hop("http://cdn-lfs.huggingface.co/abc")
    # http is honored for loopback only on the initial user-supplied URL (the
    # "download a local model" feature).
    assert (
        check_redirect_hop("http://localhost/x.safetensors", is_initial_url=True)
        is not None
    )
    assert (
        check_redirect_hop("http://127.0.0.1/x.safetensors", is_initial_url=True)
        is not None
    )


def test_check_redirect_hop_blocks_loopback_and_ip_literals_on_redirect():
    # A redirect (is_initial_url=False, the default) must never reach loopback,
    # whether by hostname or by IP literal, nor any other internal IP literal.
    for target in (
        "http://localhost/x.safetensors",
        "http://127.0.0.1/x.safetensors",
        "https://[::1]/x.safetensors",
        "https://169.254.169.254/x.safetensors",  # cloud metadata
        "https://10.0.0.5/x.safetensors",  # RFC1918
    ):
        with pytest.raises(SSRFError):
            check_redirect_hop(target)
    # Off-allowlist public CDN hosts (hostnames) remain valid redirect targets;
    # their resolved IPs are screened by the connector's resolver.
    assert check_redirect_hop("https://cdn-lfs.huggingface.co/abc") is not None


# ----- path safety -----


def test_parse_model_id_valid(model_root):
    directory, filename = paths.parse_model_id("loras/my_lora.safetensors")
    assert directory == "loras"
    assert filename == "my_lora.safetensors"


@pytest.mark.parametrize(
    "model_id",
    [
        "loras/../etc/passwd.safetensors",  # traversal
        "loras/sub/dir.safetensors",  # nested
        "unknownfolder/x.safetensors",  # unknown folder
        "loras/model.txt",  # bad extension
        "noslash.safetensors",  # missing directory
        "loras/",  # empty filename
    ],
)
def test_parse_model_id_rejects(model_root, model_id):
    with pytest.raises(paths.InvalidModelId):
        paths.parse_model_id(model_id)


def test_resolve_destination_stays_in_root(model_root):
    final_path, temp_path = paths.resolve_destination("loras/x.safetensors")
    assert final_path.startswith(model_root)
    assert temp_path.startswith(model_root)
    assert temp_path != final_path


@pytest.mark.parametrize(
    "model_id,ext,expected",
    [
        # no extension -> append the resolved one
        ("loras/my_civitai_model", ".safetensors", "loras/my_civitai_model.safetensors"),
        # different known extension -> replace it
        ("loras/mymodel.ckpt", ".safetensors", "loras/mymodel.safetensors"),
        # same extension -> unchanged
        ("loras/mymodel.safetensors", ".safetensors", "loras/mymodel.safetensors"),
        # non-model suffix is treated as a stem, extension appended
        ("loras/my.model.v2", ".safetensors", "loras/my.model.v2.safetensors"),
        # malformed (no slash) is returned untouched for parse_model_id to reject
        ("noslash", ".safetensors", "noslash"),
    ],
)
def test_apply_extension(model_id, ext, expected):
    assert paths.apply_extension(model_id, ext) == expected


# ----- Content-Disposition filename parsing -----


@pytest.mark.parametrize(
    "header,expected",
    [
        ('attachment; filename="model.safetensors"', "model.safetensors"),
        ("attachment; filename=model.ckpt", "model.ckpt"),
        # RFC 5987 form is preferred and percent-decoded
        (
            "attachment; filename=\"fallback.bin\"; filename*=UTF-8''my%20model.safetensors",
            "my model.safetensors",
        ),
        # directory components in a hostile header are stripped to the basename
        ('attachment; filename="../../etc/passwd"', "passwd"),
        ('attachment; filename="a\\\\b\\\\model.pt"', "model.pt"),
        ("inline", None),
        (None, None),
    ],
)
def test_filename_from_content_disposition(header, expected):
    from app.model_downloader.net.http import filename_from_content_disposition

    assert filename_from_content_disposition(header) == expected
