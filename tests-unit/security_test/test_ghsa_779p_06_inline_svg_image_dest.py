"""CI unit guard for the Sec-Fetch-Dest exemption to FIX #5 of GHSA-779p-m5rp-r4h4.

FIX #5 forced every SVG served by /view and the assets download route to
application/octet-stream + Content-Disposition: attachment. That blocked the
stored XSS, but it also stopped SVG node outputs and Media Assets thumbnails
from rendering, because the frontend requests them with a plain <img>.

An SVG referenced by an <img> is loaded in secure static mode: scripting and
external references are disabled, so the payload from vuln #5 cannot execute.
The attack needs the SVG to load as a document, which arrives with a different
Sec-Fetch-Dest. Browsers set that header themselves and page script cannot
override it, so renders_safely_as_image() uses it to re-allow only the <img>
case. Everything else, including a missing header, keeps the forced download.

server.py cannot be imported in a unit test (importing it spins up the full
PromptServer/aiohttp app and its global side effects), so these tests pin the
helper the /view and assets closures actually call.
"""

import folder_paths


# Every Sec-Fetch-Dest that must keep the forced download. 'document' is the
# vuln #5 attack itself; the rest either cannot execute an SVG or have no
# reason to receive one inline. None covers curl and proxies that strip the
# header.
UNSAFE_DESTS = [
    'document',
    'iframe',
    'object',
    'embed',
    'frame',
    'script',
    'style',
    'empty',
    '',
    None,
]


def test_svg_renders_inline_for_image_dest():
    assert folder_paths.renders_safely_as_image('image/svg+xml', 'image')


def test_svg_forced_to_download_for_every_other_dest():
    for dest in UNSAFE_DESTS:
        assert not folder_paths.renders_safely_as_image('image/svg+xml', dest), (
            f"SVG must not be served inline for Sec-Fetch-Dest={dest!r}. Only "
            "an <img> load is safe, everything else can reach document context"
        )


def test_missing_header_fails_closed():
    assert not folder_paths.renders_safely_as_image('image/svg+xml', None)


def test_exemption_normalises_parameters_and_casing():
    for content_type in ('IMAGE/SVG+XML', 'image/svg+xml; charset=utf-8', '  image/svg+xml  '):
        assert folder_paths.renders_safely_as_image(content_type, 'image'), (
            f"{content_type!r} is an SVG and must not be excluded from the "
            "exemption by casing or a charset parameter"
        )


def test_other_dangerous_types_are_never_exempt():
    # Only SVG is safe as an <img>. Nothing else in the blocklist may ride the
    # image dest back into an inline response.
    for content_type in ('text/html', 'text/javascript', 'text/css',
                         'application/xml', 'text/xml', 'application/xhtml+xml',
                         'image/svg+xml.html', 'application/rss+xml'):
        assert not folder_paths.renders_safely_as_image(content_type, 'image'), (
            f"{content_type!r} must stay forced-download regardless of Sec-Fetch-Dest"
        )


def test_exemption_does_not_widen_the_blocklist():
    # The exemption is a call-site gate, not a change to what counts as
    # dangerous. is_dangerous_content_type must still flag SVG on its own.
    assert folder_paths.is_dangerous_content_type('image/svg+xml')
