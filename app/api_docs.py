"""Browsable API documentation for the ComfyUI HTTP API.

Renders the repository's ``openapi.yaml`` with Redoc. Redoc is used rather than
Swagger UI because it has no request-execution feature at all: the local server
is unauthenticated by default, so a docs page that could fire requests would
give one-click access to destructive endpoints such as ``/api/interrupt``,
``/api/free`` and ``DELETE /api/userdata/{file}``.

The viewer bundle is loaded from a CDN, so the page needs outbound network
access to render. Installs without it still get the raw spec from
``/openapi.yaml``; the fallback notice below points there.
"""

import os

from aiohttp import web

# openapi.yaml lives next to server.py at the repository root. Resolve it from
# __file__ rather than the cwd: ComfyUI is routinely launched from other
# directories and through wrappers, and a relative path would 404 unpredictably.
SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "openapi.yaml"
)

# Pinned to an exact version rather than a floating tag so a CDN-side release
# cannot change what this page executes.
#
# TODO: add an integrity="sha384-..." attribute. The pinned path is immutable,
# so SRI is worth having; the hash simply could not be computed where this was
# written (no outbound network), and a wrong hash fails the page closed.
REDOC_BUNDLE_URL = "https://cdn.jsdelivr.net/npm/redoc@2.5.0/bundles/redoc.standalone.js"

# The spec URL is deliberately relative. add_routes() re-registers every route
# under an /api prefix, so this page is reachable at both /api-docs and
# /api/api-docs; a relative URL resolves to the sibling spec in either case.
SPEC_URL = "openapi.yaml"

# Substituted with str.replace rather than str.format/f-string so the CSS braces
# below stay literal and a future style edit does not have to double them.
API_DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ComfyUI API Reference</title>
    <style>
      body { margin: 0; padding: 0; font-family: system-ui, sans-serif; }
      #fallback {
        display: none;
        margin: 3rem auto;
        max-width: 40rem;
        padding: 0 1.5rem;
        line-height: 1.6;
        color: #333;
      }
      #fallback code {
        background: #f2f2f2;
        border-radius: 3px;
        padding: 0.1em 0.35em;
      }
    </style>
  </head>
  <body>
    <redoc spec-url="__SPEC_URL__"></redoc>
    <div id="fallback">
      <h1>API docs viewer unavailable</h1>
      <p>
        The documentation viewer is loaded from a CDN and could not be reached.
        This is expected on an offline or air-gapped install.
      </p>
      <p>
        The specification itself is served locally and needs no network access:
        <a href="__SPEC_URL__">openapi.yaml</a>. Render it with any local viewer,
        for example <code>npx @redocly/cli preview-docs openapi.yaml</code>.
      </p>
    </div>
    <script
      src="__BUNDLE_URL__"
      onerror="document.getElementById('fallback').style.display='block';"
    ></script>
  </body>
</html>
""".replace("__SPEC_URL__", SPEC_URL).replace("__BUNDLE_URL__", REDOC_BUNDLE_URL)


def add_api_docs_routes(routes: web.RouteTableDef) -> None:
    """Register the spec and docs-page routes on the given route table.

    Registering on PromptServer's route table (rather than on the app directly)
    matters twice over: the table is added before the ``web.static('/')``
    catch-all that would otherwise shadow these paths, and every route in it is
    also re-registered under an ``/api`` prefix.
    """

    @routes.get("/openapi.yaml")
    async def get_openapi_spec(request):
        if not os.path.isfile(SPEC_PATH):
            return web.Response(status=404, text="openapi.yaml not found")
        response = web.FileResponse(SPEC_PATH)
        response.headers["Content-Type"] = "application/yaml"
        # The cache_control middleware only special-cases js/css/images, so a
        # .yaml response falls through untouched. Without this, a user editing
        # the spec would keep getting a stale copy from the browser cache.
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    @routes.get("/api-docs")
    async def get_api_docs(request):
        response = web.Response(text=API_DOCS_HTML, content_type="text/html")
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response
