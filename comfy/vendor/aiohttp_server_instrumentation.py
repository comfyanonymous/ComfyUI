# Copyright 2020, OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import urllib
from timeit import default_timer
from typing import Dict, List, Tuple, Union

from aiohttp import web
from multidict import CIMultiDictProxy
from opentelemetry import context, metrics, trace
from opentelemetry.context import _SUPPRESS_HTTP_INSTRUMENTATION_KEY
_instruments = ("aiohttp ~= 3.0",)
__version__ = "0.46b0.dev"
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import http_status_to_status_code
from opentelemetry.propagate import extract
from opentelemetry.propagators.textmap import Getter
from opentelemetry.semconv.metrics import MetricInstruments
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import get_excluded_urls, remove_url_credentials

_duration_attrs = [
    SpanAttributes.HTTP_METHOD,
    SpanAttributes.HTTP_HOST,
    SpanAttributes.HTTP_SCHEME,
    SpanAttributes.HTTP_STATUS_CODE,
    SpanAttributes.HTTP_FLAVOR,
    SpanAttributes.HTTP_SERVER_NAME,
    SpanAttributes.NET_HOST_NAME,
    SpanAttributes.NET_HOST_PORT,
    SpanAttributes.HTTP_ROUTE,
]

_active_requests_count_attrs = [
    SpanAttributes.HTTP_METHOD,
    SpanAttributes.HTTP_HOST,
    SpanAttributes.HTTP_SCHEME,
    SpanAttributes.HTTP_FLAVOR,
    SpanAttributes.HTTP_SERVER_NAME,
]

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__, __version__)
_excluded_urls = get_excluded_urls("AIOHTTP_SERVER")


def _parse_duration_attrs(req_attrs):
    duration_attrs = {}
    for attr_key in _duration_attrs:
        if req_attrs.get(attr_key) is not None:
            duration_attrs[attr_key] = req_attrs[attr_key]
    return duration_attrs


def _parse_active_request_count_attrs(req_attrs):
    active_requests_count_attrs = {}
    for attr_key in _active_requests_count_attrs:
        if req_attrs.get(attr_key) is not None:
            active_requests_count_attrs[attr_key] = req_attrs[attr_key]
    return active_requests_count_attrs


def get_default_span_details(request: web.Request) -> Tuple[str, dict]:
    """Default implementation for get_default_span_details
    Args:
        request: the request object itself.
    Returns:
        a tuple of the span name, and any attributes to attach to the span.
    """
    span_name = request.path.strip() or f"HTTP {request.method}"
    return span_name, {}


def _get_view_func(request: web.Request) -> str:
    """Returns the name of the request handler.
    Args:
        request: the request object itself.
    Returns:
        a string containing the name of the handler function
    """
    try:
        return request.match_info.handler.__name__
    except AttributeError:
        return "unknown"


def collect_request_attributes(request: web.Request) -> Dict:
    """Collects HTTP request attributes from the ASGI scope and returns a
    dictionary to be used as span creation attributes."""

    server_host, port, http_url = (
        request.url.host,
        request.url.port,
        str(request.url),
    )
    query_string = request.query_string
    if query_string and http_url:
        if isinstance(query_string, bytes):
            query_string = query_string.decode("utf8")
        http_url += "?" + urllib.parse.unquote(query_string)

    result = {
        SpanAttributes.HTTP_SCHEME: request.scheme,
        SpanAttributes.HTTP_HOST: server_host,
        SpanAttributes.NET_HOST_PORT: port,
        SpanAttributes.HTTP_ROUTE: _get_view_func(request),
        SpanAttributes.HTTP_FLAVOR: f"{request.version.major}.{request.version.minor}",
        SpanAttributes.HTTP_TARGET: request.path,
        SpanAttributes.HTTP_URL: remove_url_credentials(http_url),
    }

    http_method = request.method
    if http_method:
        result[SpanAttributes.HTTP_METHOD] = http_method

    http_host_value_list = (
        [request.host] if not isinstance(request.host, list) else request.host
    )
    if http_host_value_list:
        result[SpanAttributes.HTTP_SERVER_NAME] = ",".join(
            http_host_value_list
        )
    http_user_agent = request.headers.get("user-agent")
    if http_user_agent:
        result[SpanAttributes.HTTP_USER_AGENT] = http_user_agent

    # remove None values
    result = {k: v for k, v in result.items() if v is not None}

    return result


def set_status_code(span, status_code: int) -> None:
    """Adds HTTP response attributes to span using the status_code argument."""

    try:
        status_code = int(status_code)
    except ValueError:
        span.set_status(
            Status(
                StatusCode.ERROR,
                "Non-integer HTTP status: " + repr(status_code),
            )
        )
    else:
        span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, status_code)
        span.set_status(
            Status(http_status_to_status_code(status_code, server_span=True))
        )


class AiohttpGetter(Getter):
    """Extract current trace from headers"""

    def get(self, carrier, key: str) -> Union[List, None]:
        """Getter implementation to retrieve an HTTP header value from the ASGI
        scope.

        Args:
            carrier: ASGI scope object
            key: header name in scope
        Returns:
            A list of all header values matching the key, or None if the key
            does not match any header.
        """
        headers: CIMultiDictProxy = carrier.headers
        if not headers:
            return None
        return headers.getall(key, None)

    def keys(self, carrier: Dict) -> List:
        return list(carrier.keys())


getter = AiohttpGetter()


@web.middleware
async def middleware(request, handler):
    """Middleware for aiohttp implementing tracing logic"""
    if (
            context.get_value("suppress_instrumentation")
            or context.get_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY)
            or _excluded_urls.url_disabled(request.url.path)
    ):
        return await handler(request)

    span_name, additional_attributes = get_default_span_details(request)

    req_attrs = collect_request_attributes(request)
    duration_attrs = _parse_duration_attrs(req_attrs)
    active_requests_count_attrs = _parse_active_request_count_attrs(req_attrs)

    duration_histogram = meter.create_histogram(
        name=MetricInstruments.HTTP_SERVER_DURATION,
        unit="ms",
        description="Duration of HTTP client requests.",
    )

    active_requests_counter = meter.create_up_down_counter(
        name=MetricInstruments.HTTP_SERVER_ACTIVE_REQUESTS,
        unit="requests",
        description="measures the number of concurrent HTTP requests those are currently in flight",
    )

    with tracer.start_as_current_span(
            span_name,
            context=extract(request, getter=getter),
            kind=trace.SpanKind.SERVER,
    ) as span:
        attributes = collect_request_attributes(request)
        attributes.update(additional_attributes)
        span.set_attributes(attributes)
        start = default_timer()
        active_requests_counter.add(1, active_requests_count_attrs)
        try:
            resp = await handler(request)
            set_status_code(span, resp.status)
        except web.HTTPException as ex:
            set_status_code(span, ex.status_code)
            raise
        finally:
            duration = max((default_timer() - start) * 1000, 0)
            duration_histogram.record(duration, duration_attrs)
            active_requests_counter.add(-1, active_requests_count_attrs)
        return resp


class _InstrumentedApplication(web.Application):
    """Insert tracing middleware"""

    def __init__(self, *args, **kwargs):
        middlewares = kwargs.pop("middlewares", [])
        middlewares.insert(0, middleware)
        kwargs["middlewares"] = middlewares
        super().__init__(*args, **kwargs)


class AioHttpServerInstrumentor(BaseInstrumentor):
    # pylint: disable=protected-access,attribute-defined-outside-init
    """An instrumentor for aiohttp.web.Application

    See `BaseInstrumentor`
    """

    def _instrument(self, **kwargs):
        self._original_app = web.Application
        setattr(web, "Application", _InstrumentedApplication)

    def _uninstrument(self, **kwargs):
        setattr(web, "Application", self._original_app)

    def instrumentation_dependencies(self):
        return _instruments
