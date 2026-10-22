"""Opt-in OpenTelemetry traces, metrics and logs, enabled with --enable-opentelemetry.

Everything is exported over OTLP/HTTP and configured with the standard OTEL_* environment
variables, such as OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
When it is off, nothing from opentelemetry is imported and the hooks below do nothing.

Each prompt is one trace: a "prompt" span from the moment it was queued, a "queue" span for
the wait, and one span per executed node named after its class type, with the nodes a node
expands into nested under it. A client can make the prompt part of its own trace by sending
a W3C traceparent header with POST /prompt. Spans and metrics carry ids, node class types,
timings and outcomes, never node inputs or outputs. Log records are exported as ComfyUI
writes them, linked to the node span that was running.
"""

import functools
import logging
import os
import time

from comfy.cli_args import args

enabled = False

_tracer = None
_instruments = {}
_log_handler = None
_prompt = None  # the prompt being executed; the prompt worker runs one at a time

_DURATION_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
_OUTCOMES = {"execution_success": "success", "execution_error": "error", "execution_interrupted": "interrupted"}


def setup(prompt_queue=None):
    if not args.enable_opentelemetry:
        return
    try:
        from opentelemetry import _logs, metrics, trace
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        logging.warning(f"--enable-opentelemetry needs the opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http packages: {e}")
        return

    resource = _resource()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    meter_provider = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))

    # Global providers let custom nodes that use the OpenTelemetry API nest their own spans
    # under the node span that is running them.
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    _logs.set_logger_provider(logger_provider)
    _activate(tracer_provider, meter_provider, logger_provider, prompt_queue)
    logging.info(f"OpenTelemetry export enabled for service {resource.attributes.get('service.name')}, endpoint {os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')}")


def _resource():
    from opentelemetry.sdk.resources import OTELResourceDetector, Resource
    import comfyui_version

    # OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES override these defaults.
    defaults = Resource.create({"service.name": "comfyui", "service.version": comfyui_version.__version__})
    return defaults.merge(OTELResourceDetector().detect())


def _activate(tracer_provider, meter_provider, logger_provider, prompt_queue=None):
    global enabled, _tracer, _instruments, _log_handler
    from opentelemetry.metrics import Observation

    _tracer = tracer_provider.get_tracer("comfyui")
    meter = meter_provider.get_meter("comfyui")
    _instruments = {
        "prompt": meter.create_histogram("comfyui.prompt.duration", unit="s", description="Time to execute a prompt, after it left the queue.", explicit_bucket_boundaries_advisory=_DURATION_BUCKETS),
        "queue": meter.create_histogram("comfyui.prompt.queue.duration", unit="s", description="Time a prompt waited in the queue.", explicit_bucket_boundaries_advisory=_DURATION_BUCKETS),
        "node": meter.create_histogram("comfyui.node.duration", unit="s", description="Time to execute a node.", explicit_bucket_boundaries_advisory=_DURATION_BUCKETS),
        "cached": meter.create_counter("comfyui.node.cached", unit="{node}", description="Nodes whose outputs came from the cache."),
    }

    if prompt_queue is not None:
        def observe_queue(options):
            running, pending = prompt_queue.get_current_queue_volatile()
            return [Observation(len(running), {"comfyui.queue.state": "running"}), Observation(len(pending), {"comfyui.queue.state": "pending"})]
        meter.create_observable_gauge("comfyui.queue.size", [observe_queue], unit="{prompt}", description="Prompts running and waiting in the queue.")

    def observe_memory(options):
        import comfy.model_management
        device = comfy.model_management.get_torch_device()
        attributes = {"comfyui.device": str(device)}
        return [Observation(comfy.model_management.get_free_memory(device), {**attributes, "comfyui.memory.state": "free"}),
                Observation(comfy.model_management.get_total_memory(device), {**attributes, "comfyui.memory.state": "total"})]
    meter.create_observable_gauge("comfyui.device.memory", [observe_memory], unit="By", description="Free and total memory of the torch device.")

    def observe_models(options):
        import comfy.model_management
        return [Observation(len(comfy.model_management.current_loaded_models))]
    meter.create_observable_gauge("comfyui.models.loaded", [observe_models], unit="{model}", description="Models currently loaded on a device.")

    if _log_handler is not None:
        logging.getLogger().removeHandler(_log_handler)
    _log_handler = _LogHandler(logger_provider.get_logger("comfyui"))
    logging.getLogger().addHandler(_log_handler)
    enabled = True


def _deactivate():
    global enabled, _tracer, _instruments, _log_handler
    if _log_handler is not None:
        logging.getLogger().removeHandler(_log_handler)
    enabled, _tracer, _instruments, _log_handler = False, None, {}, None


def _type_name(ex):
    t = type(ex)
    return t.__qualname__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__qualname__}"


class _NoOp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def finish(self, *args):
        pass


_NOOP = _NoOp()


def prompt_span(prompt_id, extra_data):
    """Wraps the execution of one prompt. Call finish() with the executor's status messages."""
    if not enabled:
        return _NOOP
    return _PromptSpan(prompt_id, extra_data)


def node_span(dynprompt, node_id, cached=False):
    """Wraps one execute() call of a node. Call finish() with the result name and the exception, if any."""
    if _prompt is None or cached:
        return _NOOP
    return _NodeSpan(_prompt, dynprompt, node_id)


def traced(name):
    """Decorator that runs a function in a span while telemetry is on."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
            from opentelemetry.trace import StatusCode
            # Like node spans, record only the exception type: messages can contain prompt values.
            with _tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
                try:
                    return func(*args, **kwargs)
                except BaseException as e:
                    span.set_attribute("error.type", _type_name(e))
                    span.set_status(StatusCode.ERROR)
                    raise
        return wrapper
    return decorator


class _PromptSpan:
    def __init__(self, prompt_id, extra_data):
        self.prompt_id = prompt_id
        self.extra_data = extra_data
        self.open_nodes = {}  # node id -> (span, start), for nodes waiting on async work or on the nodes they expanded into
        self.executed = 0
        self.cached = 0
        self.outcome = "error"
        self.error_type = None

    def __enter__(self):
        global _prompt
        from opentelemetry import context, trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        carrier = {k: self.extra_data[k] for k in ("traceparent", "tracestate") if isinstance(self.extra_data.get(k), str)}
        parent = TraceContextTextMapPropagator().extract(carrier, context=context.Context())
        now = time.time_ns()
        queued_at = now
        if isinstance(self.extra_data.get("create_time"), (int, float)):
            queued_at = min(now, int(self.extra_data["create_time"] * 1_000_000))

        self.span = _tracer.start_span("prompt", context=parent, start_time=queued_at, attributes={"comfyui.prompt.id": self.prompt_id})
        _tracer.start_span("queue", context=trace.set_span_in_context(self.span), start_time=queued_at).end(end_time=now)
        _instruments["queue"].record((now - queued_at) / 1e9)
        self.start = time.perf_counter()
        self.token = context.attach(trace.set_span_in_context(self.span))
        _prompt = self
        return self

    def finish(self, status_messages):
        for event, data in status_messages:
            if event == "execution_cached":
                self.cached = len(data.get("nodes", []))
            elif event in _OUTCOMES:
                self.outcome = _OUTCOMES[event]

    def __exit__(self, exc_type, exc, tb):
        global _prompt
        from opentelemetry import context
        from opentelemetry.trace import StatusCode

        _prompt = None
        context.detach(self.token)
        for span, _ in self.open_nodes.values():
            span.end()
        if exc is not None:
            self.outcome = "error"
            self.error_type = _type_name(exc)
        if self.error_type is not None:
            self.span.set_attribute("error.type", self.error_type)
        if self.outcome == "error":
            self.span.set_status(StatusCode.ERROR)
        self.span.set_attributes({
            "comfyui.prompt.outcome": self.outcome,
            "comfyui.prompt.nodes.executed": self.executed,
            "comfyui.prompt.nodes.cached": self.cached,
        })
        self.span.end()
        _instruments["prompt"].record(time.perf_counter() - self.start, {"comfyui.prompt.outcome": self.outcome})
        _instruments["cached"].add(self.cached)
        return False


class _NodeSpan:
    def __init__(self, prompt, dynprompt, node_id):
        self.prompt = prompt
        self.dynprompt = dynprompt
        self.node_id = node_id
        self.result = None
        self.ex = None

    def __enter__(self):
        from opentelemetry import context, trace

        self.class_type = self.dynprompt.get_node(self.node_id)["class_type"]
        if self.node_id in self.prompt.open_nodes:
            self.span, self.start = self.prompt.open_nodes.pop(self.node_id)
        else:
            parent_id = self.dynprompt.get_parent_node_id(self.node_id)
            parent = self.prompt.open_nodes.get(parent_id, (self.prompt.span,))[0]
            attributes = {"comfyui.node.id": self.node_id, "comfyui.node.class_type": self.class_type}
            if parent_id is not None:
                attributes["comfyui.node.parent_id"] = parent_id
            self.span = _tracer.start_span(self.class_type, context=trace.set_span_in_context(parent), attributes=attributes)
            self.start = time.perf_counter()
        self.token = context.attach(trace.set_span_in_context(self.span))
        return self

    def finish(self, result, ex=None):
        self.result = result
        self.ex = ex

    def __exit__(self, exc_type, exc, tb):
        from opentelemetry import context
        from opentelemetry.trace import StatusCode

        context.detach(self.token)
        if exc is None and self.result == "PENDING":
            self.prompt.open_nodes[self.node_id] = (self.span, self.start)
            return False

        error = exc if exc is not None else self.ex
        outcome = "success"
        if error is not None:
            if type(error).__name__ == "InterruptProcessingException":
                outcome = "interrupted"
            else:
                outcome = "error"
                self.prompt.error_type = _type_name(error)
                self.span.set_attribute("error.type", self.prompt.error_type)
                self.span.set_status(StatusCode.ERROR)
        self.span.set_attribute("comfyui.node.outcome", outcome)
        self.span.end()
        self.prompt.executed += 1
        _instruments["node"].record(time.perf_counter() - self.start, {"comfyui.node.class_type": self.class_type, "comfyui.node.outcome": outcome})
        return False


class _LogHandler(logging.Handler):
    """Forwards log records to OpenTelemetry, linked to the span that was active when they were written."""

    def __init__(self, logger):
        super().__init__()
        from opentelemetry import context
        from opentelemetry._logs import SeverityNumber

        self.logger = logger
        self.context = context
        self.severities = [(50, SeverityNumber.FATAL), (40, SeverityNumber.ERROR), (30, SeverityNumber.WARN),
                           (20, SeverityNumber.INFO), (10, SeverityNumber.DEBUG), (0, SeverityNumber.UNSPECIFIED)]

    def emit(self, record):
        # The exporters log through these, so forwarding them would feed back into the export.
        if record.name.startswith(("opentelemetry", "urllib3")):
            return
        try:
            self.logger.emit(
                timestamp=int(record.created * 1e9),
                context=self.context.get_current(),
                severity_number=next(s for level, s in self.severities if record.levelno >= level),
                severity_text=record.levelname,
                body=record.getMessage(),
                attributes={"code.file.path": record.pathname, "code.function.name": record.funcName, "code.line.number": record.lineno},
                exception=record.exc_info[1] if record.exc_info else None,
            )
        except Exception:
            self.handleError(record)
