import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import comfy.model_management
import comfy.telemetry as telemetry
import nodes
from comfy_execution.graph_utils import GraphBuilder
from execution import PromptExecutor

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PARENT_SPAN_ID = "b7ad6b7169203331"


class Constant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value,)


class Increment(Constant):
    def execute(self, value):
        return (value + 1,)


class AsyncIncrement(Constant):
    async def execute(self, value):
        await asyncio.sleep(0.05)
        return (value + 1,)


class ExpandIncrement(Constant):
    def execute(self, value):
        graph = GraphBuilder()
        increment = graph.node("TelemetryIncrement", "increment", value=value)
        return {"result": (increment.out(0),), "expand": graph.finalize()}


class Log(Constant):
    def execute(self, value):
        logging.getLogger("telemetry_test").warning(f"value is {value}")
        return (value,)


class Fail(Constant):
    def execute(self, value):
        raise ValueError("node failed")


class Interrupt(Constant):
    def execute(self, value):
        raise comfy.model_management.InterruptProcessingException()


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


@pytest.fixture
def otel(monkeypatch):
    for name, node in {
        "TelemetryConstant": Constant,
        "TelemetryIncrement": Increment,
        "TelemetryAsyncIncrement": AsyncIncrement,
        "TelemetryExpandIncrement": ExpandIncrement,
        "TelemetryLog": Log,
        "TelemetryFail": Fail,
        "TelemetryInterrupt": Interrupt,
    }.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, node)

    spans = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    reader = InMemoryMetricReader()
    logs = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
    telemetry._activate(tracer_provider, MeterProvider(metric_readers=[reader]), logger_provider)
    yield SimpleNamespace(spans=spans, reader=reader, logs=logs)
    telemetry._deactivate()


def execute(prompt, outputs, extra_data=None):
    executor = PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})
    executor.execute(prompt, "telemetry-prompt", extra_data or {}, outputs)
    return executor


def by_name(spans):
    found = {}
    for span in spans:
        assert span.name not in found, f"more than one {span.name} span"
        found[span.name] = span
    return found


def metric_points(reader, name):
    for resource_metrics in reader.get_metrics_data().resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    return list(metric.data.data_points)
    return []


def test_prompt_trace_nests_nodes_and_links_logs(otel):
    prompt = {
        "1": {"class_type": "TelemetryConstant", "inputs": {"value": 1}},
        "2": {"class_type": "TelemetryAsyncIncrement", "inputs": {"value": ["1", 0]}},
        "3": {"class_type": "TelemetryExpandIncrement", "inputs": {"value": ["2", 0]}},
        "4": {"class_type": "TelemetryLog", "inputs": {"value": ["3", 0]}},
    }
    extra_data = {"create_time": int(time.time() * 1000) - 200, "traceparent": f"00-{TRACE_ID}-{PARENT_SPAN_ID}-01"}
    assert execute(prompt, ["4"], extra_data).success

    spans = by_name(otel.spans.get_finished_spans())
    assert set(spans) == {"prompt", "queue", "TelemetryConstant", "TelemetryAsyncIncrement",
                          "TelemetryExpandIncrement", "TelemetryIncrement", "TelemetryLog"}
    assert {format(s.context.trace_id, "032x") for s in spans.values()} == {TRACE_ID}

    root = spans["prompt"]
    assert format(root.parent.span_id, "016x") == PARENT_SPAN_ID
    assert root.attributes["comfyui.prompt.id"] == "telemetry-prompt"
    assert root.attributes["comfyui.prompt.outcome"] == "success"
    assert root.attributes["comfyui.prompt.nodes.executed"] == 5
    assert spans["queue"].parent.span_id == root.context.span_id
    assert spans["queue"].end_time - spans["queue"].start_time >= 150_000_000

    for name in ("TelemetryConstant", "TelemetryAsyncIncrement", "TelemetryExpandIncrement", "TelemetryLog"):
        assert spans[name].parent.span_id == root.context.span_id
        assert spans[name].attributes["comfyui.node.outcome"] == "success"
    assert spans["TelemetryAsyncIncrement"].end_time - spans["TelemetryAsyncIncrement"].start_time >= 50_000_000

    expand, expanded = spans["TelemetryExpandIncrement"], spans["TelemetryIncrement"]
    assert expanded.parent.span_id == expand.context.span_id
    assert expanded.attributes["comfyui.node.parent_id"] == "3"
    assert expand.start_time <= expanded.start_time and expanded.end_time <= expand.end_time

    records = [r.log_record for r in otel.logs.get_finished_logs() if r.log_record.body == "value is 3"]
    assert len(records) == 1
    assert records[0].span_id == spans["TelemetryLog"].context.span_id
    assert records[0].trace_id == int(TRACE_ID, 16)

    node_points = {p.attributes["comfyui.node.class_type"]: p for p in metric_points(otel.reader, "comfyui.node.duration")}
    assert node_points["TelemetryAsyncIncrement"].count == 1
    assert node_points["TelemetryAsyncIncrement"].sum >= 0.05
    [prompt_point] = metric_points(otel.reader, "comfyui.prompt.duration")
    assert prompt_point.attributes == {"comfyui.prompt.outcome": "success"}
    [queue_point] = metric_points(otel.reader, "comfyui.prompt.queue.duration")
    assert queue_point.sum >= 0.15


def test_failed_node_marks_node_and_prompt_as_errors(otel):
    prompt = {
        "1": {"class_type": "TelemetryConstant", "inputs": {"value": 1}},
        "2": {"class_type": "TelemetryFail", "inputs": {"value": ["1", 0]}},
    }
    assert not execute(prompt, ["2"]).success

    spans = by_name(otel.spans.get_finished_spans())
    assert spans["prompt"].parent is None
    for name in ("prompt", "TelemetryFail"):
        assert spans[name].status.status_code == StatusCode.ERROR
        assert spans[name].attributes["error.type"] == "ValueError"
    assert spans["prompt"].attributes["comfyui.prompt.outcome"] == "error"
    assert spans["TelemetryFail"].attributes["comfyui.node.outcome"] == "error"
    assert "node failed" not in str(dict(spans["TelemetryFail"].attributes))
    assert spans["TelemetryConstant"].status.status_code == StatusCode.UNSET


def test_interrupted_prompt_is_not_an_error(otel):
    prompt = {
        "1": {"class_type": "TelemetryConstant", "inputs": {"value": 1}},
        "2": {"class_type": "TelemetryInterrupt", "inputs": {"value": ["1", 0]}},
    }
    assert not execute(prompt, ["2"]).success

    spans = by_name(otel.spans.get_finished_spans())
    assert spans["prompt"].attributes["comfyui.prompt.outcome"] == "interrupted"
    assert spans["TelemetryInterrupt"].attributes["comfyui.node.outcome"] == "interrupted"
    for name in ("prompt", "TelemetryInterrupt"):
        assert spans[name].status.status_code == StatusCode.UNSET
        assert "error.type" not in spans[name].attributes


def test_cached_nodes_are_counted_not_traced(otel):
    prompt = {
        "1": {"class_type": "TelemetryConstant", "inputs": {"value": 1}},
        "2": {"class_type": "TelemetryIncrement", "inputs": {"value": ["1", 0]}},
    }
    executor = PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})
    executor.execute(prompt, "first", {}, ["2"])
    otel.spans.clear()
    executor.execute(prompt, "second", {}, ["2"])

    spans = by_name(otel.spans.get_finished_spans())
    assert set(spans) == {"prompt", "queue"}
    assert spans["prompt"].attributes["comfyui.prompt.nodes.cached"] == 2
    assert spans["prompt"].attributes["comfyui.prompt.nodes.executed"] == 0
    [cached_point] = metric_points(otel.reader, "comfyui.node.cached")
    assert cached_point.value == 2


def test_traced_functions_nest_under_the_running_node(otel, monkeypatch):
    @telemetry.traced("load models")
    def load():
        return "loaded"

    class Load(Constant):
        def execute(self, value):
            load()
            return (value,)

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TelemetryLoad", Load)
    assert execute({"1": {"class_type": "TelemetryLoad", "inputs": {"value": 1}}}, ["1"]).success
    spans = by_name(otel.spans.get_finished_spans())
    assert spans["load models"].parent.span_id == spans["TelemetryLoad"].context.span_id


def test_disabled_telemetry_records_nothing(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TelemetryConstant", Constant)
    assert not telemetry.enabled
    assert telemetry.prompt_span("p", {}) is telemetry._NOOP
    assert execute({"1": {"class_type": "TelemetryConstant", "inputs": {"value": 1}}}, ["1"]).success
    assert telemetry._prompt is None


def test_importing_without_the_flag_does_not_load_opentelemetry():
    code = ("import sys, comfy.telemetry; comfy.telemetry.setup(); "
            "print(any(m == 'opentelemetry' or m.startswith('opentelemetry.') for m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_resource_defaults_yield_to_otel_environment(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    assert telemetry._resource().attributes["service.name"] == "comfyui"

    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=from-attributes,deployment.environment=lab")
    assert telemetry._resource().attributes["service.name"] == "from-attributes"
    monkeypatch.setenv("OTEL_SERVICE_NAME", "render-box")
    resource = telemetry._resource()
    assert resource.attributes["service.name"] == "render-box"
    assert resource.attributes["deployment.environment"] == "lab"
