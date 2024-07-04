"""
This should be imported before entrypoints to correctly configure global options prior to importing packages like torch and cv2.

Use this instead of cli_args to import the args:

>>> from comfy.cmd.main_pre import args

It will enable command line argument parsing. If this isn't desired, you must author your own implementation of these fixes.
"""
import logging
import os
import sys
import warnings

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.semconv.resource import ResourceAttributes as ResAttrs

from .. import options
from ..tracing_compatibility import ProgressSpanSampler
from ..tracing_compatibility import patch_spanbuilder_set_channel
from ..vendor.aiohttp_server_instrumentation import AioHttpServerInstrumentor

options.enable_args_parsing()
if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from ..cli_args import args

if args.cuda_device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    logging.info("Set cuda device to: {}".format(args.cuda_device))

if args.deterministic:
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _create_tracer():
    resource = Resource.create({
        ResAttrs.SERVICE_NAME: args.otel_service_name,
        ResAttrs.SERVICE_VERSION: args.otel_service_version,
    })

    # omit progress spans from aio pika
    sampler = ProgressSpanSampler()
    provider = TracerProvider(resource=resource, sampler=sampler)

    is_debugging = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    has_endpoint = args.otel_exporter_otlp_endpoint is not None

    if has_endpoint:
        otlp_exporter = OTLPSpanExporter()
    # elif is_debugging:
    #     otlp_exporter = ConsoleSpanExporter("comfyui")
    else:
        otlp_exporter = SpanExporter()

    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    # enable instrumentation
    patch_spanbuilder_set_channel()
    AioPikaInstrumentor().instrument()
    AioHttpServerInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    return trace.get_tracer(args.otel_service_name)


tracer = _create_tracer()
__all__ = ["args", "tracer"]
