from typing import Optional, Sequence

from aio_pika.abc import AbstractChannel
from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult, Decision
from opentelemetry.semconv.attributes.network_attributes import NETWORK_PEER_ADDRESS, NETWORK_PEER_PORT
from opentelemetry.trace import SpanKind, Link, TraceState
from opentelemetry.util.types import Attributes


def patch_spanbuilder_set_channel() -> None:
    """
    The default SpanBuilder.set_channel does not work with aio_pika 9.1 and the refactored connection
    attribute
    """
    import opentelemetry.instrumentation.aio_pika.span_builder
    from opentelemetry.instrumentation.aio_pika.span_builder import SpanBuilder

    def set_channel(self: SpanBuilder, channel: AbstractChannel) -> None:
        if hasattr(channel, "_connection"):
            url = channel._connection.url
            port = url.port or 5672
            self._attributes.update(
                {
                    NETWORK_PEER_ADDRESS: url.host,
                    NETWORK_PEER_PORT: port,
                }
            )

    opentelemetry.instrumentation.aio_pika.span_builder.SpanBuilder.set_channel = set_channel  # type: ignore[misc]


class ProgressSpanSampler(Sampler):
    def get_description(self) -> str:
        return "Sampler which omits aio_pika messages destined/related to progress"

    def should_sample(
            self,
            parent_context: Optional["Context"],
            trace_id: int,
            name: str,
            kind: Optional[SpanKind] = None,
            attributes: Attributes = None,
            links: Optional[Sequence["Link"]] = None,
            trace_state: Optional["TraceState"] = None,
    ) -> "SamplingResult":
        if attributes is not None and "messaging.destination" in attributes and attributes["messaging.destination"].endswith("progress"):
            return SamplingResult(Decision.DROP)
        # the ephemeral reply channels are not required for correct span correlation
        if name.startswith(",amq_") or name.startswith("amq"):
            return SamplingResult(Decision.DROP)
        return SamplingResult(Decision.RECORD_AND_SAMPLE)
