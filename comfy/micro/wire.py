from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class MicroPayload(Protocol):
    """A wire payload that can materialize itself as bytes."""

    def as_bytes(self) -> bytes:
        ...


@dataclass(frozen=True)
class BytesPayload:
    """The only payload implementation in v1."""

    data: bytes

    def as_bytes(self) -> bytes:
        return self.data


@dataclass(frozen=True)
class MicroValue:
    """A graph value in wire form."""

    type_name: str
    payload: MicroPayload
