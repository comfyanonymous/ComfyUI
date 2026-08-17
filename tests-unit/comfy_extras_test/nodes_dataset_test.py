"""Tests for the class-attribute contract of the dataset node base classes.

``ImageProcessingNode`` and ``TextProcessingNode`` let subclasses configure
themselves with plain class attributes, and their shared ``define_schema()`` is
what forwards those attributes into ``io.Schema``. Anything ``define_schema()``
forgets to forward is silently dropped from ``/object_info``, so these tests pin
the forwarding itself rather than any single field.
"""

import pytest

from comfy_extras.nodes_dataset import ImageProcessingNode, TextProcessingNode

BASE_CLASSES = (ImageProcessingNode, TextProcessingNode)

# Class attribute -> io.Schema field. The names match today, but keeping the
# mapping explicit means a rename on either side has to be made deliberately.
FORWARDED_ATTRS = {
    "description": "description",
    "search_aliases": "search_aliases",
    "is_deprecated": "is_deprecated",
}


def _node_classes():
    """Every concrete node built on the dataset base classes."""
    found = []
    stack = list(BASE_CLASSES)
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls in BASE_CLASSES or cls.node_id is None:
            continue  # abstract base class, not a registered node
        found.append(cls)
    return sorted(found, key=lambda c: c.__name__)


@pytest.mark.parametrize("node_cls", _node_classes(), ids=lambda c: c.__name__)
def test_class_attributes_are_forwarded_to_schema(node_cls):
    schema = node_cls.define_schema()
    for attr, field in FORWARDED_ATTRS.items():
        assert getattr(schema, field) == getattr(node_cls, attr), (
            f"{node_cls.__name__}.{attr} is not forwarded into io.Schema by "
            f"define_schema(), so /object_info reports "
            f"{field}={getattr(schema, field)!r} instead of "
            f"{getattr(node_cls, attr)!r}"
        )


def test_node_classes_are_discovered():
    """Guard against the parametrization above collapsing to zero cases."""
    assert len(_node_classes()) > 10
