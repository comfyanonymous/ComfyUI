from __future__ import annotations

import typing

# for nodes that return:
# { "ui" : { "some_field": Any, "other": Any }}
# the outputs dict will be
#  (the node id)
# { "1":         { "some_field": Any, "other": Any }}
OutputsDict = dict[str, dict[str, typing.Any]]
