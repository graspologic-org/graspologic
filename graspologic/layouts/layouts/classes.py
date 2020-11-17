# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import NamedTuple, Optional

__all__ = [
    "NodePosition",
]


class NodePosition(NamedTuple):
    node_id: str
    x: float
    y: float
    size: float
    community: int
