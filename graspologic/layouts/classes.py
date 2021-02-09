# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import NamedTuple

__all__ = [
    "NodePosition",
]


class NodePosition(NamedTuple):
    """
    Contains the node id, 2d coordinates, size, and community id for a node.
    """

    node_id: str
    x: float
    y: float
    size: float
    community: int
