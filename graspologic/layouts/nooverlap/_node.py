# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class _Node:
    def __init__(self, node_id, x, y, size, community=9999999, color=""):
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.original_x = self.x
        self.original_y = self.y
        self.size = float(size)
        self.community = community
        self.color = color

    def reset_original_position(self, newx, newy):
        self.original_x = self.x = newx
        self.original_y = self.y = newy

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __hash__(self):
        return hash(self.node_id)
