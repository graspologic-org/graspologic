# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from graspologic.layouts.classes import NodePosition
from ._node import _Node
import math
import logging
from sklearn.preprocessing import normalize
import numpy as np
from typing import NamedTuple, List, Tuple

from scipy.spatial import distance

_EPSILON = 0.001
logger = logging.getLogger(__name__)


def node_positions_overlap(n1: NodePosition, n2: NodePosition):
    return is_overlap(n1.x, n1.y, n1.size, n2.x, n2.y, n2.size)


def is_overlap(x1, y1, s1, x2, y2, s2):
    min_dist = s1 + s2
    if abs(x2 - x1) > min_dist or abs(y2 - y1) > min_dist:
        # shortcut to reduce the amount of math required, the distance calc is more expensive
        return False
    d = distance.euclidean([x1, y1], [x2, y2])
    if d <= s1 + s2:
        return True
    return False


def get_overlapping_any_node_and_index(node, new_x, new_y, nodes, start, end):
    #print (f"nde: {node}, new ({new_x},{new_y}), #nodes: {len(nodes)}, start:end {start}:{end}")
    overlapping = None
    idx = 0
    for idx, n in enumerate(nodes[start:end]):
        if n.node_id == node.node_id:
            # don't check against self
            continue
        if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
            overlapping = n
            break
    return idx + start, overlapping


def is_overlapping_any_node_and_index_with_grid(
    node, new_x, new_y, nodes, start, end, size, grid
):
    overlapping = None
    idx = 0
    potentially_over_lapping = grid.get_potential_overlapping_nodes(new_x, new_y, size)
    for idx, n in enumerate(nodes[start:end]):
        if n not in potentially_over_lapping:
            continue
        if n.node_id == node.node_id:
            # don't check against self
            continue
        if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
            overlapping = n
            break
    return idx + start, overlapping

def max_node_size(nodes):
    max_size = -1
    for n in nodes:
        max_size = max(n.size, max_size)
    return max_size

def stats_nodes(nodes):
    max_x = -math.inf
    min_x = math.inf
    max_y = -math.inf
    min_y = math.inf
    max_size = -1
    for n in nodes:
        max_x = max(n.x, max_x)
        max_y = max(n.y, max_y)
        min_x = min(n.x, min_x)
        min_y = min(n.y, min_y)
        max_size = max(n.size, max_size)
    return min_x, min_y, max_x, max_y, max_size


def total_area(min_x, min_y, max_x, max_y):
    return (max_x - min_x) * (max_y - min_y)


def move_point_on_line(a, b, ratio):
    npa = np.array(a)
    npb = np.array(b)
    d = npb - npa
    d_norm = normalize([d])
    new_point = b + d_norm[0] * ratio
    return new_point


def scale_graph(g, scale_factor):
    for _, n in g.items():
        n.x, n.y = move_point_on_line([0, 0], [n.x, n.y], scale_factor)
    return g

class Extent(NamedTuple):
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def total_area(self):
        return abs((self.max_x - self.min_x) * (self.max_y - self.min_y))
    def x_range(self) -> float:
        return self.max_x - self.min_x
    def y_range(self) -> float:
        return self.max_y - self.min_y

def find_bounds(nodes: List[_Node]):
    max_x = -math.inf
    min_x = math.inf
    max_y = -math.inf
    min_y = math.inf
    for n in nodes:
        max_x = max(n.x, max_x)
        max_y = max(n.y, max_y)
        min_x = min(n.x, min_x)
        min_y = min(n.y, min_y)
    return Extent(min_x, min_y, max_x, max_y)

class _SmartCell:
    """
    A SmartCell either contains one node or a set of children SmartCells that
    are inside of this SmartCell. The number children depends on the relative
    sizes of the nodes trying to be inserted.
    """
    def __init__(self, bounds: Extent, size: float) -> None:
        self.size = size
        self.bounds = bounds
        self.full = False
        self.node = None
        self.children = {}

    def add_node (self, node: _Node) -> Tuple[bool, float, float]:
        if self.is_full():
            raise Exception(f"Can't add node to full _SmartCell, size: {node.size}")
        center_x, center_y = 0.0, 0.0
        if len(self.children) > 0:
            # if we get in here we might be full after we leave we need
            #to verify that
            col, row, new_bounds = self.find_cell_and_bounds(node)
            if self.contains( col, row):
                child_cell = self.get_child_cell(col, row)
            else:
                child_cell = _SmartCell(new_bounds, self.children_size)
                self.children[(col, row)] = child_cell

            ##now we have the cell that we want to insert into, if it is full
            #we need to find another one
            if not child_cell.is_full():
                center_x, center_y = child_cell.add_node(node)
                self.full = child_cell.is_full() and self.are_others_full_by_cell(child_cell)
            else:
                #brute force search for new cell. TODO: FIX THIS
                num_checked = 0
                c, r = 0,0
                breaking = False
                for c in range(self.columns):
                    for r in range(self.rows):
                        num_checked += 1
                        if self.contains(c, r):
                            child_cell = self.get_child_cell(c, r)
                            if not child_cell.is_full():
                                center_x, center_y = child_cell.add_node(node)
                                breaking = True
                                break
                        else:
                            new_bounds = self.bounds_for_cell(c, r)
                            child_cell = _SmartCell(new_bounds, self.children_size)
                            self.children[(c, r)] = child_cell
                            center_x, center_y = child_cell.add_node(node)
                            breaking = True
                            break
                    if breaking:
                        break
                # if we put the last child in there then we are full.
                #print (f"checking to see if we should be full, checked: {num_checked}, rxc: {self.rows * self.columns} child_full: {child_cell.is_full()}, c: {c}, r: {r}, others: {self.are_others_full(c,r)}")
                self.full = child_cell.is_full() and self.are_others_full(c,r)
        else:
            if node.size >= self.size/2:
                #this Cell can only fit one node, set it and mark full
                self.node = node
                self.full = True
                #center_x, center_y = self.get_cell_center(0, 0)
                center_x = self.bounds.min_x + self.size/2
                center_y = self.bounds.min_y + self.size/2
            else:
                # This is the first time we are dividing into children nodes
                #first find the cell we should be in, then create the cell
                # we are guaranteed to not be full after this
                self.children_size = node.size*2
                self.columns = int(self.bounds.x_range() // self.children_size)
                self.rows = int(self.bounds.y_range() // self.children_size)
                col, row, new_bounds = self.find_cell_and_bounds(node)
                new_cell = _SmartCell(new_bounds, self.children_size)
                self.children[(col, row)] = new_cell
                center_x, center_y = new_cell.add_node(node)
        return center_x, center_y

    def are_others_full(self, col: int, row: int):
        if len(self.children) == 0:
            return self.is_full()
        for c in range(self.columns):
            for r in range(self.rows):
                #print (f"({c},{r}) - {row}:{col} - contains: {self.contains(c,r)}")
                if r == row and c == col:
                    pass
                if not self.contains(c, r):
                    return False
                if not self.get_child_cell(c, r).is_full():
                    return False
        return True

    def are_others_full_by_cell(self, child):
        if len(self.children) == 0:
            return self.is_full()
        for c in range(self.columns):
            for r in range(self.rows):
                #print (f"by cell ({c},{r}) - contains: {self.contains(c,r)}")
                if not self.contains(c, r):
                    return False
                other_child = self.get_child_cell(c, r)
                if child != other_child and not other_child.is_full():
                    return False
        return True



    def get_child_cell(self, col: int, row: int):
        return self.children[(col, row)]

    def is_full(self) -> bool:
        return self.full

    def contains(self, col: int, row: int):
        return (col, row) in self.children

    def bounds_for_cell(self, col: int, row: int):
        side_size = self.children_size
        min_x = self.bounds.min_x+(col*side_size)
        min_y = self.bounds.min_y+(row*side_size)
        max_x = min_x + side_size
        max_y = min_y + side_size
        bounds = Extent(min_x, min_y, max_x, max_y )
        return bounds

    def get_cell_center(self, col: int, row: int) -> Tuple[float, float]:
        center_x = self.bounds.min_x + col*self.size + self.size/2
        center_y = self.bounds.min_y + row*self.size + self.size/2
        return center_x, center_y

    def find_cell_and_bounds(self, node: _Node):
        # zero the cordinates
        side_size = self.children_size
        zeroed_x = node.x - self.bounds.min_x
        zeroed_y = node.y - self.bounds.min_y
        column = int(zeroed_x // side_size)
        row = int(zeroed_y // side_size)
        bounds = self.bounds_for_cell(column, row)
        return column, row, bounds

class _SmartGrid:
    def __init__(self, bounds: Extent, max_node_size: float) -> None:
        self.bounds = bounds
        self.max_node_size = max_node_size
        self.grid = _SmartCell(bounds, max_node_size*2)

    def find_grid_cell_and_center(self, x: float, y: float)-> Tuple[int, int, float, float]:
        # zero the cordinates
        side_size = self.max_node_size * 2.0
        zeroed_x = x - self.bounds.min_x
        zeroed_y = y - self.bounds.min_y
        x_cell = int(zeroed_x // side_size)
        y_cell = int(zeroed_y // side_size)
        return x_cell, y_cell, self.bounds.min_x + side_size * x_cell, self.bounds.min_y + side_size * y_cell

    def is_full(self) -> bool:
        return self.grid.is_full()

#    def contains_node(self, col: int, row: int) -> bool:
#        return (col, row) in self.grid
#
#    def get_cell(self, col: int, row: int) -> _SmartCell:
#        return self.grid[(col, row)]
#
#    def set_cell(self, col: int, row: int, new_cell: _SmartCell) -> None:
#        self.grid[(col, row)] = new_cell

    def add_node(self, node: _Node) -> Tuple[float, float]:
        return self.grid.add_node(node)

class _SmartQuadNode:
    """
    Represents a node in a quad tree. Each node has a list of nodes that are represented here
    or in its children.
    Each node knows its own depth in the tree. Each node will have less than max_nodes_per_quad
    in the nodes list or it will have populated children.
    Each node also has a pointer back up to its parent.  The root node in a tree will have None for
    its parent attribute.
    Each node has an x and y that are the center point which is a weighted center.
    Each node has min_x, min_y, max_x_, max_y that defines the area of this node
    Each node also has four child nodes NW, NE, SE, and SW they represent the nodes that are
    that direction from the center.

    Each node has a property indicating if has been laid out.
    """

    max_ratio = 0.95

    def __init__(self, nodes, depth: int, extent: Extent, max_nodes_per_quad : int, parent=None):
        if len(nodes) <= 0:
            raise Exception("Invalid to create a quad node with zero nodes!")

        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.depth = depth
        self.extent = extent
        self.max_nodes_per_quad = max_nodes_per_quad
        self.is_laid_out = False
        self.parent = parent

        self.max_size = max_node_size(self.nodes)
        self.circle_size = self.total_circle_size()
        self.square_size = self.total_square_size()
        self.tot_area = self.extent.total_area()
        if self.tot_area == 0:
            self.tot_area = 0.001 # just to prevent division by 0
        self.sq_ratio = self.square_size / self.tot_area
        self.cir_ratio = self.circle_size / self.tot_area
        if self.sq_ratio < 0:
            raise Exception(f"sq_ratio: {self.sq_ratio}, sq_size: {self.square_size}, tot_size: = {self.tot_area}, bounds: {self.extent}")
        self.x_range = self.extent.x_range()
        self.y_range = self.extent.y_range()

        self.total_basic_cells = self.get_total_basic_cells()
        self._find_center()
        self._push_to_kids()
        ## these can be removed eventually, they are for debugging
        self.total_nodes_moved = 0
        self.not_first_choice = 0

    def __lt__(self, other):
        return self.x < other.x

    def child_list(self):
        return [self.NW, self.NE, self.SW, self.SE]

    def get_total_basic_cells(self):
        side_size = self.max_size * 2.0
        number_of_x_cells = self.x_range // side_size
        number_of_y_cells = self.y_range // side_size
        total_cells = number_of_y_cells * number_of_x_cells
        return total_cells

    def _find_center(self):
        # sum X and Y values then divide by number of nodes to get the average (x,y) or center
        x, y = 0.0, 0.0
        for n in self.nodes:
            x += n.x
            y += n.y
        # need to fix x and y
        self.x = x / self.num_nodes
        self.y = y / self.num_nodes

    def _push_to_kids(self):
        if len(self.nodes) <= self.max_nodes_per_quad:
            # if len == nodes then we already have the center and mass initialized because of
            # the calculation done in the constructor
            self.NW = None
            self.NE = None
            self.SW = None
            self.SE = None
            return
        nw_nodes, sw_nodes, ne_nodes, se_nodes = [], [], [], []
        for node in self.nodes:
            if node.y > self.y:
                if node.x > self.x:
                    ne_nodes.append(node)
                else:
                    nw_nodes.append(node)
            else:
                if node.x > self.x:
                    se_nodes.append(node)
                else:
                    sw_nodes.append(node)

        extent_nw = Extent(self.extent.min_x, self.y, self.x, self.extent.max_y)
        extent_ne = Extent(self.x, self.y, self.extent.max_x, self.extent.max_y)
        extent_sw = Extent(self.extent.min_x, self.extent.min_y, self.x, self.y)
        extent_se = Extent(self.x, self.extent.min_y, self.extent.max_x, self.y)
        self.NW = (
            _SmartQuadNode(nw_nodes, self.depth + 1, extent_nw, self.max_nodes_per_quad, self)
            if len(nw_nodes) > 0
            else None
        )
        self.NE = (
            _SmartQuadNode(ne_nodes, self.depth + 1, extent_ne, self.max_nodes_per_quad, self)
            if len(ne_nodes) > 0
            else None
        )
        self.SW = (
            _SmartQuadNode(sw_nodes, self.depth + 1, extent_sw, self.max_nodes_per_quad, self)
            if len(sw_nodes) > 0
            else None
        )
        self.SE = (
            _SmartQuadNode(se_nodes, self.depth + 1, extent_se, self.max_nodes_per_quad, self)
            if len(se_nodes) > 0
            else None
        )

        return

    def num_children(self):
        total = 1
        for quad in self.child_list():
            if quad is not None:
                total += quad.num_children()
        return total

    def number_of_nodes(self):
        return self.num_nodes

    def print_node(self, max_depth):
        # print (tot_area, min_x, min_y, max_x, max_y)
        # if tot_area == 0:
        # 	for n in self.nodes:
        # 		print(n.x, n.y)
        # 	tot_area = 0.001
        tag = "ttttt" if self.circle_size / self.tot_area > self.max_ratio else ""
        print(
            "-" * self.depth,
            "size: %d, (%g, %g), (%g, %g) cs: %g, tot: %g, ratio: %g, ss: %g, ratio: %g, tag: %s"
            % (
                len(self.nodes),
                self.min_x,
                self.min_y,
                self.max_x,
                self.max_y,
                self.circle_size,
                self.tot_area,
                self.circle_size / self.tot_area,
                self.square_size,
                self.square_size / self.tot_area,
                tag,
            ),
        )
        if self.depth >= max_depth:
            return

        for quad in self.child_list():
            if quad:
                quad.print_node(max_depth)

    def total_circle_size(self):
        total_size = 0
        for n in self.nodes:
            total_size += n.size * n.size * math.pi
        return total_size

    def total_square_size(self):
        total_size = 0
        for n in self.nodes:
            total_size += n.size * n.size * 2 * 2
        return total_size

    def _node_stats(self):
        return [
                    self.x,
                    self.y,
                    self.depth,
                    self.extent.min_x,
                    self.extent.min_y,
                    self.extent.max_x,
                    self.extent.max_y,
                    self.circle_size / self.tot_area,
                    self.total_basic_cells,
                    len(self.nodes)
                ]
    def _node_stats_header(self):
        return [
            "x",
            "y",
            "depth",
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "covered_ratio",
            "cells",
            "nodes"]

    def get_stats_for_quad(self, max_depth, stats_list, magnification=10):
        if self.depth >= max_depth:
            stats_list.append(self._node_stats())
            return

        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad is not None:
                quad.get_stats_for_quad(max_depth, stats_list, magnification)
        stats_list.append(self._node_stats())
        return

    def find_grid_cell_and_center(self, min_x, min_y, max_size, x, y):
        # zero the cordinates
        side_size = max_size * 2.0
        zeroed_x = x - min_x
        zeroed_y = y - min_y
        x_cell = int(zeroed_x // side_size)
        y_cell = int(zeroed_y // side_size)
        return x_cell, y_cell, min_x + side_size * x_cell, min_y + side_size * y_cell

    #### This method could be must more intelligent about how to find the next free cell
    def find_free_cell(
        self, cells, x, y, num_x_cells, num_y_cells, min_x, min_y, max_size
    ):
        # zero the cordinates
        square_size = max_size * 2
        new_x, new_y = x, y
        distance_to_move = 1
        count = 0
        done = True
        while (new_x, new_y) in cells or not done:
            done = False
            # go right
            # print ('newx:', new_x+1, 'end x', new_x+distance_to_move+1)
            for new_x in range(new_x + 1, new_x + distance_to_move + 1):
                # print ("ffc-r (%g,%g) to_move %d"  %(new_x, new_y, distance_to_move))
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    # print('breaking-r')
                    done = True
                    break
            if done:
                break
            # go up
            for new_y in range(new_y + 1, new_y + distance_to_move + 1):
                # print ("ffc-u (%g,%g) to_move %d"  %(new_x, new_y, distance_to_move))
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    # print('breaking-u')
                    done = True
                    break
            distance_to_move += 1
            if done:
                break
            # go left
            # print ('about to go left, %g, %g' %(new_x, new_y))
            for new_x in range(new_x - 1, new_x - distance_to_move - 1, -1):
                # print ("ffc-l (%g,%g) to_move %d"  %(new_x, new_y, distance_to_move))
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    # print('breaking-l')
                    done = True
                    break
            if done:
                break
            # go down
            for new_y in range(new_y - 1, new_y - distance_to_move - 1, -1):
                # print ("ffc-d (%g,%g) to_move %d"  %(new_x, new_y, distance_to_move))
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    # print('breaking-d')
                    done = True
                    break
            distance_to_move += 1
            if done:
                break
            # keep going
            # print ("ALL FULL", count)
            count += 1

        return new_x, new_y, min_x + square_size * new_x, min_y + square_size * new_y

    def layout_node_list(self, bounds : Extent, max_size : float, node_list: List[_Node]) -> bool:
        """
        This method will layout the nodes in the current quad.  If there are more nodes than cells an Exception
        will be raised.
        :param bounds:
        :param max_size:
        :param node_list:
        :return:
        """
        # print ("layout_node_list %d", len(node_list))
        self._mark_laid_out()
        nodes_by_size = sorted(node_list, key=lambda n: n.size, reverse=True)
        largest_size = nodes_by_size[0].size
        if largest_size != max_size:
            raise Exception(
                f"This can not be!!! max: {max_size}, largest: {largest_size}"
            )
        num_nodes = len(node_list)

        side_size = max_size * 2.0
        x_range = bounds.x_range()
        y_range = bounds.y_range()
        number_of_x_cells = x_range // side_size
        number_of_y_cells = y_range // side_size

        #if num_nodes > number_of_x_cells*number_of_y_cells:
        #    raise Exception(
        #        "Too many nodes per Cell for this quad! nodes: %d, cells: %d"
        #        % (num_nodes, self.total_basic_cells)
        #    )

        number_overlapping = 0
        for idx, node_to_move in enumerate(nodes_by_size):
            for placed_node in nodes_by_size[idx + 1 :]:
                if is_overlap(
                    node_to_move.x,
                    node_to_move.y,
                    node_to_move.size,
                    placed_node.x,
                    placed_node.y,
                    placed_node.size,
                ):
                    number_overlapping += 1
                    break
            if number_overlapping > 0:
                break
        ## if none of the nodes are overlapping then they all fit and no need to move them
        if number_overlapping == 0:
            return True

        smart_grid = _SmartGrid(bounds, largest_size)
        for idx, node_to_move in enumerate(nodes_by_size):
            if smart_grid.is_full():
                return False
            cell_center_x, cell_center_y, = smart_grid.add_node(node_to_move)
            node_to_move.x = cell_center_x
            node_to_move.y = cell_center_y

        return True

    def _mark_laid_out(self):
        """
        Recursively mark all children as being laid out already
        :return:
        """
        self.is_laid_out = True
        for qn in self.child_list():
            if qn is not None:
                qn._mark_laid_out()

    def get_new_bounds(self, extent, max_size, nodes):
        xrange = extent.max_x - extent.min_x
        yrange = extent.max_y - extent.min_y
        side_size = 2 * max_size
        x_cells = xrange // side_size
        y_cells = yrange // side_size
        total_cells = y_cells * x_cells
        cells_needed = len(nodes)
        # print ('original_total cells %d, (%d, %d) needed: %d' %(total_cells, x_cells, y_cells, cells_needed))
        while total_cells < cells_needed:
            x_cells += 1
            y_cells += 1
            total_cells = y_cells * x_cells
            # print ('new_cells %d, (%d, %d) needed: %d' %(total_cells, x_cells, y_cells, cells_needed))
        new_xrange = x_cells * side_size + 2
        new_yrange = y_cells * side_size + 2
        # print("bounds", extent.min_x, extent.min_y, extent.max_x, max_y, new_xrange, new_yrange, side_size, x_cells, y_cells, total_cells)
        expanded_min_x = extent.min_x - (new_xrange - xrange) / 2
        expanded_min_y = extent.min_y - (new_yrange - yrange) / 2
        expanded_max_x = extent.max_x + (new_xrange - xrange) / 2
        expanded_max_y = extent.max_y + (new_yrange - yrange) / 2

        ## now I have the number of cells we need to go back and expand the bounds
        return Extent(expanded_min_x, expanded_min_y, expanded_max_x, expanded_max_y)

    def layout(self):
        # print("layout_quad")
        num_skipped = 1
        if self.is_laid_out:
            # print ("ALREADY LAID OUT!!, ratio: %g " %(nodes_per_cell))
            return num_skipped

        if self.total_basic_cells == 0:
            nodes_per_cell = math.inf
        else:
            nodes_per_cell = self.num_nodes / self.total_basic_cells

        has_children = False
        for quad in self.child_list():
            if quad is not None:
                has_children = True

        if not has_children:
            did_fit = self.layout_node_list(self.extent, self.max_size, self.nodes)
            current = self
            while not did_fit:
                logger.info (
                    f"Did not fit current level {current.depth}, "
                    f"sq_ratio: {current.sq_ratio}, sq_size: {current.square_size} "
                    f"nodes: {current.num_nodes}, bounds: {current.extent} "
                    f"total size: {current.tot_area}"
                 )
                current = current.parent
                if current is None:
                    break
                did_fit = current.layout_node_list(
                    current.extent,
                    current.max_size,
                    current.nodes,
                )

            if did_fit:
                logger.info (f"contracting nodes that fit")
                current._do_contraction()
                return

            # if we get here current should be None always
            if current is not None:
                raise Exception("This should never happen remove after verifying correctness")

            # expand the canvas and try to lay it out.
            root = self.get_top_quad_node()
            expanded_bounds = self.get_new_bounds(root.extent, root.max_size, root.nodes)
            did_fit = root.layout_node_list(
                expanded_bounds,
                root.max_size,
                root.nodes,
            )
            logger.info (f"contracting nodes that did not fit")
            root._do_contraction()

        else:
            for quad in self.child_list():
                if quad:
                    num_skipped += quad.layout_quad()
        return num_skipped

    def quad_stats(self):
        num_quad_no_kids = 0
        num_quad_to_dense = 0
        num_quad_fits = 0
        total_nodes_moved = self.total_nodes_moved
        not_first_choice = self.not_first_choice
        lowest_level = math.inf
        max_nodes_in_grid = 0

        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                no_kids, too_dense, fits, tnm, nfc, low_level, mng = quad.quad_stats()
                num_quad_no_kids += no_kids
                num_quad_to_dense += too_dense
                num_quad_fits += fits
                total_nodes_moved += tnm
                not_first_choice += nfc
                has_children = True
                lowest_level = min(lowest_level, low_level)
                max_nodes_in_grid = max(max_nodes_in_grid, mng)

        if not has_children:
            num_quad_no_kids += 1
            if len(self.nodes) > self.total_basic_cells:
                if self.total_basic_cells == 0:
                    nodes_to_cells = math.inf
                else:
                    nodes_to_cells = len(self.nodes) / self.total_basic_cells
                # print ("too dense, nodes/cells: %g, nn: %d, cells: %d, level: %d" %(nodes_to_cells, len(self.nodes), self.total_basic_cells, self.depth))
                num_quad_to_dense = 1
                lowest_level = math.inf
                parent = self.parent
                while parent is not None:
                    max_nodes_in_grid = parent.number_of_nodes()
                    lowest_level = parent.depth
                    if parent.number_of_nodes() > parent.total_basic_cells:
                        # doesn't fit, go up one more level
                        parent = parent.parent
                    else:
                        break
            else:
                # we are at the bottom and we can fit everyone in here.
                lowest_level = self.depth
                max_nodes_in_grid = self.num_nodes
                num_quad_fits = 1
        return (
            num_quad_no_kids,
            num_quad_to_dense,
            num_quad_fits,
            total_nodes_moved,
            not_first_choice,
            lowest_level,
            max_nodes_in_grid,
        )

    def num_overlapping(self):
        has_children = False
        num_overlapping_nodes = 0
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                overlapping = quad.num_overlapping()
                num_overlapping_nodes += overlapping
                has_children = True

        if not has_children:
            for idx, node_a in enumerate(self.nodes):
                for node_b in self.nodes[:idx]:
                    if is_overlap(
                        node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size
                    ):
                        num_overlapping_nodes += 1
                        # print ("***Is OVERLAPPING!!!!, to_move: (%g,%g) %g, (%g,%g) %g" % (node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size))
                        break
        return num_overlapping_nodes

    def is_just_outside_box(self, minx, miny, maxx, maxy, maxsize, x, y, size):
        dist = maxsize + size
        if x > minx - dist and x < maxx + dist:
            if y <= maxy + dist and y > miny - dist:
                # Here we are inside the box with buffer around it. now we need to check we are in the area just outside
                # of the original box
                if y > maxy:
                    return True
                elif y < miny:
                    return True
                elif x > maxx:
                    return True
                elif x < minx:
                    return True
                else:
                    return False
        else:
            return False

    def get_top_quad_node(self):
        cur = self.parent
        prev = self
        while cur is not None:
            prev = cur
            cur = cur.parent
        return prev

    def get_nodes_near_lines(self, all_nodes):
        nodes_just_outside = []
        for n in all_nodes:
            if self.is_just_outside_box(
                self.extent.min_x,
                self.extent.min_y,
                self.extent.max_x,
                self.extent.max_y,
                self.max_size,
                n.x,
                n.y,
                n.size,
            ):
                nodes_just_outside.append(n)
        return nodes_just_outside

    def num_overlapping_across_quads(self, all_nodes):
        has_children = False
        num_overlapping_nodes = 0
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                overlapping = quad.num_overlapping_across_quads(all_nodes)
                num_overlapping_nodes += overlapping
                has_children = True

        if not has_children:
            nodes_to_check = self.get_nodes_near_lines(all_nodes)
            for idx, node_a in enumerate(self.nodes):
                # the first one does not need to move all the rest might need to move
                for node_b in nodes_to_check:
                    if node_a.node_id == node_b.node_id:
                        continue
                    if is_overlap(
                        node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size
                    ):
                        num_overlapping_nodes += 1
                        # print ("***Is OVERLAPPING!!!! across quads, to_move: (%g,%g) %g, (%g,%g) %g" % (node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size))
                        break
        return num_overlapping_nodes

    def get_leaf_density_list(self):
        ratio = self.circle_size / self.tot_area

        retval = []
        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                retval += quad.get_leaf_density_list()
                has_children = True
        if has_children:
            return retval

        if self.total_basic_cells == 0:
            nodes_to_cells = math.inf
        else:
            nodes_to_cells = len(self.nodes) / self.total_basic_cells
        return [(nodes_to_cells, ratio, self.total_basic_cells, self)]

    def get_overlapping_node_list(self, node, new_x, new_y, nodes):
        overlapping_nodes = []
        for n in nodes:
            if n.node_id == node.node_id:
                # don't check against self
                continue
            if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
                overlapping_nodes.append(n)
        return overlapping_nodes

    def is_overlapping_any_node(self, node, new_x, new_y, nodes):
        overlapping = None
        for n in nodes:
            if n.node_id == node.node_id:
                # don't check against self
                continue
            if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
                overlapping = n
                break
        return overlapping

    def is_between(self, x, one_end, other_end):
        if x <= one_end:
            return x >= other_end
        else:
            return x <= other_end

    def _do_contraction(self):
        node_list = self.nodes
        self._do_contraction_with_given_nodes(node_list=node_list)
        return

    ### I wanted to add a little thank you to the webiste: https://www.calculator.net/triangle-calculator.html
    ### it helped me debug the issues I was having in the calculation of the overlaps.
    def _do_contraction_with_given_nodes(self, node_list):
        logger.info("contracting nodes:%d" % (len(node_list)))
        nodes_by_size = sorted(node_list, key=lambda n: n.size, reverse=True)
        if self.parent is None:
            #if we are at the root we can skip this check there is nothing just outside of the box
            nodes_around_the_edge = []
        else:
            nodes_around_the_edge = self.get_nodes_near_lines(
                self.get_top_quad_node().nodes
            )
        num_nodes_around_edge = len(nodes_around_the_edge)
        nodes_to_consider = nodes_around_the_edge + nodes_by_size

        for idx, node_to_move in enumerate(nodes_by_size):
            if idx % 100 == 0:
                logger.info(f"processing {idx}")
            # move to its original spot node_to_move.original_x, node_to_move.original_y
            # then move it toward where it is until it does not overlap with anything already placed.
            prev_x, prev_y = node_to_move.original_x, node_to_move.original_y
            new_x, new_y = node_to_move.x, node_to_move.y
            ov_idx = 0
            ov_idx, overlapping_node = get_overlapping_any_node_and_index(
                node_to_move,
                node_to_move.original_x,
                node_to_move.original_y,
                nodes_to_consider,
                ov_idx,
                idx + num_nodes_around_edge,
            )
            if overlapping_node is None:
                new_x, new_y = prev_x, prev_y

            #if node_to_move.x == 0.0:
            if node_to_move.x == node_to_move.original_x:
                # this is needed just in case the min x node is overlapping.
                # then the orginal X is equal to the X where is moves and that give us a divide by zero
                # when calculating the slope
                # We wiggle it just a little bit to prevent an error
                node_to_move.x += _EPSILON
            if node_to_move.y == node_to_move.original_y:
                node_to_move.y += _EPSILON

            # print ("contracting: %d, node_to_move %s, overlapping: %s" % (idx, str(node_to_move.to_list()), overlapping_node))
            while (
                overlapping_node is not None
                and node_to_move.x != node_to_move.original_x
            ):
                # slope doesn't change leave as original_xy
                slope_ca = (node_to_move.y - node_to_move.original_y) / (
                    node_to_move.x - node_to_move.original_x
                )
                if node_to_move.node_id == overlapping_node.node_id:
                    raise Exception(
                        "They should not be the same node!! %s" % (node_to_move.node_id)
                    )
                if node_to_move.original_x == new_x:
                    new_x += _EPSILON
                a = dist_original_to_over = distance.euclidean(
                    [node_to_move.original_x, node_to_move.original_y],
                    [overlapping_node.x, overlapping_node.y],
                )
                b = dist_from_original_to_new = distance.euclidean(
                    [node_to_move.original_x, node_to_move.original_y], [new_x, new_y]
                )
                c = dist_from_new_to_overlapping = distance.euclidean(
                    [new_x, new_y], [overlapping_node.x, overlapping_node.y]
                )
                # print ("not None, a: %g, b: %g, c: %g" %(a, b, c), node_to_move.node_id, node_to_move.size, overlapping_node.size)
                # print ("original(%g,%g), current(%g,%g), overlap(%g,%g)" %(node_to_move.original_x, node_to_move.original_y, new_x, new_y, overlapping_node.x, overlapping_node.y))
                denominator = 2 * a * b
                if 0 == denominator:
                    denominator = 0.0000001
                value = (a ** 2 + b ** 2 - c ** 2) / denominator
                if value >= 1:
                    value = 0.999999
                elif value <= -1:
                    value = -0.999999
                angle_c = math.acos(value)
                len_c_new = node_to_move.size + overlapping_node.size + _EPSILON
                angle_a_new = math.asin(a * math.sin(angle_c) / len_c_new)
                angle_b_new = 180 - math.degrees(angle_c) - math.degrees(angle_a_new)
                new_len_b = (
                    len_c_new * math.sin(math.radians(angle_b_new)) / math.sin(angle_c)
                )
                #print ("slope: %g, angle c: %g, new angle a: %g, newlenC: %g, new angle a: %g, new lenB %g" %(slope_ca, math.degrees(angle_c), math.degrees(angle_a_new), len_c_new, math.degrees(angle_a_new), new_len_b))
                x_new_plus = node_to_move.original_x + math.sqrt(
                    new_len_b ** 2 / (1 + slope_ca ** 2)
                )
                x_new_neg = node_to_move.original_x - math.sqrt(
                    new_len_b ** 2 / (1 + slope_ca ** 2)
                )
                x_plus_diff = x_new_plus - new_x
                x_neg_diff = x_new_neg - new_x
                # print ("both outsize, plus diff: %g, minus diff: %g" %(x_plus_diff, x_neg_diff))
                prev_x, prev_y = new_x, new_y
                if abs(x_plus_diff) < abs(x_neg_diff):
                    new_x = x_new_plus
                    new_y = prev_y - slope_ca * prev_x + slope_ca * x_new_plus
                else:
                    new_x = x_new_neg
                    new_y = prev_y - slope_ca * prev_x + slope_ca * x_new_neg
                #print (f"before: idx: {ov_idx}, ov_node: {overlapping_node}, node_to_move: {node_to_move}, around: {len(nodes_around_the_edge)}, by_size: {len(nodes_by_size)}: new (x,y): ({new_x},{new_y})")
                ov_idx, overlapping_node = get_overlapping_any_node_and_index(
                    node_to_move,
                    new_x,
                    new_y,
                    nodes_to_consider,
                    0,
                    idx + num_nodes_around_edge,
                )
            # print ("after: idx: %d, node: %s" %(ov_idx, str(overlapping_node is not None) ))
            node_to_move.x = new_x
            node_to_move.y = new_y

        # if idx > 20: # only do a few in the first quad
        # 	break

        return

    def collect_nodes(self, all_nodes):
        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                quad.collect_nodes(all_nodes)
                has_children = True

        if not has_children:
            return all_nodes.extend(self.nodes)

    def boxes_by_level(self, boxes):
        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                quad.boxes_by_level(boxes)
                has_children = True

        boxes.append((self.depth, self.extent.min_x, self.extent.min_y, self.extent.max_x, self.extent.max_y))
        return
