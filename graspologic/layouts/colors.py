# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
from itertools import cycle
import json
import math
import numpy as np
import os
from pathlib import Path
import pkg_resources
from sklearn.preprocessing import minmax_scale
from typing import Any, Dict, Optional, Tuple


__all__ = ["nominal_colors", "sequential_colors"]


def _load_thematic_json(path: Optional[str]) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    if path is not None and Path(path).is_file():
        colors_path = path
    else:
        atexit.register(pkg_resources.cleanup_resources)
        include_path = pkg_resources.resource_filename(__package__, "include")
        colors_path = os.path.join(include_path, "colors-100.json")

    with open(colors_path) as thematic_json_io:
        thematic_json = json.load(thematic_json_io)
    light = thematic_json["light"]
    dark = thematic_json["dark"]
    return light, dark


_CACHED_LIGHT, _CACHED_DARK = _load_thematic_json(None)


def _get_colors(light_background: bool, theme_path: Optional[str]) -> Dict[Any, Any]:
    (
        light,
        dark,
    ) = _CACHED_LIGHT, _CACHED_DARK if theme_path is None else _load_thematic_json(
        theme_path
    )
    return light if light_background else dark


def nominal_colors(
    partitions: Dict[Any, int],
    light_background: bool = True,
    theme_path: Optional[str] = None,
) -> Dict[Any, str]:
    """

    Parameters
    ----------
    partitions
    light_background
    theme_path

    Returns
    -------

    """
    # get nominal colors
    color_scheme = _get_colors(light_background, theme_path)
    partition_populations = {}
    for node_id, partition in partitions.items():
        count = partition_populations.get(partition, 0) + 1
        partition_populations[partition] = count

    ordered_partitions = sorted(
        partition_populations.items(), key=lambda x: x[1], reverse=True
    )
    nominal_cycle = cycle(color_scheme["nominal"])
    colors_by_partitions = {}
    for index, item in enumerate(ordered_partitions):
        partition, _ = item
        color = next(nominal_cycle)
        colors_by_partitions[partition] = color

    colors_by_node = {
        node_id: colors_by_partitions[partition]
        for node_id, partition in partitions.items()
    }
    return colors_by_node


def sequential_colors(
    node_and_value: Dict[Any, float],
    light_background: bool = True,
    use_log_scale: bool = False,
    theme_path: Optional[str] = None,
) -> Dict[Any, str]:
    """

    Parameters
    ----------
    node_and_value
    light_background
    use_log_scale
    theme_path

    Returns
    -------

    """
    color_scheme = _get_colors(light_background, theme_path)
    color_list = color_scheme["sequential"]
    num_colors = len(color_list)

    keys, values = zip(*node_and_value.items())

    if use_log_scale:
        values = map(math.log, values)

    np_values = np.array(values).reshape(1, -1)
    new_values = minmax_scale(np_values, feature_range=(0, num_colors - 1), axis=1)
    node_colors = {}
    for key_index, node_id in enumerate(keys):
        index = int(new_values[0, key_index])

        color = color_list[index]
        node_colors[node_id] = color

    return node_colors
