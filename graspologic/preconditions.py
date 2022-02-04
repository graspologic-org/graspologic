# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numbers
from typing import Any, Union

import networkx as nx

from graspologic.types import Tuple


def check_argument_types(
    value: Any, required_types: Union[type, Tuple[type, ...]], message: str
) -> None:
    """
    Raises a TypeError if the provided ``value`` is not one of the ``required_types``

    Parameters
    ----------
    value : Any
        The argument to test for valid type
    required_types : Union[type, Tuple[type, ...]]
        A type or a n-ary tuple of types to test for validity
    message : str
        The message to use as the body of the TypeError

    Raises
    ------
    TypeError if the type is not one of the ``required_types``
    """
    if not isinstance(value, required_types):
        raise TypeError(message)


def check_optional_argument_types(
    value: Any, required_types: Union[type, Tuple[type, ...]], message: str
) -> None:
    """
    Raises a TypeError if the provided ``value`` is not one of the ``required_types``,
    unless it is None.  A None value is treated as a valid type.

    Parameters
    ----------
    value : Any
        The argument to test for valid type
    required_types : Union[type, Tuple[type, ...]]
        A type or a n-ary tuple of types to test for validity
    message : str
        The message to use as the body of the TypeError

    Raises
    ------
    TypeError if the type is not one of the ``required_types``, unless it is None
    """
    if value is None:
        return
    check_argument_types(value, required_types, message)


def check_argument(check: bool, message: str) -> None:
    """
    Raises a ValueError if the provided check is false

    >>> from graspologic import preconditions
    >>> x = 5
    >>> preconditions.check_argument(x < 5, "x must be less than 5")
    Traceback (most recent call last):
        ...
    ValueError: x must be less than 5

    Parameters
    ----------
    value : Any
        The argument to test for valid type
    required_types : Union[type, Tuple[type, ...]]
        A type or a n-ary tuple of types to test for validity
    message : str
        The message to use as the body of the TypeError

    Raises
    ------
    TypeError if the type is not one of the ``required_types``
    """
    if not check:
        raise ValueError(message)


def is_real_weighted(
    graph: Union[nx.Graph, nx.DiGraph], weight_attribute: str = "weight"
) -> bool:
    """
    Checks every edge in ``graph`` to ascertain whether it has:

        - a ``weight_attribute`` key in the data dictionary for the edge
        - if that ``weight_attribute`` value is a subclass of numbers.Real

    If any edge fails this test, it returns ``False``, else ``True``

    Parameters
    ----------
    graph : Union[nx.Graph, nx.DiGraph]
        The networkx graph to test
    weight_attribute : str (default="weight")
        The edge dictionary data attribute that holds the weight. Default is ``weight``.

    Returns
    -------
    bool
        ``True`` if every edge has a numeric ``weight_attribute`` weight, ``False`` if
        any edge fails this test

    """
    # not only must every edge have a weight attribute but the value must be numeric
    return all(
        (
            weight_attribute in data
            and isinstance(data[weight_attribute], numbers.Real)
            for _, _, data in graph.edges(data=True)
        )
    )
