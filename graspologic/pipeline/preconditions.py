# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Tuple, Union


def check_argument_types(
    value: Any, required_types: Union[type, Tuple[type, ...]], message: str
):
    if not isinstance(value, required_types):
        raise TypeError(message)


def check_optional_argument_types(
    value: Any, required_types: Union[type, Tuple[type, ...]], message: str
):
    if value is None:
        return
    check_argument_types(value, required_types, message)


def check_argument(check: bool, message: str):
    if not check:
        raise ValueError(message)
