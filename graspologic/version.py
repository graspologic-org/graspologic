# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import configparser
import datetime
from typing import Optional

import pkg_resources


def __from_distribution() -> Optional[str]:
    """
    This will be coming from our sdist's setup.cfg that we package up, and usually (or
    always?) means it will mean the sdist has been installed in the python environment.

    This is the common case when you've installed graspologic from PyPI
    """
    try:
        return pkg_resources.get_distribution("graspologic").version
    except BaseException:
        return None


def __from_filesystem(setup_config: str = "setup.cfg") -> Optional[str]:
    """
    If we aren't an installed library, pkg_resources.get_distribution() won't be
    able to find setup.cfg's version, so we'll try to look at it from the filesystem
    We can probably presume in this circumstance that we are not a properly installed
    version and thus we're going to mark it with a dev label and a time entry

    This is the common case when you are developing graspologic itself.
    """
    try:
        cfg_parser = configparser.RawConfigParser()
        cfg_parser.read(setup_config)
        base_version = cfg_parser.get("metadata", "version")
        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        return f"{base_version}.dev{now}"
    except BaseException:
        return None


def __version() -> str:
    """
    Distribution takes precedence, but otherwise we try to read it from the filesystem

    If all else fails, we have no way of knowing what version we are
    """
    possible_version = __from_distribution() or __from_filesystem()
    return possible_version if possible_version is not None else "unknown"
