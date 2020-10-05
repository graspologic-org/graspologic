# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
from typing import List
import pkg_resources

__all__: List[str] = ["version", "name"]

name = "graspologic"

# manually updated
__semver = "0.1.0"
#  full version (may be same as __semver on release)
__version_file = "version.txt"


def _from_resource() -> str:
    version_file = pkg_resources.resource_stream(__name__, __version_file)
    version_file_contents = version_file.read()
    return version_file_contents.decode("utf-8").strip()


def local_build_number() -> str:
    return datetime.datetime.today().strftime("%Y%m%d%H%M%S")


def get_version() -> str:
    version_file_contents = _from_resource()
    if len(version_file_contents) == 0:
        return f"{__semver}.dev{local_build_number()}"
    else:
        return version_file_contents


version = get_version()
