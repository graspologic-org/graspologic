"""Utilities for exposing the package version."""

from importlib import metadata


def __version() -> str:
    """Return the installed graspologic version.

    Falls back to a sensible default when the distribution metadata is
    unavailable (for example when running directly from a source checkout).
    """

    try:
        return metadata.version("graspologic")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = __version()