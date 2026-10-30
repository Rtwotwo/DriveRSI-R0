"""Lazy legacy module search paths; implementations live only in canonical packages."""

from pathlib import Path


def extend_path(package_name, paths):
    directory = Path(__file__).parent.joinpath(*package_name.split(".")[1:])
    if directory.is_dir() and str(directory) not in paths:
        paths.append(str(directory))
