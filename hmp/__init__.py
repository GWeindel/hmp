"""Software for fitting HMP on EEG/MEG data."""

from importlib.metadata import PackageNotFoundError, version

from . import (
    basedata,
    crossvalidation,
    distributions,
    io,
    models,
    patterndata,
    patterns,
    projectors,
    utils,
    visu,
)

try:
    __version__ = version("hmp")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["models", "simulations", "utils", "visu", "io", "patterns", "projectors",
           "patterndata", "basedata", "distributions" ,"crossvalidation", "__version__"]
