"""Software for fitting HMP on EEG/MEG data."""

from importlib.metadata import PackageNotFoundError, version

from . import (
    basedata,
    crossvalidation,
    distributions,
    estimators,
    io,
    loocv,
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


__all__ = ["basedata", "patterndata", "projectors", "loocv", "models",
           "simulations", "utils", "visu", "io", "patterns",
           "distributions" ,"crossvalidation", "estimators",
           "__version__"]
