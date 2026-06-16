"""Software for fitting HMP on EEG/MEG data."""

from importlib.metadata import PackageNotFoundError, version

from . import crossvalidation, distributions, io, models, patterns, preprocessors, utils, visu, loocv

try:
    __version__ = version("hmp")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["models", "simulations", "utils", "visu", "io", "preprocessors", "patterns",
           "distributions" ,"crossvalidation", "loocv", "__version__"]
