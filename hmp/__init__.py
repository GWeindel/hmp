"""Software for fitting HMP on EEG/MEG data."""

from importlib.metadata import PackageNotFoundError, version

from . import distributions, io, loocv, models, patterns, transformers, utils, visu

try:
    __version__ = version("hmp")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["models", "simulations", "utils", "visu", "io", "transformers", "patterns",
           "distributions" ,"mcca", "loocv", "__version__"]
