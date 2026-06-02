"""Software for fitting HMP on EEG/MEG data."""

from importlib.metadata import PackageNotFoundError, version

from . import crossvalidation, distributions, io, models, patterns, transformers, utils, visu

try:
    __version__ = version("hmp")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["models", "simulations", "utils", "visu", "io", "transformers", "patterns",
           "distributions" ,"crossvalidation", "__version__"]
