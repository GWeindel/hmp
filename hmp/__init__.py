"""Software for fitting HMP on EEG/MEG data."""


from . import models
from . import utils
from . import visu
from . import resample
from . import clusters
from . import loocv

__all__ = ["models", "simulations", "utils","visu", "clusters","mcca","loocv"]