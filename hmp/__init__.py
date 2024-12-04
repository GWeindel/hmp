"""Software for fitting HMP on EEG/MEG data."""


from . import models
from . import utils
from . import visu
from . import loocv

__all__ = ["models", "simulations", "utils","visu","mcca","loocv"]