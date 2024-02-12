"""Software for fitting HMP on EEG/MEG data."""


from . import models
from . import utils
from . import visu
from . import resample
from . import clusters

__all__ = ["models", "simulations", "utils","visu", "clusters"]