"""Models to estimate event probabilities."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hmp.distributions import Gamma
from hmp.trialdata import TrialData


class BaseModel(ABC):
    """The model to analyze the cross-correlated data.

    Parameters
    ----------
    data : xr.Dataset
        xr.Dataset obtained through the hmp.utils.transform_data() function
    sfreq : float
        (optional) Sampling frequency of the signal if not provided, inferred from the epoch_data
    cpus: int
        How many cpus to use for the functions`using multiprocessing`
    event_width : int
        Width of the pattern defining events in samples.
    shape: float
        shape of the probability distributions of the by-trial stage onset
        (one shape for all stages)
    location : int
        Minimum duration between events in samples. Default is the event_width.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        distribution: Any = None
    ):
        if distribution is None:
            distribution = Gamma()
        self.distribution = distribution
        self._fitted = False

    def __getattribute__(self, attr):
        if attr in ["sfreq", "steps", "location", "template", "width"]:
            return getattr(self.event_properties, attr)

        if attr == "event_width":
            return self.event_properties.width

        return super().__getattribute__(attr)

    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")

    def compute_max_events(self, trial_data: TrialData ):
        """Compute the maximum possible number of events given location and minimum duration."""
        return int(np.rint(np.min(trial_data.durations.values) // (self.location)))
    
    @abstractmethod
    def fit(self, trial_data: TrialData):
        ...

    @abstractmethod
    def transform(self, trial_data: TrialData):
        ...

    def fit_transform(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)
        return self.transform(data)
