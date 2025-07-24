"""Models to estimate event probabilities."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hmp.distributions import Gamma
from hmp.trialdata import TrialData


class BaseModel(ABC):
    """The model to analyze the raw data.

    Parameters
    ----------
    data : xr.Dataset
        xr.Dataset obtained through the hmp.utils.transform_data() function
    sfreq : float
        (optional) Sampling frequency of the signal if not provided, inferred from the epoch_data
    cpus: int
        How many cpus to use for the functions`using multiprocessing`
    event_width : float
        width of events in milliseconds, by default 50 ms.
    shape: float
        shape of the probability distributions of the by-trial stage onset
        (one shape for all stages)
    template: ndarray
        Expected shape for the transition event used in the cross-correlation,
        should be a vector of values capturing the expected shape over the sampling frequency
        of the data. If None, the template is created as a half-sine shape with a frequency
        derived from the event_width argument
    location : float
        Minimum duration between events in samples. Default is the event_width.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        pattern: Any,
        distribution: Any = None
    ):
        self.pattern = pattern
        if distribution is None:
            distribution = Gamma()
        self.distribution = distribution
        self._fitted = False


    def compute_max_events(self, trial_data: TrialData):
        """Compute the maximum possible number of events given event width minimum reaction time."""
        return int(np.rint(np.percentile(trial_data.durations, 10) // (self.location)))


    def __getattribute__(self, attr):
        if attr in ["sfreq", "steps", "location", "template"]:
            return getattr(self.pattern, attr)
        if attr == "event_width":
            return self.pattern.width
        if attr == "event_width_samples":
            return self.pattern.width_samples

        return super().__getattribute__(attr)

    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")


    @abstractmethod
    def fit(self, trial_data: TrialData):
        ...

    @abstractmethod
    def transform(self, trial_data: TrialData):
        ...

    def fit_transform(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)
        return self.transform(data)
