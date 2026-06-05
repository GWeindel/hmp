"""Models to estimate event probabilities."""
from abc import ABC, abstractmethod
from typing import Any
from warnings import warn

import numpy as np

from hmp.distributions import Gamma
from hmp.patterndata import PatternData
from hmp.patterns import HalfSine, Pattern


class BaseModel(ABC):
    """The model to analyze the cross-correlated data.

    Parameters
    ----------
    pattern : Pattern
        The pattern and properties to use for cross-correlation. Default is
        half sine with 50 ms width.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        pattern: Pattern = None,
        distribution: Any = None
    ):
        # default pattern is HalfSine, 50 ms width
        if pattern is None:
            pattern = HalfSine(width=50)
        self.pattern = pattern

        if distribution is None:
            distribution = Gamma()
        self.distribution = distribution
        self._fitted = False

    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")

    def _instantiate_data_pattern(self, data):
        """Load data pattern, cross-correlate if needed.

        If data is PatternData object, use directly. Otherwise
        create pattern template and do cross correlation.
        If previously fitted (ie transform()), use existing pattern

        """
        if isinstance(data, PatternData):
            pattern_data = data
            if self._fitted and data.pattern != self.pattern:
                warn(f"Cross-correlation pattern {data.pattern}is different in provided data "
                     f"than in model {self.pattern}. Data pattern is used.")
            self.pattern = data.pattern
        else: #assume transformed (is checked later)
            pattern_data = PatternData.from_transformer(data, self.pattern)
        return pattern_data

    def _time_to_samples(self, time, sfreq):
        """Calculate samples (int) based on time(s).

        Parameters
        ----------
        time : float | ndarray
            Time or times that need to be converted to samples.
        sfreq : sample frequency of data
        """
        return np.rint(time * sfreq / 1000).astype(int)

    def _compute_max_events(self, pattern_data : PatternData, location : float):
        """Compute max nr of events that fit in trial.

        Parameters
        ----------
        pattern_data : Pattern
            PatternData object
        location : float
            Location in ms.
        """
        min_dur = np.min(pattern_data.durations.values)
        location_samples = self._time_to_samples(location, pattern_data.sfreq)
        if self.pattern.width < location:
            return int(np.floor((min_dur - self.pattern.width)/ \
                            location_samples)) + 1
        else:
            return int(np.floor(min_dur / location_samples))

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)

        cpus = kwargs['cpus'] if 'cpus' in kwargs else 1
        return self.transform(data, cpus=cpus)
