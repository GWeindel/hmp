"""Models to estimate event probabilities."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from warnings import resetwarnings, warn

from hmp.distributions import Gamma
from hmp.patterns import Pattern, HalfSine
from hmp.patterndata import PatternData

class BaseModel(ABC):
    """The model to analyze the cross-correlated data.

    Parameters
    ----------
    pattern : Pattern
        The pattern and properties to use for cross-correlation. Default is
        half sine with 50 ms width.
    location_ms : float, optional
        How much milliseconds should be censored in the EM() step of model fitting.
        Default is width of the event.
        Shorter values than `width` allow overlap of neighboring events
        but might result in the same event being duplicated in several events.
        Larger values will prevent duplication at the risk of missing neighboring events
        Censoring is done on samples lower or equal to the location,
        thus requesting 50ms at 1000Hz will censor up to 50ms
        Defaults to width of pattern, which is by default 50 ms.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        pattern: Pattern = None,
        location_ms: float = None,
        distribution: Any = None
    ):
        self.pattern = pattern
        # default pattern is HalfSine, 50 ms width
        if pattern is None:
            self.pattern = HalfSine()
        self.location_ms = location_ms
        if location_ms is None:
            self.location_ms = self.pattern.width_ms
        if distribution is None:
            distribution = Gamma()
        self.distribution = distribution
        self._fitted = False

    def __getattribute__(self, attr):
        if attr in ["sfreq", "steps", "template"]:
            return getattr(self.pattern, attr)

        if attr in ["width", "event_width", "width_samples"]:
            return self.pattern.width_samples

        return super().__getattribute__(attr)
    
    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")

    def instantiate_data_pattern_location(self, data):
        """ 
        If data is PatternData object, use directly. Otherwise
        create pattern template based on data sfreq, and do
        cross correlation.

        Next, set location based on sfreq of data.

        If previously fitted (ie transform()), use existing pattern and location.

        """
        if isinstance(data, PatternData):
            self.pattern_data = data
            if self._fitted and not (data.pattern.template == self.pattern.template).all():
                warn(f"Cross-correlation pattern {data.pattern} is different in provided data than in model {self.pattern}. Data pattern is used.")
            self.pattern = data.pattern
        else: #assume transformed (is checked later)
            if self.pattern.sfreq is None:
                self.pattern.create_template(data.sfreq)
            self.pattern_data = PatternData.from_transformer(data, self.pattern)

        #instantiate location in samples based on data frequency
        if not hasattr(self,'location') or self.location is None:
            steps = 1000 / self.sfreq
            self.location = int(np.ceil(self.location_ms / steps))


    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)
        return self.transform(data)
