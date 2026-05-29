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
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        pattern: Pattern = None,
        distribution: Any = None
    ):
        self.pattern = pattern
        # default pattern is HalfSine, 50 ms width
        if pattern is None:
            self.pattern = HalfSine()
        if distribution is None:
            distribution = Gamma()
        self.distribution = distribution
        self._fitted = False

    def __getattribute__(self, attr):
        if attr in ["sfreq", "steps", "template"]:
            return getattr(self.pattern, attr)

        if attr == "event_width":
            return self.pattern.width

        return super().__getattribute__(attr)
    
    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")

    def _instantiate_data_pattern(self, data):
        """ 
        If data is PatternData object, use directly. Otherwise
        create pattern template based on data sfreq, and do
        cross correlation.

        If previously fitted (ie transform()), use existing pattern

        """
        if isinstance(data, PatternData):
            self.pattern_data = data
            if self._fitted and data.pattern != self.pattern:
                warn(f"Cross-correlation pattern {data.pattern} is different in provided data than in model {self.pattern}. Data pattern is used.")
            self.pattern = data.pattern
        else: #assume transformed (is checked later)
            if self.pattern.sfreq is None:
                self.pattern.create_template(data.sfreq)
            self.pattern_data = PatternData.from_transformer(data, self.pattern)

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)
        return self.transform(data)
