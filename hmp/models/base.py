"""Models to estimate event probabilities."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hmp.distributions import Gamma
from hmp.patterns import Pattern, HalfSine

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
        if attr in ["sfreq", "steps", "location", "template"]:
            return getattr(self.pattern, attr)

        if attr in ["width", "event_width", "width_samples"]:
            return self.pattern.width_samples

        return super().__getattribute__(attr)
    
    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)
        return self.transform(data)
