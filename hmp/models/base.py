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
    data : Data to fit the model on. One of two options:
            1. data from BaseTransformer or xr.DataArray containing transformed data.
            2. TrialData object.
            In case of option 1, data is cross-correlated with the pattern in event_properties.
            If event_properties is None, a half sine with 50 ms width is used.
    event_properties :
        The pattern and properties to use for cross-correlation. Default is
        half sine with 50 ms width.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        data: Any,
        event_properties = None, 
        distribution: Any = None
    ):
        
        if isinstance(data, TrialData):
            self.trial_data = data
        else: #assume transformed (is checked later)
            self.trial_data = TrialData.from_transformer(data, event_properties)

        if distribution is None:
            distribution = Gamma()
        self.distribution = distribution
        self._fitted = False

    def __getattribute__(self, attr):
        if attr in ["sfreq", "steps", "location", "template", "width"]:
            return getattr(self.trial_data.event_properties, attr)

        if attr == "event_width":
            return self.trial_data.event_properties.width

        return super().__getattribute__(attr)
    
    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")

    def compute_max_events(self):
        """Compute the maximum possible number of events given location and minimum duration."""
        return int(np.rint(np.min(self.trial_data.durations.values) // (self.location)))
    
    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform()
