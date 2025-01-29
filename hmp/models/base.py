"""Models to estimate event probabilities."""
import gc
import itertools
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import cycle, product
from typing import Any
from warnings import resetwarnings, warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import MultiIndex
from scipy.signal import correlate
from scipy.stats import norm as norm_pval

from hmp.trialdata import TrialData

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm


@dataclass
class EventProperties():
    sfreq: float
    steps: float
    shape: float
    width: int
    width_samples: int
    location: int
    template: Any

    @classmethod
    def from_standard_data(cls, data, sfreq=None, shape=2, width=50, template=None, location=None):

        if sfreq is None:
            sfreq = data.sfreq
        steps = 1000 / sfreq
        shape = float(shape)

        width_samples = int(np.round(width / steps))
        if location is None:
            location = int(width / steps)
        else:
            location = int(np.rint(location))

        # Use or compute the template
        if template is None:
            template = cls._create_template(width_samples, steps, width)

        return cls(sfreq, steps, shape, width, width_samples, location, template)


    @staticmethod
    def _create_template(width_samples, steps, width):
        """Compute the event shape.

        Computes the template of a half-sine (event) with given frequency f and sampling frequency.

        Equations in section 2.4 in the 2024 paper
        """
        event_idx = np.arange(width_samples) * steps + steps / 2
        # gives event frequency given that events are defined as half-sines
        event_frequency = 1000 / (width * 2)

        # event morph based on a half sine with given event width and sampling frequency
        template = np.sin(2 * np.pi * event_idx / 1000 * event_frequency)
        template = template / np.sum(template**2)  # Weight normalized
        return template




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
        trial_data: TrialData,
        event_properties: EventProperties,
        distribution: str = "gamma",
    ):

        match distribution:
            case "gamma":
                from scipy.stats import gamma as sp_dist

                from hmp.utils import _gamma_mean_to_scale, _gamma_scale_to_mean

                self.scale_to_mean, self.mean_to_scale = _gamma_scale_to_mean, _gamma_mean_to_scale
            case "lognormal":
                from scipy.stats import lognorm as sp_dist

                from hmp.utils import _logn_mean_to_scale, _logn_scale_to_mean

                self.scale_to_mean, self.mean_to_scale = _logn_scale_to_mean, _logn_mean_to_scale
            case "wald":
                from scipy.stats import invgauss as sp_dist

                from hmp.utils import _wald_mean_to_scale, _wald_scale_to_mean

                self.scale_to_mean, self.mean_to_scale = _wald_scale_to_mean, _wald_mean_to_scale
            case "weibull":
                from scipy.stats import weibull_min as sp_dist

                from hmp.utils import _weibull_mean_to_scale, _weibull_scale_to_mean

                self.scale_to_mean, self.mean_to_scale = (
                    _weibull_scale_to_mean,
                    _weibull_mean_to_scale,
                )
            case _:
                raise ValueError(f"Unknown Distribution {distribution}")
        self.distribution = distribution
        self.trial_data = trial_data
        self.events = event_properties
        self.pdf = sp_dist.pdf

    def compute_max_events(self):
        """Compute the maximum possible number of events given event width minimum reaction time."""
        return int(np.rint(np.percentile(self.durations, 10) // (self.location)))

    def gen_random_stages(self, n_events):
        """Compute random stage duration.

        Returns random stage duration between 0 and mean RT by iteratively drawind sample from a
        uniform distribution between the last stage duration (equal to 0 for first iteration) and 1.
        Last stage is equal to 1-previous stage duration.
        The stages are then scaled to the mean RT

        Parameters
        ----------
        n_events : int
            how many events

        Returns
        -------
        random_stages : ndarray
            random partition between 0 and mean_d
        """
        mean_d = int(self.mean_d)
        rnd_durations = np.zeros(n_events + 1)
        while any(rnd_durations < self.event_width_samples):  # at least event_width
            rnd_events = np.random.default_rng().integers(
                low=0, high=mean_d, size=n_events
            )  # n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, mean_d)) - np.hstack(
                (0, rnd_events)
            )  # associated durations
        random_stages = np.array(
            [[self.shape, self.mean_to_scale(x, self.shape)] for x in rnd_durations]
        )
        return random_stages

    def __getattribute__(self, attr):
        if attr in ["sfreq", "steps", "shape", "location", "template"]:
            return getattr(self.events, attr)
        if attr == "event_width":
            return self.events.width
        if attr == "event_width_samples":
            return self.events.width_samples

        if attr in ["named_durations", "coords", "starts", "ends", "n_trials", "n_samples",
                    "n_dims", "trial_coords", "max_duration", "mean_duration",
                    "durations"]:
            return getattr(self.trial_data, attr)

        if attr == "mean_d":
            return self.trial_data.mean_duration
        if attr == "max_d":
            return self.trial_data.max_duration
        if attr == "crosscorr":
            return self.trial_data.cross_corr

        return super().__getattribute__(attr)
