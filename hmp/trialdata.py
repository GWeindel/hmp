"""Models to estimate event probabilities."""
import gc
import itertools
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from itertools import cycle, product
from typing import Any
from warnings import resetwarnings, warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import MultiIndex
from scipy.signal import correlate
from scipy.stats import norm as norm_pval


@dataclass
class TrialData():
    named_durations: Any
    coords: Any
    starts: Any
    ends: Any
    n_trials: int
    n_samples: int
    cross_corr: Any
    trial_coords: Any

    @classmethod
    def from_standard_data(cls, data, template):

        # compute sequence durations based on number of samples
        durations = (
            data.unstack()
            .sel(component=0)
            .rename({"epochs": "trials"})
            .stack(trial_x_participant=["participant", "trials"])
            .dropna(dim="trial_x_participant", how="all")
            .groupby("trial_x_participant")
            .count(dim="samples")
            .cumsum()
            .squeeze()
        )

        if durations.trial_x_participant.count() > 1:
            dur_dropped_na = durations.dropna("trial_x_participant")
            starts = np.roll(dur_dropped_na.data, 1)
            starts[0] = 0
            ends = dur_dropped_na.data - 1
            named_durations = durations.dropna("trial_x_participant") - durations.dropna(
                "trial_x_participant"
            ).shift(trial_x_participant=1, fill_value=0)
            coords = durations.reset_index("trial_x_participant").coords
        else:
            dur_dropped_na = durations
            starts = np.array([0])
            ends = np.array([dur_dropped_na.data - 1])
            named_durations = durations
            coords = durations.coords

        n_trials = durations.trial_x_participant.count().values
        n_samples, n_dims = np.shape(data.data.T)
        cross_corr = cross_correlation(data.data.T, n_trials, n_dims, starts, ends, template)  # Equation 1 in 2024 paper
        trial_coords = (
            data.unstack()
            .sel(component=0, samples=0)
            .rename({"epochs": "trials"})
            .stack(trial_x_participant=["participant", "trials"])
            .dropna(dim="trial_x_participant", how="all")
            .coords
        )
        return cls(named_durations=named_durations, coords=coords, starts=starts, ends=ends,
                   n_trials=n_trials, n_samples=n_samples, cross_corr=cross_corr,
                   trial_coords=trial_coords)

    @cached_property
    def durations(self):
        return self.ends - self.starts + 1

    @property
    def n_dims(self):
        return self.cross_corr.shape[1]


def cross_correlation(data, n_trials, n_dims, starts, ends, template):
    """Set the correlation between the samples and the pattern.

    This function puts on each sample the correlation of that sample and the next
    x samples (depends on sampling frequency and event size) with a half sine on time domain.

    Parameters
    ----------
    data : ndarray
        2D ndarray with n_samples * components

    Returns
    -------
    events : ndarray
        a 2D ndarray with samples * PC components where cell values have
        been correlated with event morphology
    """
    events = np.zeros(data.shape)
    for trial in range(n_trials):  # avoids confusion of gains between trials
        for dim in np.arange(n_dims):
            events[starts[trial] : ends[trial] + 1, dim] = correlate(
                data[starts[trial] : ends[trial] + 1, dim],
                template,
                mode="same",
                method="direct",
            )
    return events
