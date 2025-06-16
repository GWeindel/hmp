"""Builds the data to be used in HMP model estimation."""
import gc
import itertools
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from itertools import cycle, product
from typing import Any
from warnings import resetwarnings, warn
from hmp.preprocessing import Preprocessing

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import MultiIndex
from scipy.signal import correlate
from scipy.stats import norm as norm_pval


@dataclass
class TrialData:
    """
    A class building trial data and its associated properties to use in the estimations.

    Attributes
    ----------
    named_durations : xr.DataArray
        Durations of each trial with names corresponding to trial indices.
    coords : dict
        Coordinates of the trial data (metadata to keep).
    starts : np.ndarray
        Array of start indices for each trial (usually stimulus onsets position in samples).
    ends : np.ndarray
        Array of end indices for each trial (usually response onsets position in samples)
    n_trials : int
        Total number of trials.
    n_samples : int
        Total number of samples across all trials.
    sfreq : float
        Sampling frequency of the data.
    offset : int
        Offset applied to the data.
    cross_corr : np.ndarray
        Cross-correlation values between the data and a given pattern.
    trial_coords : dict
        Coordinates specific to each trial.
    """
    named_durations: xr.DataArray
    coords: dict
    starts: np.ndarray
    ends: np.ndarray
    n_trials: int
    n_samples: int
    sfreq: float
    offset: int
    cross_corr: np.ndarray
    trial_coords: dict

    @classmethod
    def from_preprocessed(cls, preprocessed, pattern):
        """
        Create a TrialData instance from preprocessed data and a given pattern.

        Parameters
        ----------
        preprocessed : Preprocessing or xr.DataArray
            The preprocessed object or xarray DataArray containing the preprocessed data.
        pattern : np.ndarray
            The pattern to use for cross-correlation computation.

        Returns
        -------
        TrialData
            An instance of TrialData with computed durations, cross-correlation, and metadata.
        """
        if isinstance(preprocessed, Preprocessing):
            data = preprocessed.data
        elif 'component' in preprocessed.dims:
            data = preprocessed
        else:
            raise ValueError(f"preprocessed must be an hmp preprocessed object obtained using hmp.preprocessing")
        # compute sequence durations based on number of samples
        durations = (
            data.unstack()
            .sel(component=0)
            .stack(trial=["participant", "epoch"])
            .dropna(dim="trial", how="all")
            .groupby("trial")
            .count(dim="sample")
            .cumsum()
            .squeeze()
        )

        if durations.trial.count() > 1:
            dur_dropped_na = durations.dropna("trial")
            starts = np.roll(dur_dropped_na.data, 1)
            starts[0] = 0
            ends = dur_dropped_na.data - 1
            named_durations = durations.dropna("trial") - durations.dropna(
                "trial"
            ).shift(trial=1, fill_value=0)
            coords = durations.reset_index("trial").coords
        else:
            dur_dropped_na = durations
            starts = np.array([0])
            ends = np.array([dur_dropped_na.data - 1])
            named_durations = durations
            coords = durations.coords

        n_trials = durations.trial.count().values
        n_samples, n_dims = np.shape(data.data.T)
        cross_corr = cross_correlation(data.data.T, n_trials, n_dims, starts, ends, pattern)  # Equation 1 in 2024 paper
        trial_coords = (
            data.unstack()
            .sel(component=0, sample=0)
            .stack(trial=["participant", "epoch"])
            .dropna(dim="trial", how="all")
            .coords
        )
        return cls(named_durations=named_durations, coords=coords, starts=starts, ends=ends,
                   n_trials=n_trials, n_samples=n_samples, cross_corr=cross_corr,
                   trial_coords=trial_coords, offset=data.offset, sfreq=data.sfreq)

    @cached_property
    def durations(self):
        return self.ends - self.starts + 1

    @property
    def n_dims(self):
        return self.cross_corr.shape[1]

def cross_correlation(
    data: np.ndarray, 
    n_trials: int, 
    n_dims: int, 
    starts: np.ndarray, 
    ends: np.ndarray, 
    pattern: np.ndarray
) -> np.ndarray:
    """Compute the cross-correlation between the data and a given pattern.

    This function calculates the correlation of each sample and the next
    x samples (depending on sampling frequency and event size) with a given pattern.

    Parameters
    ----------
    data : np.ndarray
        2D ndarray with shape (n_samples, n_components).
    n_trials : int
        Number of trials in the data.
    n_dims : int
        Number of dimensions (components) in the data.
    starts : np.ndarray
        Array of start indices for each trial.
    ends : np.ndarray
        Array of end indices for each trial.
    pattern : np.ndarray
        1D array representing the pattern to correlate with.

    Returns
    -------
    np.ndarray
        A 2D ndarray with shape (n_samples, n_components) where each cell contains
        the correlation value with the given pattern.
    """

    events = np.zeros(data.shape)
    for trial in range(n_trials):  # avoids confusion of gains between trial
        for dim in np.arange(n_dims):
            events[starts[trial] : ends[trial] + 1, dim] = correlate(
                data[starts[trial] : ends[trial] + 1, dim],
                pattern,
                mode="same",
                method="direct",
            )
    return events
