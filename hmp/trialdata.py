"""Builds the data to be used in HMP model estimation."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from scipy.signal import correlate

from hmp.utils import _check_transformed


@dataclass
class TrialData:
    """
    A class building trial data and its associated properties to use in the estimations.

    Attributes
    ----------
    xrdurations : xr.DataArray
        Durations of each trial with corresponding trial coordinates.
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
    """

    xrdurations: xr.DataArray
    starts: np.ndarray
    ends: np.ndarray
    n_trials: int
    n_samples: int
    sfreq: float
    offset: int
    cross_corr: np.ndarray


    @classmethod
    def from_transformer(cls, transformed, pattern, dtype = np.float32):
        """
        Create a TrialData instance from transformed data and a given pattern.

        Parameters
        ----------
        transformed : BaseTransfromer or xr.DataArray
            The transformed object or xarray DataArray containing the transformed data.
        pattern : np.ndarray
            The pattern to use for cross-correlation computation.
        dtype: np.DTypeLike
            Precision, use np.float32 or np.int64

        Returns
        -------
        TrialData
            An instance of TrialData with computed durations, cross-correlation, and metadata.
        """
        data = _check_transformed(transformed)
        # compute sequence durations based on number of samples
        durations = (
            data
            .sel(component=0).drop_vars('component')
            .dropna(dim="trial", how="all")
            .groupby("trial")
            .count(dim="sample")
            .cumsum()
            .squeeze()
        )

        dur_dropped_na = durations.dropna("trial")
        starts = np.roll(dur_dropped_na.data, 1)
        starts[0] = 0
        ends = dur_dropped_na.data - 1
        xrdurations = durations.dropna("trial") - durations.dropna(
            "trial"
        ).shift(trial=1, fill_value=0)

        n_trials = durations.trial.count().values
        metadata = (data.sel(component=0, sample=0).sel().drop_vars(['component', 'sample']))
        metadata = {k: v for k, v in metadata.coords.items() if k not in metadata.dims}
        for name, coord in metadata.items():
            if name not in xrdurations.coords:
                xrdurations = xrdurations.assign_coords({name: coord})
        data = data.unstack().stack(all_samples=['participant','epoch','sample']).\
            dropna(dim="all_samples")
        n_dims, n_samples = data.shape
        # Equation 1 in 2024 paper
        cross_corr = cross_correlation(data.values.T, n_trials, n_dims, starts, ends, pattern,
                                       dtype)



        return cls(xrdurations=xrdurations, starts=starts, ends=ends,
                   n_trials=n_trials, n_samples=n_samples, cross_corr=cross_corr,
                   offset=data.offset, sfreq=data.sfreq)

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
        pattern: np.ndarray,
        dtype: DTypeLike,
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
    dtype: np.DTypeLike
        Precision, use np.float32 or np.int64

    Returns
    -------
    np.ndarray
        A 2D ndarray with shape (n_samples, n_components) where each cell contains
        the correlation value with the given pattern.
    """
    events = np.zeros(data.shape, dtype=dtype)
    for trial in range(n_trials):  # avoids confusion of gains between trial
        for dim in np.arange(n_dims):
            events[starts[trial] : ends[trial] + 1, dim] = correlate(
                data[starts[trial] : ends[trial] + 1, dim],
                pattern,
                mode="same",
                method="direct",
            )
    return events
