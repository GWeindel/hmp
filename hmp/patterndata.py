"""Builds the data to be used in HMP model estimation."""
from dataclasses import dataclass
from warnings import warn

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from scipy.signal import correlate

from hmp.basedata import BaseData, _check_basedata
from hmp.patterns import HalfSine, Pattern


@dataclass
class PatternData:
    """
    A class building trial data and its associated properties to use in the estimations.

    Attributes
    ----------
    durations : xr.DataArray
        Durations of each trial with corresponding trial coordinates.
    starts : np.ndarray
        Array of start indices for each trial (usually stimulus onsets position in samples).
    ends : np.ndarray
        Array of end indices for each trial (usually response onsets position in samples)
    sfreq : float
        Sampling frequency of the data.
    pattern : np.ndarray
        Values for the pattern used for the cross-correlation.
    cross_corr : np.ndarray
        Cross-correlation values between the data and a given pattern.
    """

    durations: xr.DataArray
    starts: np.ndarray
    ends: np.ndarray
    sfreq: float
    pattern: Pattern
    template: np.ndarray
    cross_corr: np.ndarray

    @classmethod
    def from_basedata(cls,
                    base_data: xr.DataArray | BaseData,
                    pattern: Pattern | None = None,
                    dtype: DTypeLike | None = None):
        """
        Create a TrialData instance from preprocessed data and a given pattern.

        Parameters
        ----------
        base_data : BaseData or xr.DataArray
            BaseData object or xarray DataArray containing the preprocessed data.
        pattern : Pattern
            The pattern to use for cross-correlation computation. Default is
            half sine with 50 ms width.
        dtype: np.DTypeLike
            Precision, use np.float32 or np.int64. By default inherits from data.

        Returns
        -------
        PatternData
            An instance of PatternData with computed durations, cross-correlation, and metadata.
        """
        data = _check_basedata(base_data)
        if dtype is None:
            dtype = base_data.data.dtype
        # compute sequence durations based on number of samples
        durations = (
            data
            .sel(component=0).drop_vars('component')
            .dropna(dim="trial", how="all")
            .groupby("trial")
            .count(dim="sample")
            .cumsum()
            .squeeze()
        ).dropna("trial")

        starts = np.roll(durations.data, 1)
        starts[0] = 0
        ends = durations.data - 1
        durations -= durations.shift(trial=1, fill_value=0)

        metadata = (data.sel(component=0, sample=0).sel().drop_vars(['component', 'sample']))
        metadata = {k: v for k, v in metadata.coords.items() if k not in metadata.dims}
        for name, coord in metadata.items():
            if name not in durations.coords:
                durations = durations.assign_coords({name: coord})
        data = data.unstack().stack(all_samples=['recording','epoch','sample']).\
            dropna(dim="all_samples")

        if pattern is None:
            pattern = HalfSine()

        if data.sfreq > 1000:
            raise NotImplementedError('Cannot use sfreq > 1000Hz')

        template = _norm_template(data.sfreq, pattern.template)
        if len(template) < 5:
            if len(template) < 2:
                raise ValueError("Cannot use pattern with only one data point")
            warn('Using a pattern defined by less than 5 points is not recommended')

        # Equation 1 in 2024 paper
        cross_corr = cross_correlation(data.values.T, starts, ends, template, dtype)

        return cls(durations=durations, starts=starts, ends=ends,
                    cross_corr=cross_corr, pattern=pattern, template=template,
                    sfreq=data.sfreq)

def _norm_template(sfreq, template):
    tstep = int(np.rint(1000/sfreq))
    template = template[::tstep]
    template = template / np.sum(template**2)
    return template

def cross_correlation(
        data: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        template: np.ndarray,
        dtype: DTypeLike,
    ) -> np.ndarray:
    """Compute the cross-correlation between the data and a given pattern.

    This function calculates the correlation of each sample and the next
    x samples (depending on sampling frequency and event size) with a given pattern.

    Parameters
    ----------
    data : np.ndarray
        2D ndarray with shape (n_samples, n_components).
    n_dims : int
        Number of dimensions (components) in the data.
    starts : np.ndarray
        Array of start indices for each trial.
    ends : np.ndarray
        Array of end indices for each trial.
    template : np.ndarray
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
    for trial in range(len(starts)):  # avoids confusion of gains between trial
        for dim in np.arange(data.shape[1]):
            events[starts[trial] : ends[trial] + 1, dim] = correlate(
                data[starts[trial] : ends[trial] + 1, dim],
                template,
                mode="same",
                method="direct",
            )
    return events
