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

        if data.sfreq > 1000:
            raise NotImplementedError('Cannot use sfreq > 1000Hz')

        offset_start = int(np.rint(data.offset_start * data.sfreq))
        offset_end = int(np.rint(data.offset_end * data.sfreq))

        # Crosscorrelating data and pattern
        if pattern is None:
            pattern = HalfSine()

        # Downsample and normalize the template
        template = _adjust_template(data.sfreq, pattern)

        # Equation 1 in 2024 paper
        cross_corr, durations = cross_correlation(data.values, template,
                                       offset_start, offset_end)
        cross_corr = cross_corr.astype(dtype)

        # Formatting durations with metadata, starts and ends
        boundaries = durations.cumsum().astype(int)
        # boundaries = boundaries[np.r_[True, np.diff(boundaries) > 1]]
        starts = np.roll(boundaries, 1)
        starts[0] = 0
        ends = boundaries - 1

        # Preserves indexing:
        duplicate_coords = data.isel(component=0, sample=0).drop_vars(['component','sample'])
        durations = xr.DataArray(
            durations,
            coords=duplicate_coords.coords,
            dims=duplicate_coords.dims,
        )
        durations = durations.dropna('trial')
        durations.attrs["sfreq"] = data.sfreq
        return cls(durations=durations, starts=starts, ends=ends,
                    cross_corr=cross_corr, pattern=pattern, template=template,
                    sfreq=data.sfreq)

def _adjust_template(sfreq, pattern):
    template = pattern.template
    tstep = int(np.rint(1000/sfreq))
    if isinstance(pattern, HalfSine):
        # Ensure closest symmetry
        offset = ((pattern.width - 1) // 2) % tstep
    else:
        # ideal in most other cases?
        offset = tstep // 2
    template = template[offset::tstep]
    template = template / np.sum(template**2)
    if len(template) < 3:
        raise ValueError("Cannot use pattern with only two data points")
    return template

def cross_correlation(
        data: np.ndarray,
        template: np.ndarray,
        offset_start: int,
        offset_end: int
    ) -> np.ndarray:
    """Compute the cross-correlation between the data and a given pattern.

    This function calculates the correlation of each sample and the next
    x samples (depending on sampling frequency and event size) with a given pattern.
    It uses the "same" mode of the scipy.signal.correlate function which is OK if
    baseline + offset end, if not it's not the worst strategy assuming centered signal

    Parameters
    ----------
    data : np.ndarray
        2D ndarray with shape (n_samples, n_components).
    template : np.ndarray
        1D array representing the pattern to correlate with.
    offset_start: int
        Samples before duration start in the data, used to pad before crosscorrelation
    offset_end: int
        Samples after duration end in the data, used to pad before crosscorrelation

    Returns
    -------
    crossc: np.ndarray
        A 2D ndarray with shape (n_samples * n_trials, n_components) where each cell
        contains the correlation value of the component time serie with the given pattern.
    """
    n_samples, n_dims, n_trials = data.shape
    durations = np.zeros(n_trials, int)
    crossc = np.zeros([n_samples*n_trials, n_dims])*np.nan

    min_offset = np.ceil(len(template)/2)
    if offset_start < min_offset or offset_end < min_offset:
        warn("Data was not padded, distortion in the crosscorrelation at the edge of the trials "
             "is likely. Use the offsets argument in basedata with offsets of at least "
             f"{int(min_offset)} samples given sampling frequency")

    for trial in range(n_trials):
        # Identify nan (samples outside of duration)
        mask = ~np.isnan(data[:, 0, trial])
        # Check for empty trials
        if sum(mask)>0:
            trial_data = np.zeros((sum(mask),n_dims))
            for dim in range(n_dims):
                # Cross correlation uses same mode, open to discussion see docstrings
                trial_data[:, dim] = correlate(
                    data[mask, dim, trial],
                    template,
                    mode="same",
                    method="direct",#Expect short template, no FFT
                )

            # Remove offsets used for crosscorrelation
            trial_data[:offset_start, :] = np.nan
            if offset_end > 0:
                trial_data[-offset_end:, :] = np.nan
            trial_data = trial_data[~np.isnan(trial_data[:,0]), :]
            # compute sequence durations based on number of samples
            t_position = sum(durations[:trial])
            t_duration = len(trial_data)
            crossc[t_position:t_position+t_duration] = trial_data
            durations[trial] = t_duration

    crossc = crossc[~np.isnan(crossc[:,0])]
    return crossc, durations
