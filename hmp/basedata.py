"""Builds the base data object to be used in HMP, including preprocessing and projections.

BaseData is typically initialized through the function 'from_io'.

Option 1, first build an object with from_io(io_data), then apply operations
in the following order:

 hmp_data = hmp.basedata.from_io(io_data)
 hmp_data.crop_reject_epochs(duration_id='response_time')
 hmp_data.project(hmp.projectors.PCA(n_comp=10))
 hmp_data.apply_variance_ops()

Option 2: use the default pipeline with:
 hmp_data = hmp.basedata.default(io_data,duration_id='response_time',n_comp=10)

Includes methods to:
    1. Reject durations whose interval exceeds lower and upper interval limits
       (`min_duration` and `max_duration`).
        Reject epochs whose amplitude exceeds a threshold on any electrode.
        For valid epochs crop the data up to `duration`
        Center the data using samples from baseline up to `duration`.
    2. Project channels to new virtual channel, either based on PCA,
        an arbitrary linear combination of channels,
        or the identity of the channels.
    3. Whiten the components,  standardize the components for each recording (False by Default)
        and standardize each trial's variance (`common_variance`).
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable
from warnings import warn

import numpy as np
import xarray as xr

from .projectors import PCA, Projector
from .utils import _sel_method


@dataclass
class BaseData:
    """
    BaseData class containing all data necessary for estimating HMP models.

    Attributes
    ----------
    data : xr.DataArray
        Data with dimensions [sample, component, trial], coordinates that
        describe the dataset including recording, subject, epoch, and a trial
        MultiIndex, and attributes sfreq and offset. Typically obtained
        through class method 'from_io(..)'.
    """

    data: xr.DataArray

    def crop_reject_epochs(self, duration_id: str = 'response_time',
                           offsets: tuple = (0,0), center: bool = True,
                           min_duration: float = 0, max_duration: float | None = None,
                           reject_amplitude = np.inf, verbose=True):
        """
        Crop and reject epochs, typically before projection.

        duration_id: str, optional
            Name of the variable that contains the trial intervals in the epoch_data
            used for cropping and rejection.
            Default = None
        offsets : tuple, optional
            Seconds of recording to keep before and after end of each epoch duration.
            First value refers to the times taken before epoch center and second value
            to the time kept after end. Should be positive. Used for padding the data
            before crosscorrelation. Adding template width / 2 is recommended.
            If float apply the offsets symmetrically.
            Default = 0
        center : bool
            Whether to use the median to center over all trials and electrodes using
            the samples from start of the epoch (baseline) to duration of the trial.
        min_duration : float, optional
            Minimum duration threshold for keeping epochs.
            Default = 0
        max_duration : float, optional
            Maximum duration threshold for keeping epochs.
            Default = Inf
        reject_amplitude : float, optional
            Amplitude threshold for rejecting noisy epochs.
            Default = Inf
        """
        self._check_order(projected=False)
        if duration_id is not None:
            assert duration_id in self.data.coords, 'duration_id not present in data'
            self.duration_id = duration_id
        if isinstance(offsets, (float,int)):
            offsets = (offsets, offsets)
        if (np.array(offsets) < 0).any():
            raise ValueError('offsets should be positive')
        self.data.attrs.update({
            "offset_start": offsets[0],
            "offset_end": offsets[1],
        })
        self.offsets = offsets
        self.center = center
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.reject_amplitude = reject_amplitude

        if self.max_duration is float('Inf') or self.max_duration is None:
            self.max_duration = int(self.data.sample.max()) / self.data.sfreq\
                                    - offsets[1]
        if self.min_duration == 0 or self.min_duration is None:
            self.min_duration = 1 / self.data.sfreq

        if self.max_duration < 0 or self.min_duration < 0 or \
            self.max_duration < self.min_duration:
            raise ValueError("Threshold for epoch duration cannot be negative.")

        if self.reject_amplitude is None:
            self.reject_amplitude = np.inf

        if self.reject_amplitude < 0:
            warn('Amplitude threshold is used in absolute values.')
            self.reject_amplitude = np.abs(self.reject_amplitude)

        # Check baseline correction if no centering
        if center is False:
            if 0 in self.data.sample:
                bsl = [min(self.data.sample), 0]
                bsl_values = self.data.sel(sample=slice(*bsl)).mean(['trial','sample'])/\
                        self.data.sel(sample=slice(*bsl)).std(['trial','sample'])
                if (bsl_values.values > 0.1).any():
                    warn("Some electrodes might not have been baseline corrected. "
                         "Non-centered data might behave unexpectedly in the next steps. "
                         "Consider baseline correcting the data before using BaseData or "
                         "to use the `center` parameter in basedata.crop_reject_epochs. ")

        self._crop_reject_epochs(verbose)

    def project(self,
                projector: Projector):
        """
        Project data from channels to components.

        projector: Projector
            Module from the projectors class
        """
        self.data = projector.fit_transform(self.data)
        self.data = self.data.transpose('sample','component','trial')
        self.projector = projector

    def apply_variance_ops(self, whiten: bool = True, common_variance: bool = False,
                            standardize_recording: bool = False):
        """
        Apply three variance operators, typically after projection.

        whiten : bool, optional
            Return the components with unit-variance
            Default = True
        common_variance : bool, optional
            Standardize variance across trials.
            Default = False
        standardize_recording: bool, optional
            Divide each component for each recording by its standard deviation
            Default = False
        """
        self._check_order(projected=True)
        self.whiten = whiten
        self.common_variance = common_variance
        self.standardize_recording = standardize_recording
        self._apply_variance_ops()

    def pca_and_variance(self, n_comp: float = None, method_pca: str='svd',
                         whiten=True, common_variance=False, standardize_recording=False,
                         verbose=True):
        """Apply PCA and variance operations."""
        self.project(PCA(n_comp=n_comp, method_pca=method_pca, verbose=verbose))
        self.apply_variance_ops(whiten=whiten, common_variance=common_variance,
                                standardize_recording=standardize_recording)

    def select_coord(self,
                value: object,
                variable: str,
                method: Callable[[xr.DataArray, object], xr.DataArray] = np.equal,
                copy: bool = True
               ):
        """Select a subset from basedata using the specified coordinate(s).

        The function selects trials where `method(data[variable], value)` is True.
        You can either use functions returning booleans or a custom function
        using lambda, e.g. `method=lambda x, v: ~x.isin(v)`

        Parameters
        ----------
        value : str | num
            Value to test with method().
        variable : str
            coordinate present in data that is used for condition selection
        method : callable
            You can use callable resulting in a boolean,
            e.g. 'np.equal', `np.greater` or lambda s, v: s.str.contains(v)
            Method also allows for 'contains' that selects trial in which value
            appears in variable (e.g. 'comp' in 'incompatible' and 'compatible')
        copy : bool
            Whether to return a copy (True, Default) or overwrite the current object (False)

        Returns
        -------
        data : BaseData
            Subset of the provided BaseData object.
        """
        if copy:
            bdata = deepcopy(self)
        else:
            bdata = self
        bdata.data = (_sel_method(bdata.data.unstack(), value, variable, method)
            .stack(trial=["recording", "epoch"]).dropna(dim="trial", how="all"))
        return bdata

    def _apply_variance_ops(self):
        """Apply one or more variance operations."""
        if self.whiten and not self.standardize_recording:
            self.data /= self.data.std(['trial','sample'], skipna=True)
        elif self.standardize_recording:
            self.data = self.data.unstack()
            self.data /= self.data.std(['epoch','sample'], skipna=True)
            self.data = self.data.stack(trial=['recording','epoch'])\
                .dropna("trial", how="all")
        else:
            self.data /= self.data.std(..., skipna=True)

        if self.common_variance:
            self.data /= self.data.std(['component','sample'], skipna=True)


    def _crop_reject_epochs(self, #noqa: PLR0912
                            verbose=True):
        """
        Crop each epoch from time 0 of the epoch to its interval.

        For epoch in the `epoch_data` xr.Dataset, this function trims the epoch data
        from epoch time 0 (e.g. stimulus onset) up to the specified interval (e.g.
        response), optionally including a fixed offset after the interval.

        Optionally also :
        1. rejects trials with duration > `max_duration` or < `min_duration`
        2. rejects the trial if any value in baseline + duration
            exceeds `reject_amplitude`
        3. center the data including baseline.
        """
        rts_arr = self.data.coords[self.duration_id].values.copy()
        rts_arr = self._check_scale_ms(rts_arr, warning=True)
        rts_arr[np.isnan(rts_arr)] = 0  # rejected during epoching or inexistant
        inexistant_dur = len(rts_arr[rts_arr == 0])

        if verbose:
            print(f"Found {len(rts_arr[rts_arr > 0])} trials with positively defined durations "
                f"and {inexistant_dur} trials without durations (0 or nan)")

        # Sample domain
        rts_arr = np.rint(rts_arr * self.data.sfreq).astype(int)
        min_dur = np.rint(self.min_duration * self.data.sfreq).astype(int)
        max_dur = np.rint(self.max_duration * self.data.sfreq).astype(int)
        rts_arr[rts_arr <= min_dur] = 0
        rts_arr[rts_arr > max_dur] = 0
        rt_criteria_rej = len(rts_arr[rts_arr == 0]) - inexistant_dur
        offset_start_samples = -int(np.rint(self.offsets[0] * self.data.sfreq))
        offset_end_samples = int(np.rint(self.offsets[1] * self.data.sfreq))

        if len(rts_arr) == 0:
            raise ValueError(f"No duration left in {self.duration_id}")

        #check nr of samples
        min_rt = min(rts_arr[rts_arr > 0])
        if min_rt < 10:
            if min_rt < 2:
                raise ValueError("Cannot model durations with a single sample")
            warn("The shortest interval is less than 10 samples. "
                "Consider rejecting too short trials using the `min_duration` "
                "parameter or increasing sampling frequency of the signal.")

        # Check given offset start if exceeds available samples
        min_sample = np.min(self.data.sample.values)
        if min_sample > offset_start_samples:
            raise ValueError("Offset before start is too large for the epoch data provided."
                             f"Max is {min_sample/self.data.sfreq} seconds.")

        # threshold based rejection, cropping and median centering
        time0 = np.argmin(np.abs(self.data.sample.values)) #Centering event
        inexistant_rej = 0
        rej = 0
        for i in range(len(self.data.data)):
            #Total sample up to duration, including baseline
            epoch_max_time = time0 + rts_arr[i] + offset_end_samples
            if rts_arr[i] > 0:
                # if doesn't exceeds threshold, including baseline
                if ~(np.abs(self.data.values[i, :, : epoch_max_time])
                            > self.reject_amplitude).any():
                    # Crops the epochs up to duration of trial
                    self.data.values[i, :, epoch_max_time + 1:] = np.nan
                    # Centering, including baseling
                    if self.center:
                        self.data.values[i] -= np.median(
                            self.data.values[i,:, : epoch_max_time], axis=-1, keepdims=True)
                # Means threshold exceeded
                elif ~np.isnan(self.data.values[i, :, time0]).any():
                    self.data.values[i, :, :] = np.nan
                    rej += 1
                # Empty trial, assumes rejected before
                else:
                    self.data.values[i, :, :] = np.nan
                    inexistant_rej += 1
            else:
                self.data.values[i, :, :] = np.nan

        if verbose:
            print()
            print(f"Rejection summary: \n {rej} trials rejected based on threshold of "
             f"{self.reject_amplitude} \n {rt_criteria_rej} trials rejected "
             f" based on duration limit of {self.min_duration, self.max_duration} \n"
             f" {inexistant_rej} with duration but without data (nan at time 0)")

        self.data = self.data.sel(
                sample=slice(offset_start_samples,
                min(
                    int(rts_arr.max() + offset_end_samples),
                    self.data.sample.max().values
                )
            ), drop=True
        )
        self.data = self.data.dropna("trial", how="all")

    @staticmethod
    def _check_scale_ms(rts, warning=True):
        max_rt = np.nanmax(rts)
        if max_rt > 500:
            if warning:
                warn(f"Found intervals with a max value value of {np.round(max_rt,2)}\n,\
                        assuming intervals are in milliseconds and converting to seconds")
            rts /= 1000
        return rts

    def _check_order(self, projected=False):
        if projected is True and 'component' not in self.data.dims:
            raise ValueError('Cannot perform operation on unprojected data. '
                             'Use BaseData.project before calling this method')
        elif projected is False and 'component' in self.data.dims:
            raise ValueError('Previous projection was applied. Use raw epoched data')


def from_io(epoch_data: xr.Dataset) -> BaseData:
    """
    Create a BaseData instance from data from io.

    Parameters
    ----------
    epoch_data : xr.Dataset
        Input EEG data with dimensions [recording, epoch, sample, channel],
        from `io` module

    Returns
    -------
    BaseData
        An instance of BaseData
    """
    base_data = BaseData(data=epoch_data.copy())
    #process data from io
    base_data.data = base_data.data.data.stack(trial=["recording", "epoch"])\
        .dropna("trial", how="all")
    base_data.data = base_data.data.transpose('trial','channel','sample')
    base_data.data.attrs["sfreq"] = epoch_data.sfreq
    return base_data

def default( # noqa: PLR0913, PLR0917
            epoch_data: xr.Dataset,
            duration_id: str = 'response_time',
            offsets: tuple | float = (0,0),
            center: bool = True,
            min_duration: float = 0,
            max_duration: float | None = None,
            reject_amplitude: float = np.inf,
            n_comp: float | None = None,
            whiten: bool = True,
            common_variance: bool = False,
            standardize_recording: bool = False,
            verbose: bool = True
    ):
    """
    Create a BaseData instance from data from io.

    Includes:
     - epoch cropping and rejection
     - PCA
     - variance operations.

    Parameters
    ----------
    epoch_data : xr.DataArray
        Data with dimensions [sample, component, trial], coordinates that
        describe the dataset including recording, subject, epoch, and a trial
        MultiIndex, and attributes sfreq and offset. Typically obtained
        through class method 'from_io(..)'.
    duration_id: str, optional
        Name of the variable that contains the trial intervals in the epoch_data
        used for cropping.
        Default = 'response_time'.
    offsets : tuple, float, optional
        Seconds of recording to keep before and after end of each epoch duration.
        First value refers to the times taken before epoch center and second value
        to the time kept after end. Should be positive. Used for padding the data
        before crosscorrelation. Adding template width / 2 is recommended.
        If float apply the offsets symmetrically.
        Default = 0
    center : bool
        Median center the data after cropping including baseline
        default = False
    min_duration : float, optional
        Minimum duration threshold for keeping epochs.
        Default = 0
    max_duration : float, optional
        Maximum duration threshold for keeping epochs.
        Default = Inf
    reject_amplitude : float, optional
        Amplitude threshold for rejecting noisy epochs.
        Default = None
    n_comp: int, optional
        Nr of components retained if > 1, otherwise (0 < n_comp < 1) nr of components
        explaining at least n_comp% variance are retained.  If None, user input requested.
        Default = None
    whiten : bool, optional
        Return the components with unit-variance
        Default = True
    common_variance : bool, optional
        Standardize variance across trials.
        Default = False
    standardize_recording: bool, optional
        Divide each component for each recording by its standard deviation
        Default = False
    verbose:
        Provide feedback on the different operations

    Returns
    -------
    BaseData
        An instance of BaseData using default preprocessing routine
    """
    base_data = from_io(epoch_data)

    base_data.crop_reject_epochs(
        duration_id=duration_id,
        offsets=offsets,
        center=center,
        min_duration=min_duration,
        max_duration=max_duration,
        reject_amplitude=reject_amplitude,
        verbose=verbose
    )

    base_data.project(PCA(n_comp=n_comp))

    base_data.apply_variance_ops(
        whiten=whiten,
        common_variance=common_variance,
        standardize_recording=standardize_recording,
    )

    return base_data


def _check_basedata(base_data):
    if isinstance(base_data, BaseData):
        data = base_data.data
    elif 'component' in base_data.dims:
        data = base_data
    else:
        raise ValueError("base_data must be an hmp base_data object")
    return data
