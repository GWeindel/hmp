"""Builds the base data object to be used in HMP, including preprocessing and projections.

BaseData is typically initialized through class method 'from_io', which ensures all
attributes are set correctly.

Option 1, specify all information in the 'from_io' call:

    preprocessed = hmp.basedata.BaseData.from_io(io_data, interval_id='RT', crop=True,
                    reject=True, apply_variance=True, projection_type='pca',
                    projection_kwargs={'n_comp': 10})

Option 2, first build an object with from_io(io_data), then apply operations
in the following order:

    preprocessed = hmp.basedata.BaseData.from_io(io_data)
    preprocessed.crop_epochs(interval_id='RT')
    preprocessed.reject_epochs()
    preprocessed.pca(n_comp=10)
    preprocessed.apply_variance_ops()

Includes methods to:
    1. Crop the data from epoch time 0 (e.g. stimulus) up to a specified interval
       (e.g. response), optionally including a fixed offset after the interval.
    2. Reject epochs whose interval exceeds lower and upper interval limits
       (`min_duration` and `max_duration`) or amplitude exceeds a threshold
    3. Center the data
    4. Project channels to new virtual channel, either based on PCA,
        an arbitrary linear combination of channels,
        or the identity of the channels.
    5. Whiten the components and standardize each trial's variance
       (`common_variance`) and standardize the components for each participant.
"""

import copy
from dataclasses import dataclass
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.linalg import eigh

import hmp


@dataclass
class BaseData:
    """
    BaseData class containing all data necessary for estimating HMP models.

    Note that __init__ does not perform any of the operations!

    Attributes
    ----------
    data : xr.DataArray
        Data with dimensions [sample, component, trial], coordinates that
        describe the dataset including recording, subject, epoch, and a trial
        MultiIndex, and attributes sfreq and offset. Typically obtained
        through class method 'from_io(..)'.
    weights : xr.DataArray
        DataArray with dimensions [channel, component] and coordinates
        channel (Fp1, CPz, ..) and component (0, 1, ..).
        Default = None
    interval_id: str, optional
        Name of the variable that contains the trial intervals in the epoch_data
        used for cropping and rejection.
        Default = 'rt'.

    #Cropping parameters used
    crop : bool, optional
        Crop data based on the duration and parameters below.
        Default = False
    offset_start : float, optional
        Time offset from interval start for cropping. Negative number extends
        epoch before start.
        Default = 0
    offset_end : float, optional
        Time offset after interval end for cropping.
        Default = 0

    #Epoch rejection parameters used
    reject : bool, optional
        Reject epochs based on parameters below.
        Default = False
    min_duration : float, optional
        Minimum duration threshold for keeping epochs.
        Default = 0
    max_duration : float, optional
        Maximum duration threshold for keeping epochs.
        Default = Inf
    reject_amplitude : float, optional
        Amplitude threshold for rejecting noisy epochs.
        Default = None

    #Projection parameters used
    projection_type: str, optional
        Type of projection to apply. 'pca' or 'custom' or None (identity)
        Default = None
    n_comp: int, optional
        Nr of components retained if > 1, otherwise (0 < n_comp < 1) nr of components
        explaining at least n_comp% variance are retained.  If None, user input requested.
        Default = None
    center : bool, optional
        Whether to center the data within each epoch for each channel before projection
        Default = True
    method_pca: str, optional
        Perform PCA ('pca') or SVD decomposition ('svd') for PCA
        Default = 'svd'

    #Variance parameters used
    whiten : bool, optional
        Return the components with unit-variance
        Default = True
    common_variance : bool, optional
        Standardize variance across trials.
        Default = True
    subject_zscore: bool, optional
        z-score each component for each participant
        Default = True

    """

    data: xr.DataArray
    weights: xr.DataArray = None
    interval_id: str = 'rt'

    #cropping
    crop: bool = False
    offset_start: float = 0
    offset_end: float = 0

    #rejection
    reject: bool = False
    min_duration: float = 0
    max_duration: float = float('Inf')
    reject_amplitude: float = None

    #projection
    projection_type: str = None
    n_comp: float = None
    center: bool = True
    method_pca: str='svd'

    #variance
    whiten: bool = True
    common_variance: bool = True
    subject_zscore: bool = True



    ##Classmethods

    @classmethod
    def from_io( # noqa: PLR0912, PLR0913
                cls, epoch_data: xr.Dataset, weights: xr.DataArray = None,
                interval_id: str = 'rt',
                crop: bool = False, crop_kwargs: dict = None,
                reject: bool = False, reject_kwargs: dict = None,
                projection_type: str = None, projection_kwargs: dict = None,
                apply_variance: bool = False, variance_kwargs: dict = None,
                verbose: bool = True):
        """
        Create a BaseData instance from data from io.

        Optionally performing data cropping, epoch rejection, component projection (PCA)
        and equalizing variance between subjects.

        Parameters
        ----------
        epoch_data : xr.Dataset
            Input EEG data with dimensions [participant, epoch, sample, channel],
            from `io` module
        weights : xr.DataArray with dimensions [channel, component] and coordinates
            channel (Fp1, CPz, ..) and component (0, 1, ..).
            Default = None
        interval_id: str, optional
            Name of the variable that contains the trial intervals in the epoch_data
            used for cropping and rejection.
            Default = 'rt'.
        crop : bool, optional
            crop data based on parameters in crop_kwargs
            Default = False
        crop_kwargs : dict, optional
            specify one or more of offset_start and offset_end
            default = {offset_start: 0, offset_end: 0)
        reject : bool, optional
            reject epochs based on parameters in reject_kwargs
        reject_kwargs: dict, optional
            specify one or more of min_duration, max_duration and reject_amplitude
            default = {min_durations: 0, max_duration: Inf, reject_amplitude: None)
        projection_type : str, optional
            Type of projection to apply. 'pca' or 'custom' or None (identity) based
            on parameters in projection_kwargs.
            Default = None
        projection_kwargs : dict, optional
            specify one or more of n_comp, center and method_pca
            default = {n_comp: None, center: True, method_pca: 'svd'}
        apply_variance : bool, optional
            Apply whitening, subject_zscores and or common_variance.
            Default = False
        variance_kwargs : dict, optional
            specify one or more of whiten, common_variance and subject_zscore.
            default = {whiten: True, common_variance: True, subject_zscore: 'svd'}
        verbose: bool, optional
            Default = True

        Returns
        -------
        BaseData
            An instance of BaseData with all attributes filled.
        """
        #init defaults
        base_data = cls(data = epoch_data.copy(), weights=weights, interval_id=interval_id)

        #process data from io
        base_data._format_io_data()

        #if crop or reject, check for ms
        if crop or reject:
            rts_arr = base_data.data.coords[base_data.interval_id].values.copy()
            hmp.basedata.BaseData._check_scale_ms(rts_arr)

        #crop data
        if crop:
            if crop_kwargs is not None:
                for key, value in crop_kwargs.items():
                    setattr(base_data, key, value)
            base_data._crop_epochs(verbose=verbose)

        #reject epochs
        if reject:
            if reject_kwargs is not None:
                for key, value in reject_kwargs.items():
                    setattr(base_data, key, value)
            base_data._reject_epochs(verbose=verbose)

        #center
        if projection_kwargs is not None and 'center' in projection_kwargs:
            base_data.center = projection_kwargs['center']

        if base_data.center:
            base_data._center_data()

        #apply projection
        base_data.projection_type = projection_type
        if projection_kwargs is not None:
            for key, value in projection_kwargs.items():
                setattr(base_data, key, value)
        base_data._project()

        #variance operators
        if apply_variance:
            if variance_kwargs is not None:
                for key, value in variance_kwargs.items():
                    setattr(base_data, key, value)
            base_data._apply_variance_ops()

        #format data
        base_data._format_data()
        return base_data

    #shortcuts from_io
    @staticmethod
    def from_io_all_pca( # noqa: PLR0913
                epoch_data: xr.Dataset, weights: xr.DataArray = None,
                interval_id: str = 'rt', offset_start: float = 0, offset_end: float = 0,
                min_duration: float = 0, max_duration: float = float('Inf'),
                reject_amplitude: float = None, n_comp: float = None, center: bool = True,
                method_pca: str='svd', whiten: bool = True, common_variance: bool = True,
                subject_zscore: bool = True):
        """
        Create a BaseData instance from data from io.

        Includes:
         - epoch cropping
         - epoch rejection
         - PCA
         - variance operations.
        All parameters are specified directly. See from_io(..) and BaseData
        for details on the parameters.
        """
        return hmp.basedata.BaseData.from_io(epoch_data, weights, interval_id,
                        crop = True, crop_kwargs = {'offset_start': offset_start,
                                                   'offset_end': offset_end},
                        reject = True, reject_kwargs = {'min_duration': min_duration,
                                                        'max_duration': max_duration,
                                                        'reject_amplitude': reject_amplitude},
                        projection_type = 'pca',
                        projection_kwargs = {'n_comp': n_comp, 'center': center,
                                             'method_pca': method_pca},
                        apply_variance = True,
                        variance_kwargs = {'whiten': whiten, 'common_variance': common_variance,
                                          'subject_zscore': subject_zscore})

    ## Public functions

    def center_data(self):
        """Center data per epoch, typically before projection."""
        #warn of previous projection applied
        if self.projection_type != 'identity':
            warn("Projection applied previously, this operation might have\n \
                unintended consequences.")

        self._unformat_data()
        self._center_data()
        self._format_data()

    def apply_variance_ops(self, whiten: bool = True, common_variance: bool = True,
                            subject_zscore: bool = True):
        """
        Apply three variance operators, typically after projection.

        whiten : bool, optional
            Return the components with unit-variance
            Default = True
        common_variance : bool, optional
            Standardize variance across trials.
            Default = True
        subject_zscore: bool, optional
            z-score each component for each participant
            Default = True
        """
        self.whiten = whiten
        self.common_variance = common_variance
        self.subject_zscore = subject_zscore
        self._apply_variance_ops()
        self._format_data()

    def crop_epochs(self, interval_id = None, offset_start: float = 0, offset_end: float = 0,
                    verbose=True):
        """
        Crop epochs, typically before projection.

        interval_id: str, optional
            Name of the variable that contains the trial intervals in the epoch_data
            used for cropping and rejection.
            Default = None = use existing in self.interval_id.
        offset_start : float, optional
            Time offset from interval start for cropping. Negative number extends
            epoch before start.
            Default = 0
        offset_end : float, optional
            Time offset after interval end for cropping.
            Default = 0
        """
        #warn of previous projection applied
        if self.projection_type != 'identity':
            warn("Projection applied previously, this operation might have\n \
                unintended consequences.")

        if interval_id is not None:
            self.interval_id = interval_id
        self.offset_start = offset_start
        self.offset_end = offset_end

        self._unformat_data()
        self._crop_epochs(verbose)
        self._format_data()

    def reject_epochs(self, interval_id = None, min_duration: float = 0,
                      max_duration: float = float('Inf'), reject_amplitude: float = None,
                      verbose=True):
        """
        Reject epochs based on duration or amplitude, typically before projection.

        interval_id: str, optional
            Name of the variable that contains the trial intervals in the epoch_data
            used for cropping and rejection.
            Default = None = use existing in self.interval_id.
        min_duration : float, optional
            Minimum duration threshold for keeping epochs.
            Default = 0
        max_duration : float, optional
            Maximum duration threshold for keeping epochs.
            Default = Inf
        reject_amplitude : float, optional
            Amplitude threshold for rejecting noisy epochs.
            Default = None
        """
        #warn of previous projection applied
        if self.projection_type != 'identity':
            warn("Projection applied previously, this operation might have\n \
                unintended consequences.")
        if interval_id is not None:
            self.interval_id = interval_id
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.reject_amplitude = reject_amplitude

        self._unformat_data()
        self._reject_epochs(verbose)
        self._format_data()

    def project(self, projection_type, weights = None, n_comp: float = None, center: bool = True,
               method_pca: str='svd', verbose=True):
        """
        Project data from channels to components.

        projection_type: str
            Type of projection to apply. 'pca' or 'custom' or None (identity)
            Default = None
        n_comp: int, optional
            Nr of components retained if > 1, otherwise (0 < n_comp < 1) nr of components
            explaining at least n_comp% variance are retained. If None, user input requested.
        Default = None
            center : bool, optional
            Whether to center the data across the last dimension before projection
            Default = True
        method_pca: str, optional
            Perform PCA ('pca') or SVD decomposition ('svd') for PCA
            Default = 'svd'
        """
        #warn of previous projection applied
        if self.projection_type != 'identity':
            warn("Projection applied previously, this operation might have\n \
                unintended consequences.")

        self.center = center
        self.projection_type = projection_type
        self.weights = weights
        self.n_comp = n_comp
        self.method_pca = method_pca

        self._unformat_data()
        self._center_data()
        self._project(verbose)
        self._format_data()

    def pca(self, n_comp: float = None, center: bool = True, method_pca: str='svd', verbose=True):
        """Apply PCA, see project()."""
        self.project('pca', n_comp = n_comp, center = center, method_pca = method_pca,
                     verbose=verbose)

    def pca_and_variance(self, n_comp: float = None, center: bool = True, method_pca: str='svd',
                         whiten=True, common_variance=True, subject_zscore=True, verbose=True):
        """Apply PCA and variance operations."""
        self.pca(n_comp=n_comp, center=center, method_pca=method_pca, verbose=verbose)
        self.apply_variance_ops(whiten=whiten, common_variance=common_variance,
                                subject_zscore=subject_zscore)

    def apply_pca_weights(self, weights, center: bool = True):
        """Apply previously obtained PCA weights through custom projection."""
        self.project('custom pca', weights, center=center)

    @staticmethod
    def remove_participant(data, participant):
        """Remove data from participant."""
        data = copy.deepcopy(data)
        data.data = data.data.unstack()
        data.data = data.data.drop_sel(participant=[participant])
        data.data = data.data.stack(trial=['participant','epoch'])
        return data

    @staticmethod
    def get_participants(data, participants):
        """Get data from specified participants."""
        data = copy.deepcopy(data)
        data.data = data.data.unstack()
        if isinstance(participants,list) or isinstance(participants,np.ndarray):
            data.data = data.data.sel(participant=participants, drop=False)
        else:
            data.data = data.data.sel(participant=[participants], drop=False)
        data.data = data.data.stack(trial=['participant','epoch'])
        return data


    ##Private functions - all operate based on settings in self

    def _format_io_data(self):
        """Format data coming from io to basedata working format."""
        self.sfreq = self.data.sfreq
        self.data = self.data.data.stack(trial=["participant", "epoch"]).dropna("trial", how="all")
        self.data = self.data.transpose('trial','channel','sample')

    def _format_data(self):
        """From basedata working format to output format."""
        if 'channel' in self.data.dims:
            self.data = self.data.rename({"channel": "component"})
        self.data = self.data.transpose('sample','component','trial')
        self.data.attrs["sfreq"] = self.sfreq
        self.data.attrs["offset"] = self.offset_end

    def _unformat_data(self):
        """From basedata output format to working format."""
        self.data = self.data.rename({'component': 'channel'})
        self.data = self.data.transpose('trial','channel','sample')

    def _center_data(self):
        """Center data per epoch."""
        self.center = True
        self.data -= self.data.median(['sample'],skipna=True)

    def _apply_variance_ops(self):
        """Apply one or more variance operations."""
        if self.whiten:
            self.data /= self.data.std(['trial','sample'], skipna=True)
        else:
            self.data /= self.data.std(..., skipna=True)

        if self.common_variance:
            self.data /= self.data.std(['component','sample'], skipna=True)

        if self.subject_zscore:
            self.data = self.data.unstack()
            self.data -= self.data.mean(['epoch','sample'], skipna=True)
            self.data /= self.data.std(['epoch','sample'], skipna=True)
            self.data = self.data.stack(trial=['participant','epoch'])

    def _crop_epochs(self, verbose=True):
        """
        Crop each epoch from time 0 of the epoch to its interval.

        For epoch in the `epoch_data` xr.Dataset, this function trims the epoch data
        from epoch time 0 (e.g. stimulus onset) up to the specified interval (e.g.
        response), optionally including a fixed offset after the interval.
        """
        assert self.interval_id in self.data.coords, 'interval_id not present in data'

        self.crop = True

        rts_arr = self.data.coords[self.interval_id].values.copy()
        rts_arr = self._check_scale_ms(rts_arr, warning=False)
        offset_end_samples = int(np.rint(self.offset_end * self.sfreq))
        offset_start_samples = int(np.rint(self.offset_start * self.sfreq))

        rts_arr[np.isnan(rts_arr)] = 0  # rejected during epoching or inexistant
        rts_arr = np.rint(rts_arr * self.sfreq).astype(int)

        #check nr of samples
        min_rt = min(rts_arr[rts_arr > 0])
        if min_rt < 10:
           warn("The shortest interval is less than 10 samples. "
                "Consider rejecting too short trials using the `min_duration` parameter "
                "or increasing sampling frequency of the signal.")

        time0 = np.argmin(np.abs(self.data.sample.values))
        min_sample = np.min(self.data.sample.values)
        if min_sample > offset_start_samples:
            raise ValueError("Offset before start is too large for the epoch data provided."
                             f"Max is {time0/self.sfreq} seconds.")

        # Crop the epochs up to duration
        rej_missing_data = 0
        for i in range(len(self.data)):
            #positive RT and no missing channels
            if rts_arr[i] > 0 and ~np.isnan(self.data.values[i, :, 0]).any():
                self.data.values[i, :, time0 + rts_arr[i] + offset_end_samples:] = np.nan
            else:
                self.data.values[i, :, :] = np.nan
                rej_missing_data += 1

        if verbose:
            print()
            print(f"{len(rts_arr[rts_arr > 0])} epochs cropped.")
            print(f"{rej_missing_data} trials rejected because of missing data or interval_id of 0")

        self.data = self.data.sel(
                sample=slice(offset_start_samples,
                min(
                    int(rts_arr.max() + offset_end_samples + 1),
                    self.data.sample.max().values
                )
                ), drop=True
            )
        self.data = self.data.dropna("trial", how="all")

    def _reject_epochs(self, verbose=True):
        """Reject epochs that are too short, too long or have too high amplitudes."""
        assert self.interval_id in self.data.coords, 'interval_id not present in data'

        self.reject_epochs = True

        if self.max_duration < 0 or self.min_duration < 0 or \
            self.max_duration < self.min_duration:
            raise ValueError("Threshold for epoch duration cannot be negative.")

        if self.max_duration is float('Inf'):
            self.max_duration = (int(self.data.sample.max()) - self.offset_end\
                                      + self.offset_start * self.sfreq)/self.sfreq
        if self.min_duration == 0:
            self.min_duration = 1 / self.sfreq

        rts_arr = self.data.coords[self.interval_id].values.copy()
        rts_arr = self._check_scale_ms(rts_arr, warning=False)
        rts_arr[rts_arr > self.max_duration] = 0
        rts_arr[rts_arr < self.min_duration] = 0
        rt_criteria_rej = len(rts_arr[rts_arr == 0])
        inexistant_rej = len(rts_arr[np.isnan(rts_arr)])
        rts_arr[np.isnan(rts_arr)] = 0  # rejected during epoching or inexistant
        rts_arr = np.rint(rts_arr * self.sfreq).astype(int)

        if len(rts_arr[rts_arr > 0]) == 0:
            raise ValueError("No intervals are between the requested limits of "\
                f"minimum {self.min_duration} and maximum {self.max_duration} seconds")

        rej = 0
        reject_threshold = np.abs(self.reject_amplitude)\
            if self.reject_amplitude is not None else np.inf

        for i in range(len(self.data.data)):
            if rts_arr[i] > 0:
                # check amplitude
                if (np.abs(self.data.values[i, :, :]) > reject_threshold).any():
                    self.data.values[i, :, :] = np.nan
                    rej += 1
            else:
                self.data.values[i, :, :] = np.nan

        if verbose:
            print()
            print(f"Rejection summary: \n {rej} trials rejected based on threshold of "
             f"{self.reject_amplitude} \n {rt_criteria_rej} trials rejected based on interval "
             f"limit of {self.min_duration, self.max_duration} \n {inexistant_rej} trials "
             "detected with no interval (e.g. preprocessing or interval exceeding epoch)) ")

        self.data = self.data.dropna("trial", how="all")

    def _project(self, verbose=True):
        """
        Project data from channels to components.

        Three options:
            1. None/identity for identity projection: channels become components.
            2. Custom. Based on provided weights, apply projection.
            3. PCA. Perform and apply PCA.
        """
        if self.projection_type is None or self.projection_type == 'identity':
            if self.weights is not None:
                warn('Projection type None/identity, but weights provided,\n \
                      continuing with custom projection')
                self.projection_type = 'custom'
            else:
                self.projection_type = 'identity'
                self.weights = xr.DataArray(
                                    np.identity(len(self.data.channel)),
                                    dims=("channel", "component"),
                                    coords={"channel": self.data.channel,
                                    "component": np.arange(len(self.data.channel))})
        elif self.projection_type == 'pca':
            self._project_pca(verbose)
        elif 'custom' in self.projection_type:
            assert self.weights is not None, 'Custom projection but no weights provided.'
        else:
            raise ValueError(f'Projection type {self.projection_type} unknown, aborting.')

        assert len(self.weights.channel) == len(self.data.channel),\
            'Different nr of channels in weights and data.'
        self.data = self.data @ self.weights.astype(self.data.dtype)

    def _project_pca(self, verbose=True):
        """Perform and apply PCA."""
        #covariance
        participants = set(self.data.participant.values)
        group_cov = np.zeros((len(participants), self.data.sizes["channel"],
                              self.data.sizes["channel"]), dtype=np.float64)
        for j, participant in enumerate(participants):
            part_data = self.data.where(self.data.participant == participant, drop=True)
            group_cov[j] = self._compute_covariance(part_data)
        vcov_mat = np.mean(group_cov, axis=0)

        #pca
        pca_weights, eigenvalues = self._pca(vcov_mat, self.data.sizes["channel"],
                                            self.data.coords["channel"].values)
        explained_variance_ratio = eigenvalues/np.sum(eigenvalues)

        #select components
        if self.n_comp is None:
            self.n_comp = self._user_input_n_comp(explained_variance_ratio)
        elif self.n_comp < 1: #return nr of components based on explained variance
            self.n_comp = np.where(np.cumsum(explained_variance_ratio) >= self.n_comp)[0][0] + 1
            if verbose:
                print(f"{self.n_comp} components retained, explaining "
                      f"{np.cumsum(explained_variance_ratio)[self.n_comp]:.5f}% variance")
        self.weights = pca_weights[:,:self.n_comp]

    def _user_input_n_comp(self, explained_variance_ratio) -> int:
        """Request user input on number of retained PCA components."""
        _, ax = plt.subplots(1, 2, figsize=(0.2 * len(explained_variance_ratio), 4))
        ax[0].plot(explained_variance_ratio, ".-")
        ax[0].set_ylabel("Normalized explained variance")
        ax[0].set_xlabel("Component")
        ax[1].plot(np.cumsum(explained_variance_ratio), ".-")
        ax[1].set_ylabel("Cumulative normalized explained variance")
        ax[1].set_xlabel("Component")
        plt.tight_layout()
        plt.show()

        n_comp = int(
            input(
                f"How many PCs would you like to retain?"
                f"95 and 99% explained variance at components "
                f"{np.where(np.cumsum(explained_variance_ratio) >= 0.95)[0][0] + 1} and "
                f"{np.where(np.cumsum(explained_variance_ratio) >= 0.99)[0][0] + 1}; "
                f"components till {np.where(explained_variance_ratio >= 0.01)[0][-1] + 1} "
                f"explain at least 1% variance per component)?"
            )
        )
        return n_comp

    def _pca(self, vcov_mat: xr.DataArray, n_comp: int, channel: xr.DataArray):
        """
        Calculate PCA.

        vcov_mat : xr.DataArray
            data to perform PCA on.
        n_comp : int
            nr of components to return
        channel: xr.DataArray
            channel names
        """
        if self.method_pca == 'pca':
            eigvals, eigvecs = eigh(vcov_mat)
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            evecs = evecs[:, :n_comp]
            eigvals = eigvals[ix]
        elif self.method_pca == 'svd':
            _, s, evecs = np.linalg.svd(vcov_mat, full_matrices=False)
            eigvals = (s**2) / (vcov_mat.shape[0] - 1)
            evecs = evecs.T[:, :n_comp]

        # Rebuilding as xarray to ease computation
        coords = dict(channel=("channel", channel), component=("component", np.arange(n_comp)))
        pca_weights = xr.DataArray(evecs, dims=("channel", "component"), coords=coords)
        return pca_weights, eigvals

    @staticmethod
    def _compute_covariance(data):
        """Compute covariance of data trial by trial."""
        vcov_mat = np.zeros((data.sizes["channel"], data.sizes["channel"]), dtype=np.float64)
        # Iteratively for memory efficiency
        count = 0
        for i in data.trial:
            x_i = np.squeeze(data.sel(trial=i).values)
            x_i = x_i[:, ~np.isnan(x_i[0, :])]
            if x_i.shape[1] > x_i.shape[0]:
                count += 1
                # Assumes centered data
                cov_i = (x_i @ x_i.T) / (x_i.shape[1]-1)
                # Regularization using MNE python's default
                sigma = np.mean(np.diag(cov_i))
                cov_i.flat[:: len(cov_i) + 1] += 0.1 * sigma
                vcov_mat += cov_i
        if count < len(data.trial)/10:
            warn(f"Less than 10% of the trials used to compute covariance for"
                 f"{np.unique(data.participant.values)}. Covariance matrix might be unreliable")
        return vcov_mat/count

    @staticmethod
    def _check_scale_ms(rts, warning=True):
        max_rt = np.nanmax(rts)
        if max_rt > 500:
            if warning:
                warn(f"Found intervals with a max value value of {np.round(max_rt,2)}\n,\
                        assuming intervals are in milliseconds and converting to seconds")
            rts /= 1000
        return rts
