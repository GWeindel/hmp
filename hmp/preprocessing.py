import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from enum import Enum
from mne.filter import filter_data
from sklearn.decomposition import PCA
from typing import Union, Optional
from warnings import warn
from hmp import mcca


class ApplyZScore(Enum):
    ALL = 'all'
    PARTICIPANT = 'participant'
    TRIAL = 'trial'
    DONT_APPLY = 'dont_apply'

    def __str__(self) -> str:
        return self.value

    def __bool__(self) -> bool:
        return self != self.DONT_APPLY

    @classmethod
    def parse(cls, label):
        if isinstance(label, str):
            label = label.lower()

        if isinstance(label, cls):
            return label
        elif label in (False, None, 'dont_apply'):
            return cls.DONT_APPLY
        elif label in ('trial', True):
            return cls.TRIAL
        elif label == 'participant':
            return cls.PARTICIPANT
        elif label == 'all':
            return cls.ALL
        else:
            raise KeyError(f"Unknown value for apply_zscore: '{label}'; valid options: [{', '.join([e.value for e in cls])}] or Bool (True defaults to {cls.TRIAL})")# noqa: E501


class AnalysisMethod(Enum):
    PCA = 'pca'
    MCCA = 'mcca'
    NO_ANALYSIS = 'no_analysis'

    def __str__(self) -> str:
        return self.value

    def __bool__(self) -> bool:
        return self != self.NO_ANALYSIS

    @classmethod
    def parse(cls, label):
        if isinstance(label, str):
            label = label.lower()

        if isinstance(label, cls):
            return label
        elif label in (None, 'no_analysis'):
            return cls.NO_ANALYSIS
        elif label == 'pca':
            return cls.PCA
        elif label == 'mcca':
            return cls.MCCA
        else:
            raise KeyError(f"Unknown method: '{label}'; valid options: {', '.join([e.value for e in cls])} or None")  # noqa: E501


def user_input_n_comp(data):

    n_comp = np.shape(data)[0] - 1
    fig, ax = plt.subplots(1, 2, figsize=(0.2 * n_comp, 4))
    pca = PCA(n_components=n_comp, svd_solver="full", copy=False)  # selecting PCs
    pca.fit(data)

    ax[0].plot(np.arange(pca.n_components) + 1, pca.explained_variance_ratio_, ".-")
    ax[0].set_ylabel("Normalized explained variance")
    ax[0].set_xlabel("Component")
    ax[1].plot(np.arange(pca.n_components) + 1, np.cumsum(pca.explained_variance_ratio_), ".-")
    ax[1].set_ylabel("Cumulative normalized explained variance")
    ax[1].set_xlabel("Component")
    plt.tight_layout()
    plt.show()

    # TODO: needs user input validation?
    n_comp = int(
        input(
            f"How many PCs (95 and 99% explained variance at component "
            f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1} and "
            f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0][0] + 1}; "
            f"components till n{np.where(pca.explained_variance_ratio_ >= 0.01)[0][-1] + 1} "
            f"explain at least 1%)?"
        )
    )

    return n_comp


# TODO: n_comp  None value input
# TODO: printing to proper logging

class Preprocessing:

    def __init__(
        self,
        epoch_data: xr.DataArray,
        participants_variable: str = 'participant',
        apply_standard: bool = False,
        averaged: bool = False,
        apply_zscore: Union[bool, str, ApplyZScore] = ApplyZScore.TRIAL,
        zscore_across_pcs: bool = False,
        method: Optional[Union[bool, str, AnalysisMethod]] = AnalysisMethod.PCA,
        cov: bool = True,
        centering: bool = True,
        n_comp: Optional[int] = None,
        n_ppcas: Optional[int] = None,
        weights: Optional[xr.DataArray] = None,
        bandfilter: Optional[Union[tuple[float, float]]] = None,
        mcca_reg: float = 0,
        copy: bool = False,
    ) -> None:
        """Preprocess EEG epoched data (in xarray format) for HMP analysis.

        This function performs several preprocessing steps on EEG data:
        1. Optionally standardizes individual variances between participants (if apply_standard=True).
        2. Applies a spatial PCA or MCCA on the data, depending on the selected method.
        3. Stacks the data from the format [n_participants * n_epochs * n_samples * n_channels] 
           to [sample * channel].
        4. Optionally applies z-scoring on the data, either across all data, by participant, 
           or by trial, depending on the apply_zscore parameter.

        Parameters
        ----------
        epoch_data : xr.DataArray
            Input data with dimensions [n_participants * n_epochs * n_samples * n_channels].
        participants_variable : str, optional
            Name of the dimension for participant IDs. Default is 'participant'.
        apply_standard : bool, optional
            Whether to standardize variance between participants. Recommended when there are 
            few participants (e.g., < 10). Default is False.
        averaged : bool, optional
            Whether to apply PCA/MCCA on the averaged ERP (True) or single-trial ERP (False). 
            Only applicable for the MCCA method when cov=False. Default is False.
        apply_zscore : Union[bool, str, ApplyZScore], optional
            Specifies whether to apply z-scoring and on what data. Options are:
            - 'all': Z-score across all data.
            - 'participant': Z-score by participant.
            - 'trial': Z-score by trial.
            - None or 'dont_apply': No z-scoring.
            If set to True, defaults to 'trial' for backward compatibility. Default is ApplyZScore.TRIAL.
        method : Union[bool, str, AnalysisMethod], optional
            Analysis method to apply. Options are:
            - 'pca': Apply PCA.
            - 'mcca': Apply MCCA.
            - 'no_analysis': Skip analysis. Default is AnalysisMethod.PCA.
        cov : bool, optional
            Whether to apply PCA/MCCA to the variance-covariance matrix (True) or the epoched data (False). 
            Only applicable for the MCCA method. Default is True.
        n_comp : int, optional
            Number of components to retain in the PC space. If None, a scree plot is displayed, 
            and the user is prompted to specify the number of components. Default is None.
        n_ppcas : int, optional
            For the MCCA method, controls the number of components retained for by-participant PCAs. Default is None.
        weights : Optional[xr.DataArray], optional
            Precomputed linear combinations of channels. Default is None.
        bandfilter : Optional[tuple[float, float]], optional
            Frequency range for bandpass filtering (lfreq, hfreq). If None, no filtering is applied. 
            Filtering at this step is suboptimal; it is recommended to filter before epoching. Default is None.
        mcca_reg : float, optional
            Regularization parameter for the MCCA computation. Default is 0.
        copy : bool, optional
            Whether to copy the data before transforming. If False, the data is modified in place. Default is False.

        Returns
        -------
        data : xr.DataArray
            Preprocessed data with dimensions [n_samples * n_comp], expressed in the PC space, ready for HMP analysis.
        """
        offset = epoch_data.offset
        if copy is True:
            data = epoch_data.copy(deep=True)
        else:
            data = epoch_data
            warn(
                    "Data will be modified inplace, re-read the data or use copy=True if multiple"
                    "calls to this function"
                )

        if isinstance(data, xr.DataArray):
            raise ValueError(
                "Expected a xarray Dataset with data and event as DataArrays, check the data format"
                )

        apply_zscore = ApplyZScore.parse(apply_zscore)

        if np.sum(
            np.isnan(data.groupby("participant",
                                  squeeze=False).mean(["epoch", "sample"]).data.values)) != 0:
            raise ValueError("at least one participant has an empty channel")

        method = AnalysisMethod.parse(method)

        if method == AnalysisMethod.MCCA and data.sizes["participant"] == 1:
            raise ValueError("MCCA cannot be applied to only one participant")

        sfreq = data.sfreq

        if bandfilter:
            data = self._apply_filtering(data, bandfilter, sfreq)

        if apply_standard:
            if "participant" not in data.dims or len(data.participant) == 1:
                warn(
                    "Requested standardization of between participant variance yet no participant "
                    "dimension is found in the data or only one participant is present. "
                    "No standardization is done, set apply_standard to False to avoid this warning."
                )
            else:
                mean_std = data.groupby(
                    participants_variable, squeeze=False).std(dim=...).data.mean()
                data = data.assign(mean_std=mean_std.data)
                data = data.groupby(participants_variable, squeeze=False).map(self._standardize)

        if isinstance(data, xr.Dataset):  # needs to be a dataset if apply_standard is used
            data = data.data

        if centering or method == Method.MCCA:
            data = self._center(data)

        data = data.transpose("participant", "epoch", "channel", "sample")

        if method == AnalysisMethod.PCA:
            if weights is None:
                indiv_data = np.zeros(
                    (data.sizes["participant"], data.sizes["channel"], data.sizes["channel"])
                )
                for i in range(data.sizes["participant"]):
                    x_i = np.squeeze(data.data[i])
                    indiv_data[i] = np.mean(
                        [
                            np.cov(x_i[trial, :, ~np.isnan(x_i[trial, 0, :])].T)
                            for trial in range(x_i.shape[0])
                            if ~np.isnan(x_i[trial, 0, :]).all()
                        ],
                        axis=0,
                    )
                pca_ready_data = np.mean(np.array(indiv_data), axis=0)
                # Performing spatial PCA on the average var-cov matrix
                weights, preprocessing_model = self._pca(pca_ready_data, n_comp, data.coords["channel"].values)
                data = data @ weights
                weights = weights
            else:
                data = data @ weights

        elif method == AnalysisMethod.MCCA:

            ori_coords = data.drop_vars("channel").coords
            if n_ppcas is None:
                n_ppcas = n_comp * 3
            mcca_m = mcca.MCCA(n_components_pca=n_ppcas, n_components_mcca=n_comp, r=mcca_reg)
            if cov:
                fitted_data = data.transpose("participant", "epoch", "sample", "channel").data
                ccs = mcca_m.obtain_mcca_cov(fitted_data)
            else:
                if averaged:
                    fitted_data = (
                        data.mean("epoch").transpose("participant", "sample", "channel").data
                    )
                else:
                    fitted_data = (
                        data.stack({"all": ["epoch", "sample"]})
                        .transpose("participant", "all", "channel")
                        .data
                    )
                ccs = mcca_m.obtain_mcca(fitted_data)
            trans_ccs = np.tile(
                np.nan,
                (data.sizes["participant"],
                 data.sizes["epoch"],
                 data.sizes["sample"],
                 ccs.shape[-1]),
            )
            for i, part in enumerate(data.participant):
                trans_ccs[i] = mcca_m.transform_trials(
                    data.sel(participant=part).transpose(
                        "epoch", "sample", "channel").data.copy()
                )
            data = xr.DataArray(
                trans_ccs,
                dims=["participant", "epoch", "sample", "component"],
                coords=dict(
                    participant=data.participant,
                    epoch=data.epoch,
                    sample=data.sample,
                    component=np.arange(n_comp),
                ),  # n_comp
            )
            data = data.assign_coords(ori_coords)
            weights = mcca_m.mcca_weights
            preprocessing_model = mcca_m

        elif not method:

            data = data.rename({"channel": "component"})
            data["component"] = np.arange(len(data.component))
            weights = np.identity(len(data.component))
            preprocessing_model = None


        if apply_zscore:

            ori_coords = data.coords

            match apply_zscore:
                case ApplyZScore.ALL:
                    if zscore_across_pcs:
                        data = self.zscore_xarray(data)
                    else:
                        data = (
                            data.stack(comp=["component"])
                            .groupby("comp", squeeze=False)
                            .map(self.zscore_xarray)
                            .unstack()
                        )
                case ApplyZScore.PARTICIPANT:
                    if zscore_across_pcs:
                        data = data.groupby("participant").map(self.zscore_xarray)
                    else:
                        data = (
                            data.stack(participant_comp=[participants_variable, "component"])
                            .groupby("participant_comp", squeeze=False)
                            .map(self.zscore_xarray)
                            .unstack()
                        )
                case ApplyZScore.TRIAL:
                    if zscore_across_pcs:
                        data = (
                            data.stack(trials=[participants_variable, "epoch"])
                            .groupby("trials")
                            .map(self.zscore_xarray)
                            .unstack()
                        )
                    else:
                        data = (
                            data.stack(trials=[participants_variable, "epoch", "component"])
                            .groupby("trials", squeeze=False)
                            .map(self.zscore_xarray)
                            .unstack()
                        )
            data = data.transpose("participant", "epoch", "sample", "component")
            data = data.assign_coords(ori_coords)

        data.attrs["sfreq"] = sfreq
        data.attrs["offset"] = offset
        self.data = self.stack_data(data)
        self.weights = weights
        self.preprocessing_model = preprocessing_model

    @staticmethod
    def _center(data: xr.DataArray) -> xr.DataArray:
        """Center the data."""
        mean_last_dim = np.nanmean(data.values, axis=-1)
        mean_last_dim_expanded = np.expand_dims(mean_last_dim, axis=-1)
        centred = data.values - mean_last_dim_expanded
        data.values = centred
        return data

    @staticmethod
    def _pca(pca_ready_data: xr.DataArray, n_comp: int, channel) -> xr.DataArray:
        # TODO: test seperate function
        n_comp = user_input_n_comp(data=pca_ready_data) if n_comp is None else n_comp
        pca = PCA(n_components=n_comp, svd_solver="full")  # selecting Principale components (PC)
        pca.fit(pca_ready_data)
        # Rebuilding pca PCs as xarray to ease computation
        coords = dict(channel=("channel", channel), component=("component", np.arange(n_comp)))
        pca_weights = xr.DataArray(pca.components_.T, dims=("channel", "component"), coords=coords)
        return pca_weights, pca

    @staticmethod
    def zscore_xarray(data: Union[xr.Dataset, xr.DataArray]) -> xr.DataArray:
        """Zscore of the data in an xarray, avoiding any nans."""
        if isinstance(data, xr.Dataset):  # if no PCA
            data = data.data
        non_nan_mask = ~np.isnan(data.values)
        if non_nan_mask.any():  # if not everything is nan, calc zscore
            data.values[non_nan_mask] = (
                data.values[non_nan_mask] - data.values[non_nan_mask].mean()
            ) / data.values[non_nan_mask].std()
        return data

    @staticmethod
    def stack_data(data: xr.DataArray) -> xr.DataArray:
        """Stack the data.

        Going from format [n_participant * n_epochs *n_ n_samples * n_channels] to
        [sample * channel] with sample indexes starts and ends to delimitate the epochs.


        Parameters
        ----------
        data : xarray
            xarray with dimensions [n_participant * n_epochs * n_samples * n_channels]
        subjects_variable : str
            name of the dimension for subjects ID

        Returns
        -------
        data : xarray.Dataset
            xarray dataset [sample * channel]
        """
        if isinstance(data, (xr.DataArray, xr.Dataset)) and "component" not in data.dims:
            data = data.rename_dims({"channel": "component"})
        if "participant" not in data.dims:
            data = data.expand_dims("participant")
        data = data.stack(
            all_samples=["participant", "epoch", "sample"]).dropna(dim="all_samples")
        return data

    @staticmethod
    def _apply_filtering(data: xr.DataArray,
                         bandfilter:Optional[Union[tuple[float, float]]],
                         sfreq: float):
        print("""
        NOTE: filtering at this step is suboptimal, filter before epoching if at all possible,
        see also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
        """)

        lfreq, hfreq = bandfilter
        n_participant, n_epochs, _, _ = data.data.values.shape
        for pp in range(n_participant):
            for trial in range(n_epochs):
                dat = data.data.values[pp, trial, :, :]

                if not np.isnan(dat).all():
                    dat = dat[:, ~np.isnan(dat[0, :])]  # remove nans

                    # pad by reflecting the whole trial twice
                    trial_len = dat.shape[1] * 2
                    dat = np.pad(dat, ((0, 0), (trial_len, trial_len)), mode="reflect")

                    # filter
                    dat = filter_data(dat, sfreq, lfreq, hfreq, verbose=False)

                    # remove padding
                    dat = dat[:, trial_len:-trial_len]
                    data.data.values[pp, trial, :, : dat.shape[1]] = dat
            return data

    @staticmethod
    def _standardize(x):
        """Scaling variances to mean variance of the group."""
        return (x.data / x.data.std(dim=...)) * x.mean_std
