import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from enum import Enum
from mne.filter import filter_data
from sklearn.decomposition import PCA
from typing import Union, Optional
from warnings import warn
from hmp import mcca


# TODO: move to utils
class ApplyZScore(Enum):
    ALL = 'all'
    PARTICIPANT = 'participant'
    TRIAL = 'trial'
    DONT_APPLY = 'dont_apply'

    def __str__(self) -> str:
        return self.value

    def __bool__(self) -> bool:
        return self != ApplyZScore.DONT_APPLY

    @staticmethod
    def parse(label):
        if isinstance(label, ApplyZScore):
            return label
        elif label is False or label is None:
            return ApplyZScore.DONT_APPLY
        elif label in ('trial', True):
            return ApplyZScore.TRIAL
        elif label == 'participant':
            return ApplyZScore.PARTICIPANT
        elif label == 'all':
            return ApplyZScore.ALL
        else:
            raise NotImplementedError


class AnalysisMethod(Enum):
    PCA = 'pca'
    MCCA = 'mcca'
    NO_ANALYSIS = 'no_analysis'

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def parse(label):
        if isinstance(label, AnalysisMethod):
            return label
        elif label is False or label is None:
            return AnalysisMethod.NO_ANALYSIS
        elif label and label.lower() == 'pca':
            return AnalysisMethod.PCA
        elif label and label.lower() == 'mcca':
            return AnalysisMethod.MCCA
        else:
            raise NotImplementedError


# TODO: move to utils
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

class DataTransformer:

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
        pca_weights: Optional[xr.DataArray] = None,
        bandfilter: Optional[Union[tuple[float, float]]] = None,
        mcca_reg: float = 0,
        copy: bool = False,
    ) -> None:
        """Adapt EEG epoched data (in xarray format) to the expected data format for hmp.

        First this code can apply standardization of individual variances
            (if apply_standard=True).
        Second, a spatial PCA on the average variance-covariance matrix is performed
            (if method='pca', more methods in development).
        Third, stacks the data going from format [participant * epochs * samples * channels] to
            [samples * channels].
        Last, performs z-scoring on each epoch and for each principal component (PC), or for each
            participant and PC, or across all data for each PC.

        Parameters
        ----------
        epoch_data : xarray
            unstacked xarray data from transform_data() or anyother source yielding an xarray with
            dimensions [participant * epochs * samples * channels]
        participants_variable : str
            name of the dimension for participants ID
        apply_standard : bool
            Whether to apply standardization of variance between participants, recommended when they
            are few of them (e.g. < 10)
        averaged : bool
            Applying the pca on the averaged ERP (True) or single trial ERP (False, default).
            No effect if cov = True
        apply_zscore : str
            Whether to apply z-scoring and on what data, either None, 'all', 'participant', 'trial',
            for zscoring across all data, by participant, or by trial, respectively. If set to true,
            evaluates to 'trial' for backward compatibility.
        method : str
            Method to apply, 'pca' or 'mcca'
        cov : bool
            Wether to apply the pca/mcca to the variance covariance (True, default) or the epoched
            data
        n_comp : int
            How many components to select from the PC space, if None plots the scree plot and a
            prompt requires user to specify how many PCs should be retained
        n_ppcas : int
            If method = 'mcca', controls the number of components retained for the by-participant
            PCAs
        pca_weigths : xarray
            Weights of a PCA to apply to the data (e.g. in the resample function)
        bandfilter: None | (lfreq, hfreq)
            If none, no filtering is appliedn. If tuple, data is filtered between lfreq-hfreq.
            NOTE: filtering at this step is suboptimal, filter before epoching if at all possible,
            see also
            https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
        mcca_reg: float
            regularization used for the mcca computation (see mcca.py)
        copy: bool
            Whether to copy the data before transforming
        Returns
        -------
        data : xarray.Dataset
            xarray dataset [n_samples * n_comp] data expressed in the PC space, ready for HMP fit
        """
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

        try:
            apply_zscore = ApplyZScore.parse(apply_zscore)
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Unknown value for apply_zscore: {apply_zscore!r}; valid options: [{', '.join([e.value for e in ApplyZScore])}] or Bool (True defaults to {ApplyZScore.TRIAL})") from e  # noqa: E501

        if np.sum(
            np.isnan(data.groupby("participant",
                                  squeeze=False).mean(["epochs", "samples"]).data.values)) != 0:
            raise ValueError("at least one participant has an empty channel")

        try:
            method = AnalysisMethod.parse(method)
        except NotImplementedError as e:
            raise NotImplementedError(f"Unknown method: {method!r}; valid options: {', '.join([e.value for e in AnalysisMethod])} or None") from e  # noqa: E501

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

        data = data.transpose("participant", "epochs", "channels", "samples")

        if method == AnalysisMethod.PCA:

            if pca_weights is None:
                if cov:
                    indiv_data = np.zeros(
                        (data.sizes["participant"], data.sizes["channels"], data.sizes["channels"])
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
                elif averaged:
                    erps = []
                    for part in data.participant:
                        erps.append(data.sel(participant=part).groupby("samples").mean("epochs").T)
                    pca_ready_data = np.nanmean(erps, axis=0)
                else:
                    pca_ready_data = data.stack(
                        {"all": ["participant", "epochs", "samples"]}
                    ).dropna("all")
                    pca_ready_data = pca_ready_data.transpose("all", "channels")

                # Performing spatial PCA on the average var-cov matrix
                pca_weights = self._pca(pca_ready_data, n_comp, data.coords["channels"].values)
                data = data @ pca_weights
                data.attrs["pca_weights"] = pca_weights

        elif method == AnalysisMethod.MCCA:

            ori_coords = data.drop_vars("channels").coords
            if n_ppcas is None:
                n_ppcas = n_comp * 3
            mcca_m = mcca.MCCA(n_components_pca=n_ppcas, n_components_mcca=n_comp, r=mcca_reg)
            if cov:
                fitted_data = data.transpose("participant", "epochs", "samples", "channels").data
                ccs = mcca_m.obtain_mcca_cov(fitted_data)
            else:
                if averaged:
                    fitted_data = (
                        data.mean("epochs").transpose("participant", "samples", "channels").data
                    )
                else:
                    fitted_data = (
                        data.stack({"all": ["epochs", "samples"]})
                        .transpose("participant", "all", "channels")
                        .data
                    )
                ccs = mcca_m.obtain_mcca(fitted_data)
            trans_ccs = np.tile(
                np.nan,
                (data.sizes["participant"],
                 data.sizes["epochs"],
                 data.sizes["samples"],
                 ccs.shape[-1]),
            )
            for i, part in enumerate(data.participant):
                trans_ccs[i] = mcca_m.transform_trials(
                    data.sel(participant=part).transpose(
                        "epochs", "samples", "channels").data.copy()
                )
            data = xr.DataArray(
                trans_ccs,
                dims=["participant", "epochs", "samples", "component"],
                coords=dict(
                    participant=data.participant,
                    epochs=data.epochs,
                    samples=data.samples,
                    component=np.arange(n_comp),
                ),  # n_comp
            )
            data = data.assign_coords(ori_coords)
            data.attrs["mcca_weights"] = mcca_m.mcca_weights
            data.attrs["pca_weights"] = mcca_m.pca_weights

        elif method is AnalysisMethod.NO_ANALYSIS:

            data = data.rename({"channels": "component"})
            data["component"] = np.arange(len(data.component))
            data.attrs["pca_weights"] = np.identity(len(data.component))


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
                            data.stack(trial=[participants_variable, "epochs"])
                            .groupby("trial")
                            .map(self.zscore_xarray)
                            .unstack()
                        )
                    else:
                        data = (
                            data.stack(trial=[participants_variable, "epochs", "component"])
                            .groupby("trial", squeeze=False)
                            .map(self.zscore_xarray)
                            .unstack()
                        )
            data = data.transpose("participant", "epochs", "samples", "component")
            data = data.assign_coords(ori_coords)

        data.attrs["pca_weights"] = pca_weights
        data.attrs["sfreq"] = sfreq

        self.data = self.stack_data(data)

    @staticmethod
    def _center(data: xr.DataArray) -> xr.DataArray:
        """Center the data."""
        mean_last_dim = np.nanmean(data.values, axis=-1)
        mean_last_dim_expanded = np.expand_dims(mean_last_dim, axis=-1)
        centred = data.values - mean_last_dim_expanded
        data.values = centred
        return data

    @staticmethod
    def _pca(pca_ready_data: xr.DataArray, n_comp: int, channels) -> xr.DataArray:
        # TODO: test seperate function
        n_comp = user_input_n_comp(pca_ready_data=pca_ready_data) if n_comp is None else n_comp
        pca = PCA(n_components=n_comp, svd_solver="full")  # selecting Principale components (PC)
        pca.fit(pca_ready_data)
        # Rebuilding pca PCs as xarray to ease computation
        coords = dict(channels=("channels", channels), component=("component", np.arange(n_comp)))
        pca_weights = xr.DataArray(pca.components_.T, dims=("channels", "component"), coords=coords)
        return pca_weights

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

        Going from format [participant * epochs * samples * channels] to
        [samples * channels] with sample indexes starts and ends to delimitate the epochs.


        Parameters
        ----------
        data : xarray
            unstacked xarray data from transform_data() or anyother source yielding an xarray with
            dimensions [participant * epochs * samples * channels]
        subjects_variable : str
            name of the dimension for subjects ID

        Returns
        -------
        data : xarray.Dataset
            xarray dataset [samples * channels]
        """
        if isinstance(data, (xr.DataArray, xr.Dataset)) and "component" not in data.dims:
            data = data.rename_dims({"channels": "component"})
        if "participant" not in data.dims:
            data = data.expand_dims("participant")
        data = data.stack(
            all_samples=["participant", "epochs", "samples"]).dropna(dim="all_samples")
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
