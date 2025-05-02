"""Functions to transform the input data and the estimates."""

from warnings import filterwarnings, warn

import numpy as np
import pandas as pd
import xarray as xr
from pandas import MultiIndex
from scipy.special import gamma as gamma_func
from scipy.stats import sem

from hmp import mcca

filterwarnings(
    "ignore",
    "Degrees of freedom <= 0 for slice.",
)  # weird warning, likely due to nan in xarray, not important but better fix it later
filterwarnings("ignore", "Mean of empty slice")  # When trying to center all-nans trials


def _gamma_scale_to_mean(scale, shape):
    return scale * shape


def _gamma_mean_to_scale(mean, shape):
    return mean / shape


def _logn_scale_to_mean(scale, shape):
    return np.exp(scale + shape**2 / 2)


def _logn_mean_to_scale(mean, shape):
    return np.exp(np.log(mean) - (shape**2 / 2))


def _wald_scale_to_mean(scale, shape):
    return scale * shape


def _wald_mean_to_scale(mean, shape):
    return mean / shape


def _weibull_scale_to_mean(scale, shape):
    return scale * gamma_func(1 + 1 / shape)


def _weibull_mean_to_scale(mean, shape):
    return mean / gamma_func(1 + 1 / shape)


def _standardize(x):
    """Scaling variances to mean variance of the group."""
    return (x.data / x.data.std(dim=...)) * x.mean_std


def _center(data):
    """Center the data."""
    mean_last_dim = np.nanmean(data.values, axis=-1)
    mean_last_dim_expanded = np.expand_dims(mean_last_dim, axis=-1)
    centred = data.values - mean_last_dim_expanded
    data.values = centred

    return data


def zscore_xarray(data):
    """Zscore of the data in an xarray, avoiding any nans."""
    if isinstance(data, xr.Dataset):  # if no PCA
        data = data.data
    non_nan_mask = ~np.isnan(data.values)
    if non_nan_mask.any():  # if not everything is nan, calc zscore
        data.values[non_nan_mask] = (
            data.values[non_nan_mask] - data.values[non_nan_mask].mean()
        ) / data.values[non_nan_mask].std()
    return data


def stack_data(data):
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
    data = data.stack(all_samples=["participant", "epochs", "samples"]).dropna(dim="all_samples")
    return data


def _filtering(data, filter, sfreq):
    print(
        "NOTE: filtering at this step is suboptimal, filter before epoching if at all possible, see"
    )
    print("also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html")
    from mne.filter import filter_data

    lfreq, hfreq = filter
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


def _pca(pca_ready_data, n_comp, channels):
    from sklearn.decomposition import PCA

    if n_comp is None:
        import matplotlib.pyplot as plt

        n_comp = np.shape(pca_ready_data)[0] - 1
        fig, ax = plt.subplots(1, 2, figsize=(0.2 * n_comp, 4))
        pca = PCA(n_components=n_comp, svd_solver="full", copy=False)  # selecting PCs
        pca.fit(pca_ready_data)

        ax[0].plot(np.arange(pca.n_components) + 1, pca.explained_variance_ratio_, ".-")
        ax[0].set_ylabel("Normalized explained variance")
        ax[0].set_xlabel("Component")
        ax[1].plot(np.arange(pca.n_components) + 1, np.cumsum(pca.explained_variance_ratio_), ".-")
        ax[1].set_ylabel("Cumulative normalized explained variance")
        ax[1].set_xlabel("Component")
        plt.tight_layout()
        plt.show()
        n_comp = int(
            input(
                f"How many PCs (95 and 99% explained variance at component "
                f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1} and "
                f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0][0] + 1}; "
                f"components till n{np.where(pca.explained_variance_ratio_ >= 0.01)[0][-1] + 1} "
                f"explain at least 1%)?"
            )
        )

    pca = PCA(n_components=n_comp, svd_solver="full")  # selecting Principale components (PC)
    pca.fit(pca_ready_data)
    # Rebuilding pca PCs as xarray to ease computation
    coords = dict(channels=("channels", channels), component=("component", np.arange(n_comp)))
    pca_weights = xr.DataArray(pca.components_.T, dims=("channels", "component"), coords=coords)
    return pca_weights


def transform_data(
    epoch_data,
    participants_variable="participant",
    apply_standard=False,
    averaged=False,
    apply_zscore="trial",
    zscore_across_pcs=False,
    method="pca",
    cov=True,
    centering=True,
    n_comp=None,
    n_ppcas=None,
    pca_weights=None,
    bandfilter=None,
    mcca_reg=0,
    copy=False,
):
    """Adapt EEG epoched data (in xarray format) to the expected data format for hmp.

    First this code can apply standardization of individual variances (if apply_standard=True).
    Second, a spatial PCA on the average variance-covariance matrix is performed (if method='pca',
    more methods in development).
    Third,stacks the data going from format [participant * epochs * samples * channels] to
    [samples * channels].
    Last, performs z-scoring on each epoch and for each principal component (PC), or for each
    participant and PC, or across all data for each PC.

    Parameters
    ----------
    data : xarray
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
        Wether to apply the pca/mcca to the variance covariance (True, default) or the epoched data
    n_comp : int
        How many components to select from the PC space, if None plots the scree plot and a prompt
        requires user to specify how many PCs should be retained
    n_ppcas : int
        If method = 'mcca', controls the number of components retained for the by-participant PCAs
    pca_weigths : xarray
        Weights of a PCA to apply to the data (e.g. in the resample function)
    bandfilter: None | (lfreq, hfreq)
        If none, no filtering is appliedn. If tuple, data is filtered between lfreq-hfreq.
        NOTE: filtering at this step is suboptimal, filter before epoching if at all possible, see
            also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
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
    if apply_zscore not in ["all", "participant", "trial"] and not isinstance(apply_zscore, bool):
        raise ValueError(
            "apply_zscore should be either a boolean or one of ['all', 'participant', 'trial']"
        )
    assert (
        np.sum(
            np.isnan(
                data.groupby("participant", squeeze=False).mean(["epochs", "samples"]).data.values
            )
        )
        == 0
    ), "at least one participant has an empty channel"
    if method == "mcca" and data.sizes["participant"] == 1:
        raise ValueError("MCCA cannot be applied to only one participant")
    sfreq = data.sfreq
    if bandfilter:
        data = _filtering(data, bandfilter, sfreq)
    if apply_standard:
        if "participant" not in data.dims or len(data.participant) == 1:
            warn(
                "Requested standardization of between participant variance yet no participant "
                "dimension is found in the data or only one participant is present. "
                "No standardization is done, set apply_standard to False to avoid this warning."
            )
        else:
            mean_std = data.groupby(participants_variable, squeeze=False).std(dim=...).data.mean()
            data = data.assign(mean_std=mean_std.data)
            data = data.groupby(participants_variable, squeeze=False).map(_standardize)
    if isinstance(data, xr.Dataset):  # needs to be a dataset if apply_standard is used
        data = data.data
    if centering or method == "mcca":
        data = _center(data)
    if apply_zscore is True:
        apply_zscore = "trial"  # defaults to trial
    data = data.transpose("participant", "epochs", "channels", "samples")
    if method == "pca":
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
            pca_weights = _pca(pca_ready_data, n_comp, data.coords["channels"].values)
            data = data @ pca_weights
            data.attrs["pca_weights"] = pca_weights
    elif method == "mcca":
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
            (data.sizes["participant"], data.sizes["epochs"], data.sizes["samples"], ccs.shape[-1]),
        )
        for i, part in enumerate(data.participant):
            trans_ccs[i] = mcca_m.transform_trials(
                data.sel(participant=part).transpose("epochs", "samples", "channels").data.copy()
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
    elif method is None:
        data = data.rename({"channels": "component"})
        data["component"] = np.arange(len(data.component))
        data.attrs["pca_weights"] = np.identity(len(data.component))
    else:
        raise ValueError(f"method {method} is unknown, choose either 'pca', 'mcca' or None")

    if apply_zscore:
        ori_coords = data.coords
        match apply_zscore:
            case "all":
                if zscore_across_pcs:
                    data = zscore_xarray(data)
                else:
                    data = (
                        data.stack(comp=["component"])
                        .groupby("comp", squeeze=False)
                        .map(zscore_xarray)
                        .unstack()
                    )
            case "participant":
                if zscore_across_pcs:
                    data = data.groupby("participant").map(zscore_xarray)
                else:
                    data = (
                        data.stack(participant_comp=[participants_variable, "component"])
                        .groupby("participant_comp", squeeze=False)
                        .map(zscore_xarray)
                        .unstack()
                    )
            case "trial":
                if zscore_across_pcs:
                    data = (
                        data.stack(trial=[participants_variable, "epochs"])
                        .groupby("trial")
                        .map(zscore_xarray)
                        .unstack()
                    )
                else:
                    data = (
                        data.stack(trial=[participants_variable, "epochs", "component"])
                        .groupby("trial", squeeze=False)
                        .map(zscore_xarray)
                        .unstack()
                    )
        data = data.transpose("participant", "epochs", "samples", "component")
        data = data.assign_coords(ori_coords)

    data.attrs["pca_weights"] = pca_weights
    data.attrs["sfreq"] = sfreq
    data = stack_data(data)
    return data


def event_times(
    estimates,
    duration=False,
    mean=False,
    add_rt=False,
    as_time=False,
    errorbars=None,
    estimate_method="max",
    add_stim=False,
):
    """Compute the likeliest peak times for each event.

    Parameters
    ----------
    estimates : xr.Dataset
        Estimated instance of an HMP model
    duration : bool
        Whether to compute peak location (False) or inter-peak duration (True)
    mean : bool
        Whether to compute the mean (True) or return the single trial estimates
        Note that mean and errorbars cannot both be true.
    add_rt : bool
        whether to append the last stage up to the RT
    as_time : bool
        if true, return time (ms) instead of samples
    errorbars : str
        calculate 95% confidence interval ('ci'), standard deviation ('std'),
        standard error ('se') on the times or durations, or None.
        Note that mean and errorbars cannot both be true.
    estimate_method : string
        'max' or 'mean', either take the max probability of each event on each trial, or the
        weighted average.
    add_stim: bool
        Adding stimulus as the first event (True) or let the first estimated HMP event be the
        first one (False, default)

    Returns
    -------
    times : xr.DataArray
        Transition event peak or stage duration with trial_x_participant*event dimensions or
        only event dimension if mean = True contains nans for missing stages.
    """
    assert not (mean and errorbars is not None), "Only one of mean and errorbars can be set."
    tstep = 1000 / estimates.sfreq if as_time else 1
    
    if estimate_method is None:
        estimate_method = "max"
    event_shift = 0
    eventprobs = estimates.fillna(0).copy()
    if estimate_method == "max":
        times = eventprobs.argmax("samples") - event_shift  # Most likely event location
    else:
        times = xr.dot(eventprobs, eventprobs.samples, dims="samples") - event_shift
    times = times.astype("float32")  # needed for eventual addition of NANs
    times_level = (
        times.groupby("levels").mean("trial_x_participant").values
    )  # take average to make sure it's not just 0 on the trial-level
    for c, e in np.argwhere(times_level == -event_shift):
        times[times["levels"] == c, e] = np.nan
    
    if add_rt:
        rts = estimates.cumsum('samples').argmax('samples').max('event')+1
        rts = xr.DataArray(rts)
        rts = rts.assign_coords(event=int(times.event.max().values + 1))
        rts = rts.expand_dims(dim="event")
        times = xr.concat([times, rts], dim="event")

    times = times * tstep     
    if duration:  # taking into account missing events, hence the ugly code
        added = xr.DataArray(
            np.repeat(0, len(times.trial_x_participant))[np.newaxis, :],
            coords={"event": [0], "trial_x_participant": times.trial_x_participant},
        )
        times = times.assign_coords(event=times.event + 1)
        times = times.combine_first(added)
        for c in np.unique(times["levels"].values):
            tmp = times.isel(trial_x_participant=estimates["levels"] == c).values
            # identify nan columns == missing events
            missing_evts = np.where(np.isnan(np.mean(tmp, axis=0)))[0]
            tmp = np.diff(
                np.delete(tmp, missing_evts, axis=1)
            )  # remove 0 columns, calc difference
            # insert nan columns (to maintain shape),
            for missing in missing_evts:
                tmp = np.insert(tmp, missing - 1, np.nan, axis=1)
            # add extra column to match shape
            tmp = np.hstack((tmp, np.tile(np.nan, (tmp.shape[0], 1))))
            times[estimates["levels"] == c, :] = tmp
        times = times[:, :-1]  # remove extra column
    elif add_stim:
        added = xr.DataArray(
            np.repeat(0, len(times.trial_x_participant))[np.newaxis, :],
            coords={"event": [0], "trial_x_participant": times.trial_x_participant},
        )
        times = times.assign_coords(event=times.event + 1)
        times = times.combine_first(added)

    if mean:
        times = times.groupby("levels").mean("trial_x_participant")
    elif errorbars:
        errorbars_model = np.zeros((len(np.unique(times["levels"])), 2, times.shape[1]))
        if errorbars == "std":
            std_errs = times.groupby("levels").reduce(np.std, dim="trial_x_participant").values
            for c in np.unique(times["levels"]):
                errorbars_model[c, :, :] = np.tile(std_errs[c, :], (2, 1))
        else:
            raise ValueError(
                "Unknown error bars, 'std' is for now the only accepted argument in the "
                "multilevel models"
            )
        times = errorbars_model
    return times


def event_topo(
    epoch_data,
    estimated,
    mean=True,
    peak=True,
    estimate_method="max",
    template=None,
):
    """Compute topographies for each trial.

    Parameters
    ----------
        epoch_data: xr.Dataset
            Epoched data
        estimated: xr.Dataset
            estimated model parameters and event probabilities
        mean: bool
            if True mean will be computed instead of single-trial channel activities
        peak : bool
            if true, return topography at peak of the event. If false, return topographies weighted
            by a normalized template.
        estimate_method : string
            'max' or 'mean', either take the max probability of each event on each trial, or the
            weighted average.
        template: int
            Length of the pattern in samples (e.g. 5 for a pattern of 50 ms with a 100Hz sampling
            frequency)

    Returns
    -------
        event_values: xr.DataArray
            array containing the values of each electrode at the most likely transition time
            contains nans for missing events
    """
    if estimate_method is None:
        estimate_method = "max"
    epoch_data = (
        epoch_data.rename({"epochs": "trials"})
        .stack(trial_x_participant=["participant", "trials"])
        .data
        .drop_duplicates("trial_x_participant")
    )

    n_events = estimated.event.count().values
    n_trials = estimated.trial_x_participant.count().values
    n_channels = epoch_data.channels.count().values

    common_trials = np.intersect1d(
        estimated["trial_x_participant"].values, epoch_data["trial_x_participant"].values
    )
    epoch_data = epoch_data.sel(trial_x_participant=common_trials)
    estimated = estimated.sel(trial_x_participant=common_trials)
    if not peak:
        normed_template = template / np.sum(template)

    times = event_times(estimated, mean=False, estimate_method=estimate_method,)
    
    event_values = np.zeros((n_channels, n_trials, n_events))*np.nan
    for ev in range(n_events):
        for tr in range(n_trials):
            # If time is nan, means that no event was estimated for that trial/level
            if np.isfinite(times.values[tr, ev]):
                samp = int(times.values[tr, ev])
                if peak:
                    event_values[:, tr, ev] = epoch_data.values[:, samp, tr]
                else:
                    vals = epoch_data.values[:, samp : samp + template // 2, tr]
                    event_values[:, tr, ev] = np.dot(vals, normed_template[: vals.shape[1]])

    event_values = xr.DataArray(
        event_values,
        dims=[
            "channels",
            "trial_x_participant",
            "event",
        ],
        coords={
            "trial_x_participant": estimated.trial_x_participant,
            "event": estimated.event,
            "channels": epoch_data.channels,
        },
    )

    event_values = event_values.assign_coords(
        levels=("trial_x_participant", times.levels.data)
    )

    if mean:
        event_values = event_values.groupby("levels").mean("trial_x_participant")
    return event_values


def centered_activity(
    data,
    times,
    channels,
    event,
    n_samples=None,
    center=True,
    cut_after_event=0,
    baseline=0,
    cut_before_event=0,
    event_width=0,
    impute=None,
):
    """Parse the single trial signal of channels in a given number of samples around one event.

    Parameters
    ----------
    data : xr.Dataset
        HMP data (untransformed but with trial and participant stacked)
    times : xr.DataArray
        Onset times as computed using onset_times()
    channels : list
        channels to pick for the parsing of the signal, must be a list even if only one
    event : int
        Which event is used to parse the signal
    n_samples : int
        How many samples to record after the event (default = maximum duration between event and
        the consecutive event)
    cut_after_event: int
        Which event after ```event``` to cut samples off, if 1 (Default) cut at the next event
    baseline: int
        How much samples should be kept before the event
    cut_before_event: int
        At which previous event to cut samples from, ```baseline``` if 0 (Default), no effect if
        baseline = 0
    event_width: int
        Duration of the fitted events, used when cut_before_event is True

    Returns
    -------
    centered_data : xr.Dataset
        Xarray dataset with electrode value (data) and trial event time (time) and with
        trial_x_participant * samples dimension
    """
    if event == 0:  # no samples before stim onset
        baseline = 0
    elif event == 1:  # no event at stim onset
        event_width = 0
    if cut_before_event == 0:  # avoids searching before stim onset
        cut_before_event = event
    if n_samples is None:
        if cut_after_event is None:
            raise ValueError(
                "One of ```n_samples``` or ```cut_after_event``` has to be filled to use an upper"
                "limit"
            )
        n_samples = (
            max(times.sel(event=event + cut_after_event).data - times.sel(event=event).data) + 1
        )
    if impute is None:
        impute = np.nan
    if center:
        centered_data = np.tile(
            impute,
            (len(data.trial_x_participant), len(channels), int(round(n_samples - baseline + 1))),
        )
    else:
        centered_data = np.tile(
            impute, (len(data.trial_x_participant), len(channels), len(data.samples))
        )

    i = 0
    trial_times = np.zeros(len(data.trial_x_participant)) * np.nan
    valid_indices = list(times.groupby("trial_x_participant", squeeze=False).groups.keys())
    for trial, trial_dat in data.groupby("trial_x_participant", squeeze=False):
        if trial in valid_indices:
            if cut_before_event > 0:
                # Lower lim is baseline or the last sample of the previous event
                lower_lim = np.max(
                    [
                        -np.max(
                            [
                                times.sel(event=event, trial_x_participant=trial)
                                - times.sel(
                                    event=event - cut_before_event, trial_x_participant=trial
                                )
                                - event_width // 2,
                                0,
                            ]
                        ),
                        baseline,
                    ]
                )
            else:
                lower_lim = 0
            if cut_after_event > 0:
                upper_lim = np.max(
                    [
                        np.min(
                            [
                                times.sel(event=event + cut_after_event, trial_x_participant=trial)
                                - times.sel(event=event, trial_x_participant=trial)
                                - event_width // 2,
                                n_samples,
                            ]
                        ),
                        0,
                    ]
                )
            else:
                upper_lim = n_samples

            # Determine samples in the signal to store
            start_idx = int(times.sel(event=event, trial_x_participant=trial) + lower_lim)
            end_idx = int(times.sel(event=event, trial_x_participant=trial) + upper_lim)
            trial_time = slice(start_idx, end_idx)
            trial_time_idx = slice(start_idx, end_idx + 1)
            trial_elec = trial_dat.sel(channels=channels, samples=trial_time).squeeze(
                "trial_x_participant"
            )
            # If center, adjust to always center on the same sample if lower_lim < baseline
            baseline_adjusted_start = int(abs(baseline - lower_lim))
            baseline_adjusted_end = baseline_adjusted_start + trial_elec.shape[-1]
            trial_time_arr = slice(baseline_adjusted_start, baseline_adjusted_end)

            if center:
                centered_data[i, :, trial_time_arr] = trial_elec
            else:
                centered_data[i, :, trial_time_idx] = trial_elec
            trial_times[i] = times.sel(event=event, trial_x_participant=trial)
            i += 1

    part, trial = data.coords["participant"].values, data.coords["epochs"].values
    trial_x_part = xr.Coordinates.from_pandas_multiindex(
        MultiIndex.from_arrays([part, trial], names=("participant", "trials")),
        "trial_x_participant",
    )
    centered_data = xr.Dataset(
        {
            "data": (("trial_x_participant", "channel", "samples"), centered_data),
            "times": (("trial_x_participant"), trial_times),
        },
        {"channel": channels, "samples": np.arange(centered_data.shape[-1]) + baseline},
        attrs={"event": event},
    )

    return centered_data.assign_coords(trial_x_part)


def condition_selection(hmp_data, condition_string, variable="event", method="equal"):
    """Select a subset from hmp_data.

    The function selects epochs for which 'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    hmp_data : xr.Dataset
        transformed EEG data for hmp, from utils.transform_data
    condition_string : str | num
        condition indicator for selection
    variable : str
        variable present in hmp_data that is used for condition selection
    method : str
        'equal' selects equal trials, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    data : xr.Dataset
        Subset of hmp_data.
    """
    unstacked = hmp_data.unstack()
    unstacked[variable] = unstacked[variable].fillna("")
    if method == "equal":
        unstacked = unstacked.where(unstacked[variable] == condition_string, drop=True)
        stacked = stack_data(unstacked)
    elif method == "contains":
        unstacked = unstacked.where(unstacked[variable].str.contains(condition_string), drop=True)
        stacked = stack_data(unstacked)
    else:
        warn("unknown method, returning original data")
        stacked = hmp_data
    return stacked


def condition_selection_epoch(epoch_data, condition_string, variable="event", method="equal"):
    """Select a subset from epoch_data.

    The function selects epochs for which 'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    epoch_data : xr.Dataset
        transformed EEG data for hmp, e.g. from utils.read_mne_data()
    condition_string : str | num
        condition indicator for selection
    variable : str
        variable present in hmp_data that is used for condition selection
    method : str
        'equal' selects equal trials, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    data : xr.Dataset
        Subset of hmp_data.
    """
    if len(epoch_data.dims) == 4:
        stacked_epoch_data = epoch_data.stack(trial_x_participant=("participant", "epochs")).dropna(
            "trial_x_participant", how="all"
        )

    if method == "equal":
        stacked_epoch_data = stacked_epoch_data.where(
            stacked_epoch_data[variable] == condition_string, drop=True
        )
    elif method == "contains":
        stacked_epoch_data = stacked_epoch_data.where(
            stacked_epoch_data[variable].str.contains(condition_string), drop=True
        )
    return stacked_epoch_data.unstack()


def participant_selection(hmp_data, participant):
    """Select a participant from hmp_data.

    Parameters
    ----------
    hmp_data : xr.Dataset
        transformed EEG data for hmp, from utils.transform_data
    participant : str | num
        Name of the participant

    Returns
    -------
    data : xr.Dataset
        Subset of hmp_data.
    """
    unstacked = hmp_data.unstack().sel(participant=participant)
    stacked = stack_data(unstacked)
    return stacked
