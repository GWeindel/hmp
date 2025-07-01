"""Functions to transform the input data and the estimates."""

from warnings import filterwarnings, warn

import numpy as np
import pandas as pd
import xarray as xr
from pandas import MultiIndex

from hmp import mcca

filterwarnings(
    "ignore",
    "Degrees of freedom <= 0 for slice.",
)  # weird warning, likely due to nan in xarray, not important but better fix it later
filterwarnings("ignore", "Mean of empty slice")  # When trying to center all-nans trial


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

    Going from format [participant * epochs * sample * channel] to
    [sample * channel] with sample indexes starts and ends to delimitate the epochs.


    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with
        dimensions [participant * epochs * sample * channel]
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
    data = data.stack(all_samples=["participant", "epoch", "sample"]).dropna(dim="all_samples")
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
    remove_offset=False,
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
        if true, return time (ms) instead of sample
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
    remove_offset: bool
        Whether to remove the eventual offset added to the reaction time

    Returns
    -------
    times : xr.DataArray
        Transition event peak or stage duration with trial*event dimensions or
        only event dimension if mean = True contains nans for missing stages.
    """
    assert not (mean and errorbars is not None), "Only one of mean and errorbars can be set."
    tstep = 1000 / estimates.sfreq if as_time else 1
    
    if estimate_method is None:
        estimate_method = "max"
    event_shift = 0
    eventprobs = estimates.fillna(0).copy()
    if estimate_method == "max":
        times = eventprobs.argmax("sample") - event_shift  # Most likely event location
    else:
        times = xr.dot(eventprobs, eventprobs.sample, dims="sample") - event_shift
    times = times.astype("float32")  # needed for eventual addition of NANs
    times_level = (
        times.groupby("level").mean("trial").values
    )  # take average to make sure it's not just 0 on the trial-level
    for c, e in np.argwhere(times_level == -event_shift):
        times[times["level"] == c, e] = np.nan
    
    if add_rt:
        rts = estimates.cumsum('sample').argmax('sample').max('event')+1
        if remove_offset:
            rts = rts-estimates.offset
        rts = xr.DataArray(rts)
        rts = rts.assign_coords(event=int(times.event.max().values + 1))
        rts = rts.expand_dims(dim="event")
        times = xr.concat([times, rts], dim="event")

    times = times * tstep     
    if duration:  # taking into account missing events, hence the ugly code
        added = xr.DataArray(
            np.repeat(0, len(times.trial))[np.newaxis, :],
            coords={"event": [0], "trial": times.trial},
        )
        times = times.assign_coords(event=times.event + 1)
        times = times.combine_first(added)
        for c in np.unique(times["level"].values):
            tmp = times.isel(trial=estimates["level"] == c).values
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
            times[estimates["level"] == c, :] = tmp
        times = times[:, :-1]  # remove extra column
    elif add_stim:
        added = xr.DataArray(
            np.repeat(0, len(times.trial))[np.newaxis, :],
            coords={"event": [0], "trial": times.trial},
        )
        times = times.assign_coords(event=times.event + 1)
        times = times.combine_first(added)

    if mean:
        times = times.groupby("level").mean("trial")
    elif errorbars:
        errorbars_model = np.zeros((len(np.unique(times["level"])), 2, times.shape[1]))
        if errorbars == "std":
            std_errs = times.groupby("level").reduce(np.std, dim="trial").values
            for c in np.unique(times["level"]):
                errorbars_model[c, :, :] = np.tile(std_errs[c, :], (2, 1))
        else:
            raise ValueError(
                "Unknown error bars, 'std' is for now the only accepted argument in the "
                "multilevel models"
            )
        times = errorbars_model
    return times


def event_channels(
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
            Length of the pattern in sample (e.g. 5 for a pattern of 50 ms with a 100Hz sampling
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
        epoch_data.stack(trial=["participant", "epoch"])
        .data
        .drop_duplicates("trial")
    )

    n_events = estimated.event.count().values
    n_trial = estimated.trial.count().values
    n_channel = epoch_data.channel.count().values

    common_trial = np.intersect1d(
        estimated["trial"].values, epoch_data["trial"].values
    )
    epoch_data = epoch_data.sel(trial=common_trial)
    estimated = estimated.sel(trial=common_trial)
    if not peak:
        normed_template = template / np.sum(template)

    times = event_times(estimated, mean=False, estimate_method=estimate_method,)
    
    event_values = np.zeros((n_channel, n_trial, n_events))*np.nan
    for ev in range(n_events):
        for tr in range(n_trial):
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
            "channel",
            "trial",
            "event",
        ],
        coords={
            "trial": estimated.trial,
            "event": estimated.event,
            "channel": epoch_data.channel,
        },
    )

    event_values = event_values.assign_coords(
        level=("trial", times.level.data)
    )

    if mean:
        event_values = event_values.groupby("level").mean("trial")
    return event_values


def centered_activity(
    data,
    times,
    channel,
    event,
    n_samples=None,
    center=True,
    cut_after_event=0,
    baseline=0,
    cut_before_event=0,
    event_width=0,
    impute=None,
):
    """Parse the single trial signal of channel in a given number of sample around one event.

    Parameters
    ----------
    data : xr.Dataset
        HMP data (untransformed but with trial and participant stacked)
    times : xr.DataArray
        Onset times as computed using onset_times()
    channel : list
        channel to pick for the parsing of the signal, must be a list even if only one
    event : int
        Which event is used to parse the signal
    n_samples : int
        How many sample to record after the event (default = maximum duration between event and
        the consecutive event)
    cut_after_event: int
        Which event after ```event``` to cut sample off, if 1 (Default) cut at the next event
    baseline: int
        How much sample should be kept before the event
    cut_before_event: int
        At which previous event to cut sample from, ```baseline``` if 0 (Default), no effect if
        baseline = 0
    event_width: int
        Duration of the fitted events, used when cut_before_event is True

    Returns
    -------
    centered_data : xr.Dataset
        Xarray dataset with electrode value (data) and trial event time (time) and with
        trial * sample dimension
    """
    if event == 0:  # no sample before stim onset
        baseline = 0
    elif event == 1:  # no event at stim onset
        event_width = 0
    if cut_before_event == 0:  # avoids searching before stim onset
        cut_before_event = event
    if 'epoch' in data.dims:
        data = data.stack({'trial':['participant','epoch']}).data
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
            (len(data.trial), len(channel), int(round(n_samples - baseline + 1))),
        )
    else:
        centered_data = np.tile(
            impute, (len(data.trial), len(channel), len(data.sample))
        )

    i = 0
    trial_times = np.zeros(len(data.trial)) * np.nan
    valid_indices = list(times.groupby("trial", squeeze=False).groups.keys())
    for trial, trial_dat in data.groupby("trial", squeeze=False):
        if trial in valid_indices:
            if cut_before_event > 0:
                # Lower lim is baseline or the last sample of the previous event
                lower_lim = np.max(
                    [
                        -np.max(
                            [
                                times.sel(event=event, trial=trial)
                                - times.sel(
                                    event=event - cut_before_event, trial=trial
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
                                times.sel(event=event + cut_after_event, trial=trial)
                                - times.sel(event=event, trial=trial)
                                - event_width // 2,
                                n_samples,
                            ]
                        ),
                        0,
                    ]
                )
            else:
                upper_lim = n_samples

            # Determine sample in the signal to store
            start_idx = int(times.sel(event=event, trial=trial) + lower_lim)
            end_idx = int(times.sel(event=event, trial=trial) + upper_lim)
            trial_time = slice(start_idx, end_idx)
            trial_time_idx = slice(start_idx, end_idx + 1)
            trial_elec = trial_dat.sel(channel=channel, sample=trial_time).squeeze(
                "trial"
            )
            # If center, adjust to always center on the same sample if lower_lim < baseline
            baseline_adjusted_start = int(abs(baseline - lower_lim))
            baseline_adjusted_end = baseline_adjusted_start + trial_elec.shape[-1]
            trial_time_arr = slice(baseline_adjusted_start, baseline_adjusted_end)

            if center:
                centered_data[i, :, trial_time_arr] = trial_elec
            else:
                centered_data[i, :, trial_time_idx] = trial_elec
            trial_times[i] = times.sel(event=event, trial=trial)
            i += 1

    part, trial = data.coords["participant"].values, data.coords["epoch"].values
    trial_x_part = xr.Coordinates.from_pandas_multiindex(
        MultiIndex.from_arrays([part, trial], names=("participant", "epoch")),
        "trial",
    )
    centered_data = xr.Dataset(
        {
            "data": (("trial", "channel", "sample"), centered_data),
            "times": (("trial"), trial_times),
        },
        {"channel": channel, "sample": np.arange(centered_data.shape[-1]) + baseline},
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
        'equal' selects equal trial, 'contains' selects trial in which conditions_string
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
        'equal' selects equal trial, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    data : xr.Dataset
        Subset of hmp_data.
    """
    if len(epoch_data.dims) == 4:
        stacked_epoch_data = epoch_data.stack(trial=("participant", "epoch")).dropna(
            "trial", how="all"
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
