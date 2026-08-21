"""Functions to transform the input data and the estimates."""

import multiprocessing as mp
from typing import Callable

import numpy as np
import xarray as xr
from numpy.random import RandomState
from pandas import MultiIndex


def _check_sf_consistency(epoch_data, estimates):
    if epoch_data.sfreq != estimates.sfreq:
        raise ValueError("Inconsistent sampling frequency between epoch data and estimates")

def event_times(  # noqa: PLR0912
    estimates,
    duration=False,
    mean=False,
    add_rt=False,
    as_time=False,
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
    tstep = 1000 / estimates.sfreq if as_time else 1

    if estimate_method is None:
        estimate_method = "max"
    event_shift = 0
    eventprobs = estimates.fillna(0).copy()
    if estimate_method == "max":
        times = eventprobs.argmax("sample") - event_shift  # Most likely event location
    else:
        times = xr.dot(eventprobs, eventprobs.sample, dims="sample") - event_shift
    times = times.astype("float64")  # needed for eventual addition of NANs
    times_group = (
        times.groupby("group").mean("trial").values
    )  # take average to make sure it's not just 0 on the trial-group
    for c, e in np.argwhere(times_group == -event_shift):
        times[times["group"] == c, e] = np.nan

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
        for c in np.unique(times["group"].values):
            tmp = times.isel(trial=estimates["group"] == c).values
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
            times[estimates["group"] == c, :] = tmp
        times = times[:, :-1]  # remove extra column
    elif add_stim:
        added = xr.DataArray(
            np.repeat(0, len(times.trial))[np.newaxis, :],
            coords={"event": [0], "trial": times.trial},
        )
        times = times.assign_coords(event=times.event + 1)
        times = times.combine_first(added)

    if mean:
        times = times.groupby("group").mean("trial")

    return times

def _filter_common_trials_data_fit(epoch_data, estimates):
    if len(epoch_data.dims) == 4:
        epoch_data = epoch_data.stack(trial=("recording", "epoch"))
    mask = ~epoch_data.data.isel(sample=0, channel=0).drop_vars(['sample','channel']).isnull()
    epoch_data = epoch_data.sel(trial=epoch_data.trial.values[mask])
    common_trial = np.intersect1d(
        estimates["trial"].values, epoch_data["trial"].values
    )
    if 'sample' in estimates.dims: #This is eventprobs
        epoch_data = epoch_data.sel(trial=common_trial, sample=estimates.sample)\
            .data.dropna(dim="trial", how="all")
    else: #Secondary estimates, e.g. times
        epoch_data = epoch_data.sel(trial=common_trial)\
            .data.dropna(dim="trial", how="all")
    estimates = estimates.sel(trial=common_trial).dropna(dim="trial", how="all")
    return epoch_data, estimates

def event_channels(
    epoch_data,
    estimates,
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
        estimates: xr.Dataset
            estimated model parameters and event probabilities
        mean: bool
            if True mean will be computed instead of single-trial channel activities
        peak : bool
            if true, return topography at peak of the event. If false, return topographies weighted
            by a normalized template.
        estimate_method : string
            'max' or 'mean', either take the max probability of each event on each trial, or the
            weighted average.
        template: np.array
            Expected shape of the event, typically the template attribute from hmp.patterns

    Returns
    -------
        event_values: xr.DataArray
            array containing the values of each electrode at the most likely transition time
            contains nans for missing events
    """
    _check_sf_consistency(epoch_data, estimates)
    if estimate_method is None:
        estimate_method = "max"

    epoch_data, estimates = _filter_common_trials_data_fit(epoch_data, estimates)
    n_events = estimates.event.count().values
    n_trial = estimates.trial.count().values
    n_channel = epoch_data.channel.count().values

    if not peak:
        normed_template = template / np.sum(template)

    times = event_times(estimates, mean=False, estimate_method=estimate_method,)
    event_values = np.zeros((n_channel, n_trial, n_events))*np.nan
    for ev in range(n_events):
        for tr in range(n_trial):
            # If time is nan, means that no event was estimated for that trial/group
            if np.isfinite(times.values[tr, ev]):
                samp = int(times.values[tr, ev])
                if peak:
                    event_values[:, tr, ev] = epoch_data.values[:, samp, tr]
                else:
                    vals = epoch_data.values[:, samp : samp + len(template) // 2, tr]
                    event_values[:, tr, ev] = np.dot(vals, normed_template[: vals.shape[1]])

    event_values = xr.DataArray(
        event_values,
        dims=[
            "channel",
            "trial",
            "event",
        ],
        coords={
            "trial": estimates.trial,
            "event": estimates.event,
            "channel": epoch_data.channel,
        },
    )

    event_values = event_values.assign_coords(
        group=("trial", times.group.data)
    )

    if mean:
        event_values = event_values.groupby("group").mean("trial")
    return event_values


def centered_activity(
    epoch_data,
    times,
    channel,
    event,
    n_samples=None,
    cut_after_event=0,
    baseline=0,
    cut_before_event=0,
    event_width=0,
):
    """Parse the single trial signal of channel in a given number of sample around one event.

    Parameters
    ----------
    epoch_data : xr.Dataset
        epoch_data from hmp.io
    times : xr.DataArray
        Onset times in sample as computed using event_times()
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
    if n_samples is None:
        if cut_after_event is None:
            raise ValueError(
                "One of ```n_samples``` or ```cut_after_event``` has to be filled to use an upper"
                "limit"
            )
        n_samples = (
            max(times.sel(event=event + cut_after_event).data - times.sel(event=event).data) + 1
        )

    n_samples = np.rint(n_samples)
    baseline = np.rint(baseline)
    epoch_data, times = _filter_common_trials_data_fit(epoch_data, times)

    assert ~np.any(times > epoch_data.sample.max()),\
        "At least one trial is longer than the maximum possible sample.\
        Provided times should be in sample not on the millisecond scale"

    centered_data = np.tile(
        np.nan,
        (epoch_data.sizes['trial'],
         len(channel),
         int(round(n_samples - baseline + 1))),
    )

    trial_times = np.zeros(epoch_data.sizes['trial']) * np.nan
    recordings = []
    epochs = np.zeros(epoch_data.sizes['trial'],)
    for i, (trial, trial_dat) in enumerate(epoch_data.groupby("trial", squeeze=False)):
        recordings.append(trial[0])
        epochs[i] = trial[1]
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
            lower_lim = baseline
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
        trial_elec = trial_dat.sel(channel=channel, sample=slice(start_idx, end_idx))\
            .squeeze("trial")
        # If requested bsl or n_samples exceed epoch window
        offshoot_bsl = start_idx - trial_elec.sample[0].values
        offshoot_epo = end_idx - trial_elec.sample[-1].values
        # If center, adjust to always center on the same sample if lower_lim > baseline
        start_idx_data = int(lower_lim - baseline - offshoot_bsl)
        end_idx_data = int(upper_lim - baseline + 1 - offshoot_epo)
        trial_time_arr = slice(start_idx_data, end_idx_data)

        centered_data[i, :, trial_time_arr] = trial_elec
        trial_times[i] = times.sel(event=event, trial=trial)

    trial_x_part = xr.Coordinates.from_pandas_multiindex(
        MultiIndex.from_arrays([recordings, epochs], names=("recording", "epoch")),
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

def _sel_method(data, value, variable, method):
    if variable in data.coords:
        result = method(data[variable], value)
        if result.dtype != bool:
            raise ValueError(
                "Unsupported method. Use a callable that returns boolean."
            )
        attrs = data.attrs.copy()
        data = data.where(result, drop=True)
        data.attrs = attrs
    else:
        raise ValueError(f"{variable} not found in data")
    return data

def _coordsel_data(epoch_data, value, variable, method):
    if len(epoch_data.dims) == 4:
        stacked_epoch_data = epoch_data.stack(trial=("recording", "epoch"))
        # Faster and less RAM
        mask = ~stacked_epoch_data.data.isel(sample=0, channel=0).squeeze().isnull()
        stacked_epoch_data = stacked_epoch_data.sel(trial=stacked_epoch_data.trial.values[mask])
    else:
        raise ValueError(
            "Unexpected epoch_data object. Expected an xarray dataset with dimensions:"
            "recording, epoch, channel, sample"
        )

    stacked_epoch_data = _sel_method(stacked_epoch_data, value, variable, method)
    return stacked_epoch_data.unstack()


def select_coord(data: xr.Dataset | xr.DataArray,
                    value: object,
                    variable: str,
                    method: Callable[[xr.DataArray, object], xr.DataArray] = np.equal,
                    copy: bool = True
                   ):
    """Select a subset from the data or estimates using the specified coordinate(s).

    The function selects trials where `method(data[variable], value)` is True.
    You can either use functions returning booleans or a custom function
    using lambda, e.g. `method=lambda x, v: ~x.isin(v)`

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Data from io or estimates from hmp
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
    data : xr.Dataset
        Subset of data.
    """
    if copy:
        data = data.copy(deep=True)
    if 'epoch' in data.dims and "channel" in data.dims: #Epoch_data stack, select, then unstack
        data = _coordsel_data(data, value, variable, method)
    elif 'trial' in data.dims: #HMP outputs, already stacked
        data = _sel_method(data, value, variable, method).dropna(dim="trial", how="all")
    else:
        raise ValueError('Unexpected data type')
    return data

def _define_random_state(seed=None):
    if seed is not None:
        random_state = RandomState(seed)
    else:
        random_state = RandomState(np.random.randint(low=0, high=3000))
    return random_state

def _get_mp_context():
    available_methods = mp.get_all_start_methods()
    for method in ["fork", "forkserver", "spawn"]:
        if method in available_methods:
            return mp.get_context(method)