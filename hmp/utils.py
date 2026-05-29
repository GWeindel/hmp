"""Functions to transform the input data and the estimates."""

from warnings import warn

import numpy as np
import xarray as xr
from mne import EpochsArray, Info, pick_info, pick_types
from mne.io.constants import FIFF
from mne.preprocessing import compute_current_source_density
from numpy.random import RandomState
from pandas import MultiIndex

from hmp.transformers.custom import ProjCustom
from hmp.transformers.identity import ProjIdentity
from hmp.transformers.pca import ProjPCA


def _check_transformed(transformed):
    if isinstance(transformed, (ProjPCA, ProjIdentity, ProjCustom)):
        data = transformed.data
    elif 'component' in transformed.dims:
        data = transformed
    else:
        raise ValueError("transformed must be an hmp transformed object from a class"
                             "in hmp.transformers")
    return data

def _check_sf_consistency(epoch_data, estimates):
    if epoch_data.sfreq != estimates.sfreq:
        raise ValueError("Inconsistent sampling frequency between epoch data and estimates")

def event_times(  # noqa: PLR0912
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
    elif errorbars:
        errorbars_model = np.zeros((len(np.unique(times["group"])), 2, times.shape[1]))
        if errorbars == "std":
            std_errs = times.groupby("group").reduce(np.std, dim="trial").values
            for c in np.unique(times["group"]):
                errorbars_model[c, :, :] = np.tile(std_errs[c, :], (2, 1))
        else:
            raise ValueError(
                "Unknown error bars, 'std' is for now the only accepted argument in the "
                "multigroup models"
            )
        times = errorbars_model
    return times


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
    epoch_data = (
        epoch_data.stack(trial=["participant", "epoch"])
        .data
        .drop_duplicates("trial")
    )

    common_trial = np.intersect1d(
        estimates["trial"].values, epoch_data["trial"].values
    )
    epoch_data = epoch_data.sel(trial=common_trial, sample=estimates.sample)
    estimates = estimates.sel(trial=common_trial)
    n_events = estimates.event.count().values
    n_trial = estimates.trial.count().values
    n_channel = epoch_data.channel.count().values

    if not peak:
        normed_template = template / np.sum(template)

    times = event_times(estimates, mean=False, estimate_method=estimate_method,)
    times = times.sel(trial=common_trial)
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
    data,
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
    data : xr.Dataset
        HMP data (untransformed but with trial and participant stacked)
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
    if 'epoch' in data.dims:
        data = (
            data.stack({'trial':['participant','epoch']})
            .data
        )
    mask = ~data.isel(sample=0, channel=0).squeeze().isnull()
    data = data.sel(trial=data.trial.values[mask])


    common_trial = np.intersect1d(
        times["trial"].values, data["trial"].values
    )
    data = data.sel(trial=common_trial)
    times = times.sel(trial=common_trial)

    assert ~np.any(times > data.sample.max()),\
        "At least one trial is longer than the maximum possible sample.\
        Provided times should be in sample not on the millisecond scale"

    centered_data = np.tile(
        np.nan,
        (len(common_trial), len(channel), int(round(n_samples - baseline + 1))),
    )

    trial_times = np.zeros(len(common_trial)) * np.nan
    participants = []
    epochs = np.zeros(len(common_trial))
    for i, (trial, trial_dat) in enumerate(data.groupby("trial", squeeze=False)):
        participants.append(trial[0])
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
        MultiIndex.from_arrays([participants, epochs], names=("participant", "epoch")),
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


def condition_selection(transformed, condition_string, variable="event", method="equal"):
    """Select a subset from transformed_data.

    The function selects epochs for which 'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    transformed : xr.Dataset
        transformed EEG data for hmp from the hmp.preprocessing classes
    condition_string : str | num
        condition indicator for selection
    variable : str
        variable present in transformed.data that is used for condition selection
    method : str
        'equal' selects equal trial, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    data : xr.Dataset
        Subset of transformed_data.
    """
    data = _check_transformed(transformed).unstack()
    data[variable] = data[variable].fillna("")
    if method == "equal":
        data = data.where(data[variable] == condition_string, drop=True)
    elif method == "contains":
        data = data.where(data[variable].str.contains(condition_string), drop=True)
    else:
        warn("unknown method, returning original data")
    return data.stack(trial=['participant','epoch'])


def condition_selection_epoch(epoch_data, condition_string, variable="event", method="equal"):
    """Select a subset from epoch_data.

    The function selects epochs for which 'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    epoch_data : xr.Dataset
        Epoched EEG data for hmp
    condition_string : str | num
        condition indicator for selection
    variable : str
        variable present in transformed_data that is used for condition selection
    method : str
        'equal' selects equal trial, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    data : xr.Dataset
        Subset of transformed_data.
    """
    if len(epoch_data.dims) == 4:
        stacked_epoch_data = epoch_data.stack(trial=("participant", "epoch"))
        mask = ~stacked_epoch_data.data.isel(sample=0, channel=0).squeeze().isnull()
        stacked_epoch_data = stacked_epoch_data.sel(trial=stacked_epoch_data.trial.values[mask])
    else:
        raise ValueError(
            "Unexpected data object. Expected an xarray dataset with dimensions:"
            "participant, epoch, channel, sample"
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


def participant_selection(transformed, participant):
    """Select a participant from transformed_data.

    Parameters
    ----------
    transformed : xr.Dataset or hmp.transformers
        transformed EEG data for hmp
    participant : str | num
        Name of the participant

    Returns
    -------
    data : xr.Dataset
        Subset of transformed_data.
    """
    data = _check_transformed(transformed).unstack()
    data = data.sel(participant=participant, drop=False)
    if 'participant' not in data.dims:
        data = data.expand_dims('participant')
    return data.stack(trial=['participant','epoch'])

def compute_csd(epoch_data: xr.Dataset,
                info: Info):
    """Compute laplacian using MNE's function.

    Parameters
    ----------
    epoch_data : xr.Dataset
        Data read through the HMP IO module
    info : Info
        Info object from MNE

    Returns
    -------
    epoch_data : xr.Dataset
        Updated dataset with CSD values
    eeg_info: Info
        Updated info ubject with correct units given CSD transform
    """
    eeg_info = pick_info(info, pick_types(info, meg=False, eeg=True))
    if eeg_info['chs'][0]['unit'] == FIFF.FIFF_UNIT_V:
        epoch_data = epoch_data.stack(trial=['participant','epoch']).dropna("trial", how="all")
        for trial in epoch_data.trial:
            trial_dat = epoch_data.sel(trial=trial).data
            # Build fake Epoch mne class and use MNE's dedicated function
            epoch = EpochsArray(np.array([trial_dat.values]), eeg_info)
            epoch = compute_current_source_density(epoch)
            epoch_data['data'].loc[dict(trial=trial)] = epoch.get_data()[0]
        epoch_data = epoch_data.unstack()
        # Set EEG channels to the correct CSD unit

        for ch in eeg_info['chs']:
            ch['unit'] = FIFF.FIFF_UNIT_V_M2

    else:
        raise ValueError(f"Cannot apply CSD on channels with units {info['chs'][0]['unit']}")
    return epoch_data, eeg_info

def _define_random_state(seed=None):
    if seed is not None:
        random_state = RandomState(seed)
    else:
        random_state = RandomState(np.random.randint(low=0, high=3000))
    return random_state


