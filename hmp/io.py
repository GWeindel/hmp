"""EEG/MEG Data Processing Utilities.

This module provides functions for reading, processing, and saving EEG/MEG data using MNE, xarray,
and pandas.
It supports reading raw or epoched data, event/response detection, reaction time trimming,
epoch cropping,
metadata handling, and conversion to xarray Datasets for fitting hmp models.
Additional utilities are provided for saving/loading data and models,
and exporting event probabilities.
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Optional

import mne
import mne_bids
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from pandas import DataFrame

def read_bids_data(
    root: str | mne_bids.BIDSPath,
    datatype: str,
    stimulus_id: dict | None,
    response_id: dict | None,
    tmin: float,
    tmax: float,
    high_pass: float | None = None,
    low_pass: float | None = None,
    sfreq: float | int = None,
    reference: str | list = 'average',
    pick_channels: str | list = None,
    epoching_kwargs: dict = {},
    dtype: DTypeLike = np.float32,
    preprocessing_fn: Optional[Callable] = None,
    preprocessing_kwargs: dict = {},
    bids_kwargs: dict = {},
    verbose: bool = True,
) -> xr.Dataset:
    """Read BIDS formated EEG/MEG data format using MNE/MNE-BIDS functions.

    Parameters
    ----------
    root: str
        Path to the data in BIDS format with all sub-*** folders
    datatype: str
        Either `eeg` or `meg`
    stimulus_id : dict, optional
        Dictionary mapping stimulus description (keys) to event codes (values).
    response_id : dict, optional
        Dictionary mapping response description (keys) to event codes (values).
    tmin : float, default=-0.2
        Start time (in seconds) relative to stimulus onset for epoching.
    tmax : float, default=5
        End time (in seconds) relative to stimulus onset for epoching.
    high_pass : float, optional
        High-pass filter cutoff frequency.
    low_pass : float, optional
        Low-pass filter cutoff frequency.
    sfreq: float, int
        Desired sampling frequency, can only be lower or equal to the one of the data
    reference: str or list
        Electrodes or method to use for referencing (see mne.set_eeg_reference).
        Average or REST are highly recommended to fit HMP models.
    pick_channels : str or list
        Provide a list of channel names to retain or use all (None, default)
    epoching_kargs: dict
        Dict of named arguments to be passed to the mne.Epoch class.
        Relevant keys:
            proj : bool, optional
                Whether to apply projection. Default is False.
            baseline : tuple or None, optional
                Baseline correction interval. Default is (None, 0).
            detrend : int or None, optional
                Detrending parameter. Default is None.
            on_missing : {'warn', 'ignore', 'raise'}, optional
                Behavior when events are missing. Default is 'warn'.
            event_repeated : {'drop', 'merge'}, optional
                How to handle repeated events. Default is 'drop'.
            reject_by_annotation : bool, optional
                Whether to reject epochs based on annotations. Default is False.
    dtype: np.DTypeLike
        Precision, use np.float32 or np.int64
    preprocessing_fn: callable, optional
        A user defined function preprocessing the raw data before epoching.
    preprocessing_kwargs: dict
        arguments to be passed to the preprocessing_fn function
    bids_kwargs : dict
        Keyword arguments passed to ``mne_bids.find_matching_paths`` to
        locate BIDS files in the dataset. The dictionary may contain any
        of the entity filters supported by the BIDS specification.
        Relevant keys:
            subjects : str | list of str | None
                BIDS subject(s) to match (``sub`` entity).
            sessions : str | list of str | None
                BIDS session(s) to match (``ses`` entity).
            tasks : str | list of str | None
                Task label(s) to match (``task`` entity).
            acquisitions : str | list of str | None
                Acquisition label(s) (``acq`` entity).
            runs : str | list of str | None
                Run number(s) (``run`` entity).
            processings : str | list of str | None
                Processing label(s) (``proc`` entity).
            recordings : str | list of str | None
                Recording label(s) (``recording`` entity).
            datatypes : str | list of str | None
                BIDS datatype(s) to match (e.g., ``'eeg'``, ``'meg'``)
    verbose : bool, default=True
        Whether to display messages.

    Returns
    -------
    epoch_data : xarray.Dataset
        An xarray Dataset containing the processed EEG/MEG data, events, channels, and participants.
        Metadata and epoch indices are preserved.
    """
    if pick_channels is None:
        pick_channels = datatype
    stimulus_id, response_id = format_trigger_description(stimulus_id, response_id)
    epoch_data = []
    subj_names = []

    all_paths = mne_bids.find_matching_paths(
        root=root,
        datatypes=datatype,
        suffixes=datatype,
        ignore_json=True,
        ignore_nosub=True,
        **bids_kwargs
    )
    all_paths = [x for x in all_paths if '.fdt' not in x.fpath.suffix]

    for recording in all_paths:
        subj_name = "_".join(str(recording.basename).split("_")[:-1])
        print(f"Processing dataset {subj_name}")
        
        data = mne_bids.read_raw_bids(
            bids_path = recording,
            verbose=False # Not ideal but too many prints
        )

        # User level preprocessing
        if preprocessing_fn is not None:
            data = preprocessing_fn(data, **preprocessing_kwargs)
            reference = 'user'
        else:
            data = data.pick(pick_channels)
            data.load_data()
            if reference is not None:
                data = data.set_eeg_reference(reference)
        
        # Filtering before epoching
        data, low_pass = _raw_filtering(data, sfreq, low_pass, high_pass, verbose)

        if sfreq is None:
            sfreq = data.info["sfreq"]
        
        events, detected_event_id = mne.events_from_annotations(data)
        # MNE bids extracts triggers from annotations but loses the 
        # original trigger values. The following ensures mapping by matching the 
        # description between the events.tsv and the event_from_annotations
        # and uptating the trigger value in the expected stimulus/response dicts
        new_stim_id, new_resp_id = _bids_to_annot(recording, detected_event_id,
                                                   stimulus_id, response_id, verbose) 
        
        # Epoching + resampling + metadata creation
        epochs, valid_epoch_index = _epoching_raw(data,
                            events,
                            new_stim_id,
                            new_resp_id,
                            tmin,
                            tmax,
                            sfreq,
                            datatype,
                            verbose,
                            epoching_kwargs)
        
        resampling_epochs(epochs, sfreq, verbose)

        if verbose:
            print(f"End sampling frequency is {sfreq} Hz")

        epoch_data.append(hmp_data_format(
                epochs.get_data(copy=False).astype(dtype),
                epochs.info["sfreq"],
                epochs.tmin,
                epochs.tmax,
                epochs=[int(x) for x in valid_epoch_index],
                channel=epochs.ch_names,
                metadata=epochs.metadata,
            )
        )
        subj_names.append(subj_name)

    epoch_data = xr.concat(
        epoch_data,
        dim=xr.DataArray(subj_names, dims="participant"),
        fill_value={"event": "", "data": np.nan},
        join='outer',
    )
    n_trials = (
        (~np.isnan(epoch_data.data[:, :, :, 0].data)).sum(axis=1)[:, 0].sum()
    )  # Compute number of trial based on trial where first sample is nan
    # Use info frm last epoch object, should all be shared
    epoch_data = epoch_data.assign_attrs(
        sfreq=epochs.info['sfreq'],
        lowpass=epochs.info["lowpass"],
        highpass=epochs.info["highpass"],
        reference=reference,
        n_trials=n_trials,
        tmin=epochs.tmin,
        tmax=epochs.tmax,
    )
    return epoch_data


def read_mne_raw(
    all_paths: list,
    datatype: str,
    stimulus_id: dict | None,
    response_id: dict | None,
    tmin: float,
    tmax: float,
    high_pass: float | None = None,
    low_pass: float | None = None,
    sfreq: float | int = None,
    reference: str | list = 'average',
    pick_channels: str | list = None,
    epoching_kwargs: dict = {},
    dtype: DTypeLike = np.float32,
    preprocessing_fn: Optional[Callable] = None,
    preprocessing_kwargs: dict = {},
    verbose: bool = True,
) -> xr.Dataset:
    """Read BIDS formated EEG/MEG data format using MNE/MNE-BIDS functions.

    Parameters
    ----------
    all_paths: str
        Path to the data in BIDS format with all sub-*** folders
    datatype: str
        Either `eeg` or `meg`
    stimulus_id : dict, optional
        Dictionary mapping stimulus description (keys) to event codes (values).
    response_id : dict, optional
        Dictionary mapping response description (keys) to event codes (values).
    tmin : float, default=-0.2
        Start time (in seconds) relative to stimulus onset for epoching.
    tmax : float, default=5
        End time (in seconds) relative to stimulus onset for epoching.
    high_pass : float, optional
        High-pass filter cutoff frequency.
    low_pass : float, optional
        Low-pass filter cutoff frequency.
    sfreq: float, int
        Desired sampling frequency, can only be lower or equal to the one of the data
    reference: str or list
        Electrodes or method to use for referencing (see mne.set_eeg_reference).
        Average or REST are highly recommended to fit HMP models.
    pick_channels : str or list
        Provide a list of channel names to retain or use all (None, default)
    epoching_kargs: dict
        Dict of named arguments to be passed to the mne.Epoch class.
        Relevant keys:
            proj : bool, optional
                Whether to apply projection. Default is False.
            baseline : tuple or None, optional
                Baseline correction interval. Default is (None, 0).
            detrend : int or None, optional
                Detrending parameter. Default is None.
            on_missing : {'warn', 'ignore', 'raise'}, optional
                Behavior when events are missing. Default is 'warn'.
            event_repeated : {'drop', 'merge'}, optional
                How to handle repeated events. Default is 'drop'.
            reject_by_annotation : bool, optional
                Whether to reject epochs based on annotations. Default is False.
    dtype: np.DTypeLike
        Precision, use np.float32 or np.int64
    preprocessing_fn: callable, optional
        A user defined function preprocessing the raw data before epoching.
    preprocessing_kwargs: dict
        arguments to be passed to the preprocessing_fn function
    verbose : bool, default=True
        Whether to display messages.

    Returns
    -------
    epoch_data : xarray.Dataset
        An xarray Dataset containing the processed EEG/MEG data, events, channels, and participants.
        Metadata and epoch indices are preserved.
    """
    if pick_channels is None:
        pick_channels = datatype
    stimulus_id, response_id = format_trigger_description(stimulus_id, response_id)

    epoch_data = []
    subj_names = []

    for recording in all_paths:
        subj_name = str(Path(recording).stem).split(".")[:-1]
        print(f"Processing dataset {subj_name}")
        
        data = mne.io.read_raw_fif(participant, preload=True, verbose=verbose)

        # User level preprocessing
        if preprocessing_fn is not None:
            data = preprocessing_fn(data, **preprocessing_kwargs)
            reference = 'user'
        else:
            data = data.pick(pick_channels)
            data.load_data()
            if reference is not None:
                data = data.set_eeg_reference(reference)
        
        # Filtering before epoching
        data, low_pass = _raw_filtering(data, sfreq, low_pass, high_pass, verbose)

        if sfreq is None:
            sfreq = data.info["sfreq"]
        
        events, detected_event_id = mne.events_from_annotations(data)
        # MNE bids extracts triggers from annotations but loses the 
        # original trigger values. The following ensures mapping by matching the 
        # description between the events.tsv and the event_from_annotations
        # and uptating the trigger value in the expected stimulus/response dicts
        new_stim_id, new_resp_id = _bids_to_annot(recording, detected_event_id,
                                                   stimulus_id, response_id, verbose) 
        
        # Epoching + resampling + metadata creation
        epochs, valid_epoch_index = _epoching_raw(data,
                            events,
                            new_stim_id,
                            new_resp_id,
                            tmin,
                            tmax,
                            sfreq,
                            datatype,
                            verbose,
                            epoching_kwargs)
        
        resampling_epochs(epochs, sfreq, verbose)

        if verbose:
            print(f"End sampling frequency is {sfreq} Hz")

        epoch_data.append(hmp_data_format(
                epochs.get_data(copy=False).astype(dtype),
                epochs.info["sfreq"],
                epochs.tmin,
                epochs.tmax,
                epochs=[int(x) for x in valid_epoch_index],
                channel=epochs.ch_names,
                metadata=epochs.metadata,
            )
        )
        subj_names.append(subj_name)

    epoch_data = xr.concat(
        epoch_data,
        dim=xr.DataArray(subj_names, dims="participant"),
        fill_value={"event": "", "data": np.nan},
        join='outer',
    )
    n_trials = (
        (~np.isnan(epoch_data.data[:, :, :, 0].data)).sum(axis=1)[:, 0].sum()
    )  # Compute number of trial based on trial where first sample is nan
    # Use info frm last epoch object, should all be shared
    epoch_data = epoch_data.assign_attrs(
        sfreq=epochs.info['sfreq'],
        lowpass=epochs.info["lowpass"],
        highpass=epochs.info["highpass"],
        reference=reference,
        n_trials=n_trials,
        tmin=epochs.tmin,
        tmax=epochs.tmax,
    )
    return epoch_data

def read_mne_epochs(
    participant,
    sfreq,
    high_pass,
    low_pass,
    pick_channels,
    tmin,
    tmax,
    verbose
):

    if Path(participant).suffix == ".fif":
        epochs = mne.read_epochs(participant, preload=True, verbose=verbose)
    else:
        raise ValueError("Incorrect file format")

    # Filtering
    if high_pass is not None or low_pass is not None:
        epochs.filter(high_pass, low_pass, fir_design="firwin", verbose=verbose)

    # Resampling
    if sfreq is None:
        sfreq = epochs.info["sfreq"]
    elif sfreq < epochs.info["sfreq"]:
        if verbose:
            print(f"Resampling data at {sfreq}")
        epochs = epochs.resample(sfreq)

    # Cropping
    if tmin > epochs.tmin:
        epochs.crop(tmin=tmin)
        if verbose:
            print(f"Cropping epochs to {tmin}s before centering events")
    else:
        tmin = epochs.tmin
    if tmax < epochs.tmax:
        epochs.crop(tmax=tmax)
        if verbose:
            print(f"Cropping epochs to {tmax}s after centering events")
    else:
        tmax = epochs.tmax
    epochs = epochs.pick(pick_channels)
    return epochs, tmin, tmax

def make_defaults_epoching(kwargs):
    kwargs.setdefault("proj", False)
    kwargs.setdefault("baseline", (None, 0))
    kwargs.setdefault("detrend", None)
    kwargs.setdefault("on_missing", "warn")
    kwargs.setdefault("event_repeated", "drop")
    kwargs.setdefault("reject_by_annotation", False)
    return kwargs

def format_trigger_description(stimulus_id, response_id):
    if any(not k.startswith("stimulus/") for k in stimulus_id.keys()):
        stimulus_id = {f"stimulus/{k}": v for k, v in stimulus_id.items()}    
    if any(not k.startswith("response/") for k in response_id.keys()):
        response_id = {f"response/{k}": v for k, v in response_id.items()}
    return stimulus_id, response_id

def _bids_to_annot(path, detected_event_id, stimulus_id, response_id, verbose):
    path_to_tsv = path.copy().update(suffix="events", extension=".tsv")
    events_dict = mne_bids.events_file_to_annotation_kwargs(path_to_tsv)
    read_event_id = events_dict['event_id']
    # Remap based on user requested dict
    if verbose:
        print(f'Found events {np.sort(list(read_event_id.values()))} in '
              f'{path_to_tsv}, \n mapping to the declared '
              f'triggers: {np.sort(list((stimulus_id | response_id).values()))}')
    old_stim_id = {v:k for k,v in stimulus_id.items()} 
    old_resp_id = {v:k for k,v in response_id.items()}
    new_stim_id = {}
    new_resp_id = {}
    for k,v in read_event_id.items():
        new_v = detected_event_id[k]
        if v in stimulus_id.values():
            new_stim_id[old_stim_id[v]] = int(new_v)
        elif v in response_id.values():
            new_resp_id[old_resp_id[v]] = int(new_v)
    return new_stim_id, new_resp_id
    
def _epoching_raw(
        data,
        events,
        stimulus_id,
        response_id,
        tmin,
        tmax,
        sfreq,
        pick_channels,
        verbose,
        epoching_kwargs
    ):

    if "tmin" in epoching_kwargs:
        warnings.warn("Ignoring tmin in epoching_kwargs; use the function argument instead.")
    if "tmax" in epoching_kwargs:
        warnings.warn("Ignoring tmax in epoching_kwargs; use the function argument instead.")
    epoching_kwargs.pop("tmin", None)
    epoching_kwargs.pop("tmax", None)

    event_id = {**stimulus_id, **response_id}
    stim = list(stimulus_id.keys())
    
    metadata_i, meta_events, stimulus_id = mne.epochs.make_metadata(
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        sfreq=data.info["sfreq"],
        row_events=stim,
        keep_first=["response"],
    )
    cols = ["event_name", "response"]
    if 'first_response' in metadata_i.columns:
        cols.append('first_response')
    metadata_i = metadata_i[cols]  # only keep event_names and rts

    epochs = mne.Epochs(
        data,
        meta_events,
        stimulus_id,
        tmin,
        tmax,
        preload=True,
        picks=pick_channels,
        metadata=metadata_i,
        **epoching_kwargs
    )
    epochs.metadata.rename({"response": "rt", "first_response":"response"}, axis=1, inplace=True)
    
    valid_epoch_index = [x for x, y in enumerate(epochs.drop_log) if len(y) == 0]
    return epochs, valid_epoch_index

def hmp_data_format(
    data: np.ndarray,
    sfreq: float,
    tmin: float,
    tmax: float,
    events: np.ndarray | None = None,
    participants: list | None = None,
    epochs: list | None = None,
    channel: list | None = None,
    metadata: DataFrame | None = None,
) -> xr.Dataset:

    """
    Convert data to the expected xarray Dataset format.

    This function reshapes a 3D or 4D matrix with dimensions
    (participant) * trial * channel * sample into an xarray Dataset.

    Parameters
    ----------
    data : np.ndarray
        3D matrix with dimensions trial X channel X sample.
    sfreq : float
        Sampling frequency of the data.
    participants : list, optional
        List of participant indices if multiple ones are processed
    epochs : list, optional
        List of epoch indices.
    channel : list, optional
        List of channel indices.
    metadata : DataFrame, optional
        Metadata associated with the epochs. Should be a pandas DataFrame.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the reshaped data, with appropriate dimensions and attributes.
    """

    n_epochs, n_channels, n_samples = np.shape(data)

    if channel is None:
        channel = np.arange(n_channels)
    if epochs is None:
        epochs = np.arange(n_epochs)
    data = xr.Dataset(
        {
            "data": (["epoch", "channel", "sample"], data),
        },
        coords={"epoch": epochs, "channel": channel,
                "sample": np.linspace(np.rint(tmin*sfreq),
                    np.rint(tmax*sfreq), n_samples, dtype=int)},
    )
    
    if metadata is not None:
        metadata = metadata.loc[epochs]
        metadata = metadata.to_xarray()
        metadata = metadata.rename_dims({"index": "epoch"})
        metadata = metadata.rename_vars({"index": "epoch"})
        data = data.merge(metadata)
        data = data.set_coords(list(metadata.data_vars))

    return data


def save_eventprobs_csv(estimates, filename):
    """
    Save event probability estimates to a CSV file.

    Parameters
    ----------
    estimates : xarray.DataArray or xarray.Dataset
        The event probability estimates to save.
    filename : str
        The path to the CSV file where the estimates will be saved.
    """
    estimates = estimates.unstack()
    estimates.to_dataframe('eventprobs').to_csv(filename)
    print(f"Saved at {filename}")


def _raw_filtering(data, sfreq, low_pass, high_pass, verbose):
    if sfreq < data.info["sfreq"]:  # Downsampling
        decim = np.round(data.info["sfreq"] / sfreq).astype(int)
        obtained_sfreq = data.info["sfreq"] / decim
        if low_pass is None:
            low_pass = obtained_sfreq / 3.1
    else:
        decim = 1
        if sfreq > data.info["sfreq"] + 1:
            warnings.warn(
                f"Requested higher frequency {sfreq} than found in the EEG data, no "
                f"resampling is performed"
            )
    if high_pass is not None or low_pass is not None:
        data.filter(high_pass, low_pass, fir_design="firwin", verbose=verbose)

    return data, low_pass

def _extract_events(data, events, stimulus_id, response_id, verbose):
    if events is None:
        try:
            events = mne.find_events(
                data, verbose=verbose, min_duration=1 / data.info["sfreq"]
            )
        except ValueError:
            events, event_id = mne.events_from_annotations(data, verbose=verbose)[0]
        if (
            events[0, 1] > 0
        ):  # bug from some stim channel, should be 0 otherwise indicates offset in triggers
            print(
                f"Correcting event values as trigger channel has offset "
                f"{np.unique(events[:, 1])}"
            )
            events[:, 2] = events[:, 2] - events[:, 1]  # correction on event value
    events_values = np.concatenate(
        [
            np.array([x for x in stimulus_id.values()]),
            np.array([x for x in response_id.values()]),
        ]
    )
    events = np.array(
        [list(x) for x in events if x[2] in events_values]
    )  # only keeps events with stim or response
    event_id = {**stimulus_id, **response_id}  # stimulus_id | response_id
    stim = list(stimulus_id.keys())

    return events, event_id, stim

def resampling_epochs(epochs, sfreq, verbose):
    if sfreq < epochs.info["sfreq"]:
        if verbose:
            print(f"Resampling data at {sfreq}")
        epochs = epochs.resample(sfreq)
    else:
        raise ValueError('Requested sampling frequency is higher than the sampling frequency of the data')
    return epochs