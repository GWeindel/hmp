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
import warnings
from pathlib import Path
from typing import Callable, Optional

import mne
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from pandas import DataFrame


def read_mne_data( # noqa: PLR0913,PLR0912
    pfiles: str | list = [],
    event_id: dict | None = None,
    resp_id: dict | None = None,
    data_format: str = 'raw',
    sfreq: float | None = None,
    subj_name: list | None = None,
    metadata: list | None = None,
    events_provided: np.ndarray | None = None,
    verbose: bool = True,
    tmin: float = -0.2,
    tmax: float = 5,
    high_pass: float | None = None,
    low_pass: float | None = None,
    pick_channels: str | list = "eeg",
    reference: str | None = None,
    bids_parameters: dict = {},
    preprocessing_fn: Optional[Callable] = None,
    dtype: DTypeLike = np.float32,
) -> xr.Dataset:
    """Read EEG/MEG data format (.fif or .bdf) using MNE's integrated function.

    Notes
    -----
    - Only EEG or MEG data are selected (other channel types are discarded).
    - All times are expressed in seconds.
    - If multiple files are provided in ``pfiles``, each participant's data is read and processed
      sequentially.
    - For non-epoched data: Reaction Times are only computed if the response trigger is in the
      epoching window (determined by ``tmin`` and ``tmax``).

    ## Procedure:

    If data is not already epoched:

        - The data is filtered using the specified ``low_pass`` and ``high_pass`` parameters.
        - If no events are provided, events are detected in the stimulus channel and only those with
          IDs in ``event_id`` and ``resp_id`` are kept.
        - Downsampling is performed if ``sfreq`` is lower than the data's sampling frequency.
        - Epochs are created based on stimulus onsets (``event_id``)
          and the ``tmin``/``tmax`` window.
          Epochs with 'BAD' annotations are removed. Baseline correction is applied from
          ``tmin`` to stimulus onset (time 0).

    Parameters
    ----------
    pfiles : str or list of str
        Path(s) to EEG files to read. Can be a single file path or a list of file paths.
        If empty list, assumes bids format
    event_id : dict, optional
        Dictionary mapping condition names (keys) to event codes (values).
    resp_id : dict, optional
        Dictionary mapping response names (keys) to event codes (values).
    data_format : str, default=epochs
        What MNE compatible data type, can be 'epochs', 'raw' or 'bids'.
    sfreq : float, optional
        Desired sampling frequency for downsampling.
    subj_name : list of str, optional
        List of subject identifiers. If not provided, defaults to "S0", "S1", etc.
    metadata : list of pandas.DataFrame, optional
        List of metadata DataFrames corresponding to each participant.
    events_provided : np.ndarray, optional
        Array with 3 columns: [sample of the event, initial value of the channel, event code].
        Used if automated event detection is not suitable.
    verbose : bool, default=True
        Whether to display MNE's messages.
    tmin : float, default=-0.2
        Start time (in seconds) relative to stimulus onset for epoching.
    tmax : float, default=5
        End time (in seconds) relative to stimulus onset for epoching.
    high_pass : float, optional
        High-pass filter cutoff frequency.
    low_pass : float, optional
        Low-pass filter cutoff frequency.
    pick_channels : str or list, default="eeg"
        Channels to retain. Use "eeg"/"meg" to keep only EEG/MEG channels
        or provide a list of channel names.
    reference : str, optional
        Reference to use for EEG data. If None, the existing reference is kept.
    bids_parameters: dict, optional
        Bids root path ('root'), datatype ('datatype") to analyze.
        A filter can also be applied to subjects, tasks and sessions
    preprocessing_fn: callable, optional
        A user defined function preprocessing the raw data before epoching
    dtype: np.DTypeLike
        Precision, use np.float32 or np.int64

    Returns
    -------
    epoch_data : xarray.Dataset
        An xarray Dataset containing the processed EEG/MEG data, events, channels, and participants.
        Metadata and epoch indices are preserved. The chosen sampling frequency
        is stored as an attribute.
    """
    epoch_data = []
    if isinstance(pfiles, (str, Path)):  # only one participant
        pfiles = [pfiles]
    if not subj_name:
        subj_name = ["sub-" + str(x) for x in np.arange(len(pfiles))]
    if isinstance(subj_name, str):
        subj_name = [subj_name]
    subj_idx = 0
    if metadata is not None:
        if len(pfiles) > 1 and len(metadata) != len(pfiles):
            raise ValueError(
                f"Incompatible dimension between the provided metadata {len(metadata)} and the "
                f"number of eeg files provided {len(pfiles)}"
            )

    if len(bids_parameters.keys())>0:
        data_format = 'bids'


    if data_format == 'bids':
        subj_name = pfiles = [
            d for d in os.listdir(bids_parameters['bids_root'])
            if d.startswith("sub-") and os.path.isdir(os.path.join(bids_parameters['bids_root'], d))
        ]
        event_id, resp_id = _bids_extract_trig(
            bids_parameters['bids_root'],
            bids_parameters['task'],
        )

    for participant in pfiles:
        print(f"Processing participant {participant}'s {data_format} {pick_channels}")
        if data_format == 'epochs':
            epochs, tmin, tmax = _read_mne_epochs(participant,
                    sfreq,
                    high_pass,
                    low_pass,
                    pick_channels,
                    tmin,
                    tmax,
                    verbose)

        elif data_format in ['raw', "bids"]:
            epochs = read_raw_and_epoch(participant,
                            subj_idx,
                            event_id,
                            resp_id,
                            sfreq,
                            metadata,
                            events_provided,
                            verbose,
                            tmin,
                            tmax,
                            high_pass,
                            low_pass,
                            pick_channels,
                            bids_parameters,
                            preprocessing_fn)
        else:
            raise ValueError(f"Unknown data type {data_format}, should be 'epochs', 'raw' or "
                             "'bids'")

        if reference is not None:
            epochs = epochs.set_eeg_reference(reference)

        if metadata is None:
            try:
                metadata_i = epochs.metadata  # accounts for dropped epochs
            except AttributeError:
                warnings.warn(f'No metadata found for {subj_name[subj_idx]}')
        elif isinstance(metadata, DataFrame):# TODO handle multiple Dataframes
            metadata_i = metadata.copy()
        else:
            raise ValueError(
                "Metadata should be a pandas data-frame as generated by mne or be contained "
                "in the passed epoch data"
            )
        sfreq = epochs.info["sfreq"] if sfreq is None else sfreq
        valid_epoch_index = [x for x, y in enumerate(epochs.drop_log) if len(y) == 0]

        if verbose:
            print(f"End sampling frequency is {sfreq} Hz")

        epoch_data.append(hmp_data_format(
                epochs.get_data(copy=False).astype(dtype),
                epochs.info["sfreq"],
                epochs.tmin,
                epochs.tmax,
                None,
                epochs=[int(x) for x in valid_epoch_index],
                channel=epochs.ch_names,
                metadata=metadata_i,
            )
        )

        subj_idx += 1

    epoch_data = xr.concat(
        epoch_data,
        dim=xr.DataArray(subj_name, dims="participant"),
        fill_value={"event": "", "data": np.nan},
        join='outer',
    )
    n_trials = (
        (~np.isnan(epoch_data.data[:, :, :, 0].data)).sum(axis=1)[:, 0].sum()
    )  # Compute number of trial based on trial where first sample is nan
    epoch_data = epoch_data.assign_attrs(
        lowpass=epochs.info["lowpass"],
        highpass=epochs.info["highpass"],
        reference=reference,
        n_trials=n_trials,
        tmin=tmin,
        tmax=tmax,
    )
    return epoch_data

def _bids_extract_trig(bids_root, task):

    # Recover the general information on task triggers
    # Path to the events.json file
    events_json_path = os.path.join(bids_root, f"task-{task}_events.json")

    with open(events_json_path, "r") as f:
        events_json = json.load(f)

    # Build stim_id dictionary: {'stimulus/description': event_code}
    stim_id, resp_id = {}, {}

    # Extract stimulus_id and resp_id
    event_code_levels = events_json['value']['Levels']
    for code, desc in event_code_levels.items():
        if 'Stimulus' in desc:
            stim_id[f'stimulus/{desc[11:]}'] = int(code)
        if 'Response' in desc:
            resp_id[f'response/{desc[11:]}'] = int(code)
    return stim_id, resp_id

def _bids_extract_events(raw, verbose):
    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    # The two next lines avoid confusion for triggers < 1000
    event_id = {k:v*1000 for k,v in event_id.items()}
    events[:,2] *= 1000
    # Replace event codes in events array with the integer at the end of each key in *_id
    for key in event_id:
        try:
            code = int(key.split('/')[-1])
            events[:, 2][events[:, 2] == event_id[key]] = code
        except Exception as e:
            print(f"Could not process key {key}: {e}")

    return events

def _read_mne_epochs(
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


def read_raw_and_epoch(  # noqa # Should probably be refactored.
    participant,
    subj_idx,
    event_id,
    resp_id,
    sfreq,
    metadata,
    events_provided,
    verbose,
    tmin,
    tmax,
    high_pass,
    low_pass,
    pick_channels,
    bids_parameters,
    preprocessing_fn
):
    if Path(participant).suffix == ".fif":
        data = mne.io.read_raw_fif(participant, preload=True, verbose=verbose)
    elif Path(participant).suffix == ".bdf":
        data = mne.io.read_raw_bdf(participant, preload=True, verbose=verbose)
    elif isinstance(bids_parameters, dict) and len(bids_parameters) > 0:
        import mne_bids  # noqa: PLC0415
        bids_path = mne_bids.BIDSPath(subject=participant.replace("sub-", ""),
                                      task=bids_parameters['task'],
                                      root=bids_parameters['bids_root'],
                                      session = bids_parameters['session'],
                                      datatype=bids_parameters['datatype'])

        data = mne_bids.read_raw_bids(
            bids_path = bids_path,
            verbose=False
        )
        events_provided = _bids_extract_events(data, verbose)
    else:
        raise ValueError(f"Unknown EEG file format for participant {participant}, only '.bdf' and "
                         "'.fif' or BIDS are accepted")
    if sfreq is None:
        sfreq = data.info["sfreq"]

    if "response" not in list(resp_id.keys())[0]:
        resp_id = {f"response/{k}": v for k, v in resp_id.items()}
    if events_provided is None:
        try:
            events = mne.find_events(
                data, verbose=verbose, min_duration=1 / data.info["sfreq"]
            )
        except ValueError:
            events = mne.events_from_annotations(data, verbose=verbose)[0]
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
                np.array([x for x in event_id.values()]),
                np.array([x for x in resp_id.values()]),
            ]
        )
        events = np.array(
            [list(x) for x in events if x[2] in events_values]
        )  # only keeps events with stim or response

    if len(np.shape(events_provided))>2:  # assumes stacked event files
        events = events_provided[subj_idx]
    else:
        events = events_provided
    data = data.pick(pick_channels)
    data.load_data()

    if sfreq < data.info["sfreq"]:  # Downsampling
        print(f"Downsampling to {sfreq} Hz")
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
    combined = {**event_id, **resp_id}  # event_id | resp_id
    stim = list(event_id.keys())

    if verbose:
        print(f"Creating epochs based on following event ID :{np.unique(events[:, 2])}")

    if preprocessing_fn is not None:
        data = preprocessing_fn(data)

    if metadata is None:
        metadata_i, meta_events, event_id = mne.epochs.make_metadata(
            events=events,
            event_id=combined,
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
    else:
        metadata_i = metadata[subj_idx]
    epochs = mne.Epochs(
        data,
        meta_events,
        event_id,
        tmin,
        tmax,
        proj=False,
        baseline=(None, 0),
        preload=True,
        picks=pick_channels,
        decim=decim,
        verbose=verbose,
        detrend=None,
        on_missing="warn",
        event_repeated="drop",
        metadata=metadata_i,
        reject_by_annotation=True,
    )
    epochs.metadata.rename({"response": "rt", "first_response":"response"}, axis=1, inplace=True)
    return epochs


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
        4D or 3D matrix with dimensions (participant) * trial * channel * sample.
    sfreq : float
        Sampling frequency of the data.
    events : np.ndarray, optional
        Description for each epoch and participant that need to be stored (e.g. condition)
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
    if len(np.shape(data)) == 4:  # means group
        n_subj, n_epochs, n_channels, n_samples = np.shape(data)
    elif len(np.shape(data)) == 3:
        n_epochs, n_channels, n_samples = np.shape(data)
        n_subj = 1
    else:
        raise ValueError(f"Unknown data format with dimensions {np.shape(data)}")
    if channel is None:
        channel = np.arange(n_channels)
    if epochs is None:
        epochs = np.arange(n_epochs)
    if n_subj == 1:
        data = xr.Dataset(
            {
                "data": (["epoch", "channel", "sample"], data),
            },
            coords={"epoch": epochs, "channel": channel,
                    "sample": np.linspace(np.rint(tmin*sfreq),
                        np.rint(tmax*sfreq), n_samples, dtype=int)},
            attrs={"sfreq": sfreq},
        )
    else:
        data = xr.Dataset(
            {
                "data": (["participant", "epoch", "channel", "sample"], data),
            },
            coords={
                "participant": participants,
                "epoch": epochs,
                "channel": channel,
                "sample": np.linspace(int(np.rint(tmin*sfreq)),
                                      int(np.rint(tmax*sfreq)), n_samples),
            },
            attrs={"sfreq": sfreq},
        )
    if metadata is not None:
        metadata = metadata.loc[epochs]
        metadata = metadata.to_xarray()
        metadata = metadata.rename_dims({"index": "epoch"})
        metadata = metadata.rename_vars({"index": "epoch"})
        data = data.merge(metadata)
        data = data.set_coords(list(metadata.data_vars))
    if events is not None:
        data["events"] = xr.DataArray(
            events,
            dims=("participant", "epoch"),
            coords={"participant": participants, "epoch": epochs},
        )
        data = data.set_coords("events")
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
