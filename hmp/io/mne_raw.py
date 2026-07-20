"""MNE raw data format reading.

This module provides functions for reading MNE raw data format
"""

import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from mne import events_from_annotations, find_events, set_log_level
from mne.channels import DigMontage
from mne.io import read_raw_bdf, read_raw_fif
from numpy.typing import DTypeLike
from xarray import Dataset

from hmp.io import preprocessing, utils


def read_mne_raw(# noqa: PLR0913
    recordings: list,
    centering_id: dict,
    event_id: dict = {},
    events_provided: np.ndarray | None = None,
    subj_name: list = None,
    montage: str | DigMontage | None = None,
    epoching_kwargs: dict = {},
    preprocessing_kwargs: dict = {},
    dtype: DTypeLike = np.float32,
    preprocessing_fn: Optional[Callable] = None,
    verbose: bool | str = True,
    cpus: int = 1,
) -> Dataset:
    """Read .bdf or .fif continuous recordings using MNE/MNE-BIDS functions.

    Parameters
    ----------
    recordings: list
        List of the paths to the recordings to load
    centering_id : dict
        Dictionary mapping stimulus description (keys) to event codes (values).
    event_id : dict
        Dictionary mapping non-centering events description (keys) to event codes (values).
    events_provided : list
        list of np.ndarray, one per recording. Each np.array has:
            1 row per event and
            3 columns: [sample of the event, initial value of the channel, event code].
        Used if automated event detection is not suitable.
        Only works if a single recording is provided
    subj_name : list
        List of subject names
    montage: str or mne.channels.DigMontage
        Either an MNE DigMontage or a string for a bulit-in MNE montage (see
        mne.channels.get_builtin_montages()) that is applied to all recordings.
    preprocessing_kwargs: dict
        arguments to be passed to the preprocessing functions. If no
        'preprocessing_fn' is specified, only the following keys are relevant:
            highpass : float
                high pass filter provided to MNE's filtering function
            lowpass : float
                lowpass filter provided to MNE's filtering function
            sfreq: float
                Desired sampling frequency, can only be lower or equal to the one of the data.
                The downsampling is performed on the raw data which can result in time jitter
                in the event triggers. This is minimzed in HMP by providing the events to the
                resampling function. Users who prefer to perform that at the epoch level can use
                the 'decim' argument in epoching_kargs
            reference: str
                Electrodes or method to use for referencing (see mne.set_eeg_reference).
                Average (common reference) or REST are highly recommended to fit HMP models.
            pick_channels: list of str
                Channels to use, can be list of channel names or 'eeg'/'meg'
    epoching_kargs: dict
        Dict of named arguments to be passed to the mne.Epoch class.
        Relevant keys:
            proj : bool
                Whether to apply projection. Default is False.
            baseline : tuple or None
                Baseline correction interval. Default is (None, 0).
            detrend : int or None
                Detrending parameter. Default is None.
            on_missing : {'warn', 'ignore', 'raise'}
                Behavior when events are missing. Default is 'warn'.
            event_repeated : {'drop', 'merge'}
                How to handle repeated events. Default is 'drop'.
            reject_by_annotation : bool
                Whether to reject epochs based on annotations. Default is False.
            decim : int
                Whether to downsample the epochs through decimation.
    dtype: np.DTypeLike
        Precision, use np.float32 or np.int64
    preprocessing_fn: callable
        A user defined function preprocessing the raw data before epoching.
    cpus : int
        How many cpus to use. If > 1 process several datasets in parallel
    verbose : bool | str, default=True
        Whether to display messages. also supports MNE logging syntax:
        DEBUG, INFO, WARNING, ERROR, or CRITICAL

    Returns
    -------
    epoch_data : xarray.Dataset
        An xarray Dataset containing the processed EEG/MEG data, events, channels, and participants.
        Metadata and epoch indices are preserved.
    info: mne.Info
        Mock info object containing channel positions for plotting with HMP functions
    """
    set_log_level(verbose)
    # Epoching defaults and check
    epoching_kwargs = utils._defaults_check_epoching(epoching_kwargs)

    # Same for preprocessing
    if preprocessing_fn is None:
        preprocessing_kwargs = utils._defaults_check_prep(preprocessing_kwargs)

    # Trigger definition and check
    centering_id, event_id = utils._format_trigger_description(centering_id, event_id)
    [utils._check_trigger_dicts(x) for x in [centering_id, event_id]]

    # Checking montage
    utils._check_montage(montage)

    if not isinstance(recordings, list):
        raise ValueError("Expected a list of paths to the recordings."
                        f"Got an object of type {type(recordings)} instead")
    if events_provided is None:
        events_provided = [None for x in recordings]
    elif not isinstance(events_provided, list):
        raise ValueError("Expected a list of events for each recording."
                        f"Got an object of type {type(events_provided)} instead")
    recordings = [Path(x) for x in recordings]
    if subj_name is None:
        subj_name = ["_".join(str(i.name).split("_")[:-1]) for i in recordings]
    # Processing loops/parallel
    if cpus == 1:
        epochs_list = [_process_raw_dataset(
            recording, montage, centering_id, event_id, event,
            verbose, preprocessing_fn, preprocessing_kwargs, epoching_kwargs
            )
            for recording, event in zip(recordings, events_provided)
        ]
    else:
        with mp.Pool(processes=cpus) as pool:
            epochs_list = pool.starmap(
                _process_raw_dataset,
                [(recording, montage, centering_id, event_id, event,
            verbose, preprocessing_fn, preprocessing_kwargs, epoching_kwargs)
                    for recording, event in zip(recordings, events_provided)
                ],
            )

    epoch_data = [
        utils.hmp_data_format(
            epochs.get_data(copy=False).astype(dtype),
            epochs.info['sfreq'],
            epochs.tmin,
            epochs.tmax,
            epochs=[int(x) for x in valid_epoch_index],
            channel=epochs.ch_names,
            metadata=epochs.metadata,
        )
        for epochs, valid_epoch_index in epochs_list
    ]

    # Recover info from first epochs object
    info = epochs_list[0][0].info
    final_prep_kwargs = deepcopy(preprocessing_kwargs)
    final_prep_kwargs['sfreq'] = epochs_list[0][0].info['sfreq']
    final_prep_kwargs['lowpass'] = epochs_list[0][0].info['lowpass']
    final_prep_kwargs['highpass'] = epochs_list[0][0].info['highpass']
    epoch_data = utils._concat_recordings(epoch_data, recordings,
                    epoching_kwargs, final_prep_kwargs, subj_name)

    return epoch_data, info

def _process_raw_dataset(recording, montage, centering_id, event_id, events_provided,
        verbose, preprocessing_fn, preprocessing_kwargs, epoching_kwargs):
    if verbose is True or verbose in ["DEBUG","INFO"]:
        print(f"Processing dataset {'_'.join(str(recording.name).split('_')[:-1])}")
    if recording.suffix == ".fif":
        data = read_raw_fif(recording, verbose=verbose)
    elif recording.suffix == ".bdf":
        data = read_raw_bdf(recording, verbose=verbose)
    else:
        raise NotImplementedError(f"Extension {recording.suffix}"
        "is not supported yet. if needed, open an "
        "issue on https://github.com/GWeindel/hmp/issues")

    events = _extract_mne_events(data, events_provided,
                                        centering_id, event_id, verbose)

    # User level preprocessing, should include: re-referencing, channel selection
    # filtering and resampling if needed and take data, events, preprocessing_kwargs
    # as input and output data and events, see example in utils.preprocess_raw
    if preprocessing_fn is not None:
        data, events = preprocessing_fn(data, montage, events,
                   verbose, preprocessing_kwargs)
    else:
        data, events = preprocessing.preprocess_data(data, montage, events,
                   verbose, preprocessing_kwargs)

    # Epoching + resampling + metadata creation
    epochs, valid_epoch_index = utils._epoching_raw(
        data, events, centering_id, event_id, verbose, epoching_kwargs)

    return epochs, valid_epoch_index

def _extract_mne_events(data, events, centering_id, event_id, verbose):
    if events is None:
        try:
            events = find_events(
                data, verbose=verbose, min_duration=1 / data.info["sfreq"]
            )
        except ValueError:
            events, event_id = events_from_annotations(data, verbose=verbose)
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
            np.array([x for x in centering_id.values()]),
            np.array([x for x in event_id.values()]),
        ]
    )
    events = np.array(
        [list(x) for x in events if x[2] in events_values]
    )  # only keeps declared events

    return events
