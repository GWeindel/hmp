"""EEG/MEG BIDS data format reading.

This module provides functions for reading BIDS data format:
1) Organize 
"""

import json
import os
import re
from pathlib import Path
from warnings import warn
from typing import Callable, Optional
import multiprocessing as mp

import mne_bids
import numpy as np
from xarray import Dataset
from numpy.typing import DTypeLike
import hmp.io.utils as utils
from mne import events_from_annotations
from mne.channels import DigMontage

def read_bids_raw(
    bids_kwargs: dict,
    epoching_kwargs: dict,
    preprocessing_kwargs: dict,
    centering_id: dict,
    response_id: dict = {},
    montage: str | DigMontage = '',
    dtype: DTypeLike = np.float32,
    preprocessing_fn: Optional[Callable] = None,
    verbose: bool = True,
    cpus: int = 1,
) -> Dataset:
    """Read BIDS formated EEG/MEG data format using MNE/MNE-BIDS functions.

    Parameters
    ----------
    bids_kwargs : dict
        Keyword arguments passed to ``mne_bids.find_matching_paths`` to
        locate BIDS files in the dataset. The dictionary may contain any
        of the entity filters supported by the BIDS specification.
        Relevant keys:
            root : str
                Mandatory key fot the BIDS dataset root path
            datatype : list of str | None
                BIDS datatype to match (e.g., ``'eeg'``, ``'meg'``)
            subjects : list of str | None
                BIDS subject(s) to match (``sub`` entity).
            sessions : list of str | None
                BIDS session(s) to match (``ses`` entity).
            tasks : list of str | None
                Task label(s) to match (``task`` entity).
            acquisitions : list of str | None
                Acquisition label(s) (``acq`` entity).
            runs : list of str | None
                Run number(s) (``run`` entity).
            processings : list of str | None
                Processing label(s) (``proc`` entity).
            recordings : list of str | None
                Recording label(s) (``recording`` entity).
    preprocessing_kwargs: dict
        arguments to be passed to the preprocessing functions. If no
        'preprocessing_fn' is specified, only the following keys are relevant:
            high_pass : float
                high pass filter provided to MNE's filtering function
            low_pass : float
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
            decim : int, optional
                Whether to downsample the epochs through decimation.
    centering_id : dict, optional
        Dictionary mapping stimulus description (keys) to event codes (values).
    response_id : dict, optional
        Dictionary mapping response description (keys) to event codes (values).
    montage: str or mne.channels.DigMontage
        Either an MNE DigMontage or a string for a bulit-in MNE montage (see 
        mne.channels.get_builtin_montages()) that is applied to all recordings.
    dtype: np.DTypeLike
        Precision, use np.float32 or np.int64
    preprocessing_fn: callable, optional
        A user defined function preprocessing the raw data before epoching.
    cpus : int
        How many cpus to use. If > 1 process several datasets in parallel
    verbose : bool, default=True
        Whether to display messages.

    Returns
    -------
    epoch_data : xarray.Dataset
        An xarray Dataset containing the processed EEG/MEG data, events, channels, and participants.
        Metadata and epoch indices are preserved.
    """
    # Dict integrity check
    check_bids_kwargs(bids_kwargs)

    # Epoching defaults and check
    epoching_kwargs = utils._defaults_check_epoching(epoching_kwargs)

    # Same for preprocessing
    preprocessing_kwargs = utils._defaults_check_prep(preprocessing_kwargs, preprocessing_fn)

    # Trigger definition and check
    centering_id, response_id = utils._format_trigger_description(centering_id, response_id)

    # Checking montage
    if montage is None:
        warn("No montage was provided, HMP plotting functions cannot be used without a "
            "valid template montage. If using standard channel montage declare one "
            "of MNE's built-in montage (see mne.channels.get_builtin_montages()) in "
            "the 'montage' argument, alternatively provide an mne.DigMontage object")
    
    # List all paths but exclude .fdt if old EEGLAB format
    all_paths = mne_bids.find_matching_paths(
        suffixes=bids_kwargs['datatypes'],
        ignore_json=True,
        ignore_nosub=True,
        **bids_kwargs
    )
    recordings = [x for x in all_paths if '.fdt' not in x.fpath.suffix]

    # Processing loops/parallel
    if cpus > 1:
        with mp.Pool(processes=cpus) as pool:
            epochs_list = pool.starmap(
                _process_bids_dataset,
                [(recording, montage, centering_id, response_id, verbose,
                    preprocessing_fn, preprocessing_kwargs, epoching_kwargs)
                    for recording in recordings
                ],
            )
    else:
        epochs_list = [
            _process_bids_dataset(
                    recording, montage, centering_id, response_id, verbose,
                    preprocessing_fn, preprocessing_kwargs, epoching_kwargs
            )
            for recording in recordings
        ]
    
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

    epoch_data, info = utils._concat_recordings(epoch_data, recordings, montage, bids_kwargs['datatypes'],
                      bids_kwargs, epoching_kwargs, preprocessing_kwargs)
    bids_info = [_parse_bids_name(r) for r in epoch_data.recording.values]

    # Add bids info to xr coords
    coords = {}
    all_keys = bids_info[0].keys()    
    for key in all_keys:
        values = [x[key] for x in bids_info]
        if any(v is not None for v in values):
            coords[key] = ("recording", values)
    epoch_data = epoch_data.assign_coords(coords)
    return epoch_data, info
            
def _process_bids_dataset(recording, montage, centering_id, response_id, verbose,
                          preprocessing_fn, preprocessing_kwargs, epoching_kwargs):
    print(f"Processing dataset {"_".join(str(recording.basename).split("_")[:-1])}")
    
    data = mne_bids.read_raw_bids(
        bids_path = recording,
        verbose=False # Not ideal but too many prints from BIDS warnings
    )
    events, detected_event_id = events_from_annotations(data)
    # MNE bids extracts triggers from annotations but (sometimes?) loses the 
    # original trigger values. The following ensures mapping by matching the 
    # description between the events.tsv and the event_from_annotations
    # and uptating the trigger value in the expected stimulus/response dicts
    new_cent_id, new_resp_id = _bids_to_annot(recording, detected_event_id,
                                               centering_id, response_id, verbose) 
    # User level preprocessing, should include: re-referencing, channel selection
    # filtering and resampling if needed and take data, events, preprocessing_kwargs
    # as input and output data and events, see example in utils.preprocess_raw
    if preprocessing_fn is not None:
        data, events = preprocessing_fn(data, events, preprocessing_kwargs)
    else:
        data, events = utils.preprocess_raw(data, montage, events,
                   preprocessing_kwargs, verbose)
    
    # Epoching + resampling + metadata creation
    epochs, valid_epoch_index = utils._epoching_raw(
        data, events, new_cent_id, new_resp_id, verbose, epoching_kwargs)
    
    return epochs, valid_epoch_index
        
def _bids_to_annot(path, detected_event_id, centering_id, response_id, verbose):
    path_to_tsv = path.copy().update(suffix="events", extension=".tsv")
    events_dict = mne_bids.events_file_to_annotation_kwargs(path_to_tsv)
    read_event_id = events_dict['event_id']
    # Remap based on user requested dict
    if verbose:
        print(f'Found events {np.sort(list(read_event_id.values()))} in '
              f'{path_to_tsv}, \n mapping to the declared '
              f'triggers: {np.sort(list((centering_id | response_id).values()))}')
    old_stim_id = {v:k for k,v in centering_id.items()} 
    old_resp_id = {v:k for k,v in response_id.items()}
    new_stim_id = {}
    new_resp_id = {}
    for k,v in read_event_id.items():
        new_v = detected_event_id[k]
        if v in centering_id.values():
            new_stim_id[old_stim_id[v]] = int(new_v)
        elif v in response_id.values():
            new_resp_id[old_resp_id[v]] = int(new_v)
    return new_stim_id, new_resp_id

def check_bids_kwargs(bids_kwargs):
    print(bids_kwargs)
    if 'root' not in bids_kwargs:
        raise ValueError("No 'root' directory provided for this BIDS dataset")
    for key in bids_kwargs.keys():
        if key != 'root' and  not isinstance(bids_kwargs[key], list):
            raise ValueError(f"The provided '{key}' variable should be a list")
    if len(bids_kwargs['datatypes']) > 1:
        raise NotImplementedError('Reading multiple datatype is not supported yet, open an issue '
                                  'on https://github.com/GWeindel/hmp/ if this is needed')

def _parse_bids_name(name):
    pattern = (
        r"sub-(?P<subject>[A-Za-z0-9]+)"
        r"(?:_ses-(?P<ses>[A-Za-z0-9]+))?"
        r"(?:_task-(?P<task>[A-Za-z0-9]+))?"
        r"(?:_acq-(?P<acq>[A-Za-z0-9]+))?"
        r"(?:_run-(?P<run>[A-Za-z0-9]+))?"
    )
    match = re.match(pattern, name)
    return match.groupdict()
