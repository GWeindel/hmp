"""MNE epoch data format reading.

This module provides functions for reading MNE epoched data format (.fif only)
"""

import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from mne import read_epochs, set_log_level
from mne.channels import DigMontage
from numpy.typing import DTypeLike
from xarray import Dataset

from hmp.io import preprocessing, utils


def read_mne_epochs(
    recordings: list,
    subj_name: list = None,
    montage: str | DigMontage | None = None,
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
    dtype: np.DTypeLike
        Precision, use np.float32 or np.int64
    cpus : int
        How many cpus to use. If > 1 process several datasets in parallel
    preprocessing_fn: callable
        A user defined function preprocessing the raw data before epoching.
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

    # Same for preprocessing
    if preprocessing_fn is None:
        preprocessing_kwargs = utils._defaults_check_prep(preprocessing_kwargs)

    utils._check_montage(montage)

    if not isinstance(recordings, list):
        raise ValueError("Expected a list of paths to the recordings."
                        f"Got an object of type {type(recordings)} instead")

    recordings = [Path(x) for x in recordings]

    if subj_name is None:
        subj_name = ["_".join(str(i.name).split("_")[:-1]) for i in recordings]

    # Processing loops/parallel
    if cpus == 1:
        epochs_list = [_process_epoch_dataset(
            recording, montage, verbose,
            preprocessing_fn, preprocessing_kwargs
            )
            for recording in recordings
        ]
    else:
        with mp.Pool(processes=cpus) as pool:
            epochs_list = pool.starmap(
                _process_epoch_dataset,
                [(recording, montage, verbose,
            preprocessing_fn, preprocessing_kwargs)
                    for recording in recordings
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
                      {}, final_prep_kwargs, subj_name)

    return epoch_data, info

def _process_epoch_dataset(recording, montage, verbose,
            preprocessing_fn, preprocessing_kwargs):
    if verbose is True or verbose in ["DEBUG","INFO"]:
        print(f"Processing dataset {'_'.join(str(recording.name).split('_')[:-1])}")
    if recording.suffix == ".fif":
        data = read_epochs(recording, verbose=verbose)
    else:
        raise NotImplementedError(f"Extension {recording.suffix}"
        "is not supported yet. if needed, open an "
        "issue on https://github.com/GWeindel/hmp/issues")

    # User level preprocessing, should include: re-referencing, channel selection
    # filtering and resampling if needed and take data, events, preprocessing_kwargs
    # as input and output data and events, see example in utils.preprocess_data
    if preprocessing_fn is not None:
        data, _ = preprocessing_fn(data, montage, None,
                   verbose, preprocessing_kwargs)
    else:
        data, _ = preprocessing.preprocess_data(data, montage, None,
                   verbose, preprocessing_kwargs)
    valid_epoch_index = [x for x, y in enumerate(data.drop_log) if len(y) == 0]
    return data, valid_epoch_index
