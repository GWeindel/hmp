"""Preprocessing functions.

This module provides functions for preprocessing either raw or epoched data
"""
import numpy as np
import xarray as xr
from mne import Epochs, EpochsArray, Info, pick_info, pick_types
from mne.channels import DigMontage, make_dig_montage, make_standard_montage
from mne.epochs import EpochsFIF
from mne.io.constants import FIFF
from mne.io.fiff.raw import Raw
from mne.preprocessing import compute_current_source_density


def preprocess_data(data: Raw | Epochs,
                   montage: str | DigMontage | None,
                   events: np.ndarray | None,
                   verbose: bool,
                   preprocessing_kwargs: dict,
                    ) -> (Raw, np.ndarray):
    """
    Apply minimal preprocessing on the raw or epoched M/EEG data.

    This function:
        1) re-references the data
        2) applies the desired channel montage
        3) Filters and resamples the data

    Parameters
    ----------
    data : mne.Raw | mne.Epochs
        MNE Raw or epoched object
    montage: str or mne.channels.DigMontage
        Either an MNE DigMontage or a string for a bulit-in MNE montage (see
        mne.channels.get_builtin_montages()) that is applied to all recordings.
    events: np.ndarray
        A 2D numpy array with dimension:
        event (one row per trigger) X description (sample, 0, trigger code)
    preprocessing_kwargs: dict
        arguments to be passed to the preprocessing functions. If no
        'preprocessing_fn' is specified, only the following keys are relevant:
            highpass : float
                high pass filter provided to MNE's filtering function
            lowpass : float
                lowpass filter provided to MNE's filtering function
            sfreq: float
                Desired sampling frequency, can only be lower or equal to the one of the data.
                When the downsampling is performed on the raw data which can result in time jitter
                in the event triggers. This is minimzed by providing the events to the
                resampling function. Users who prefer to perform that at the epoch level can use
                the 'decim' argument in epoching_kargs
            reference: str
                Electrodes or method to use for referencing (see mne.set_eeg_reference).
                Average (common reference) or REST are highly recommended to fit HMP models.
            pick_channels: list of str
                Channels to use, can be list of channel names or 'eeg'/'meg'
    verbose: bool
        Whether to print outputs or not

    Returns
    -------
    data: Raw
        The preprocessed mne.Raw object
    events: np.ndarray
        The events recorded in the data eventually resampled to the new sampling frequency
    """
    # Load data for filtering/resampling
    data.load_data()

    # Dealing with some datasets with unexpected capitalization
    data.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'}, on_missing='ignore')

    # Select channels
    data = data.pick(preprocessing_kwargs["pick_channels"])

    # Apply the desired montage
    if montage is not None:
        data = _apply_montage(data, montage)

    # Set the reference
    if preprocessing_kwargs["reference"] is not None:
        if preprocessing_kwargs["reference"] == 'REST' and montage is None:
            raise ValueError('Cannot use REST reference without a valid montage')
        data = data.set_eeg_reference(preprocessing_kwargs["reference"])

    # Resample here to fasten preprocessing steps, feed events to avoid
    # timing problem after resampling, if user prefer epoching resample
    # they can use the 'decim' argument in epoching_kwargs
    data, events = _filtering_resampling(data, preprocessing_kwargs, events, verbose)
    return data, events

def _filtering_resampling(data, preprocessing_kwargs, events, verbose):
    lowpass = preprocessing_kwargs['lowpass']
    if preprocessing_kwargs['sfreq'] is not None:
        if preprocessing_kwargs['sfreq'] < data.info["sfreq"]:  # Downsampling
            if lowpass is None:
                lowpass = preprocessing_kwargs['sfreq'] / 3.1
            elif lowpass > preprocessing_kwargs['sfreq'] / 3.1:
                raise ValueError(f"Requested low pass filter of {lowpass}"
                     "is too high for desired sampling frequency of "
                     f"{preprocessing_kwargs['sfreq']}")
    if preprocessing_kwargs['highpass'] is not None or lowpass is not None:
        data.filter(l_freq=preprocessing_kwargs['highpass'], h_freq=lowpass, verbose=verbose)
    if preprocessing_kwargs['sfreq'] is not None:
        if isinstance(data, EpochsFIF):
            data = data.resample(preprocessing_kwargs['sfreq'], verbose=verbose)
        else:
            data, events = data.resample(preprocessing_kwargs['sfreq'],
                                         events=events, verbose=verbose)
    return data, events


def _apply_montage(data, montage):
    montage = _create_montage(data.ch_names, montage)
    # Correct for eventual capitalization differences
    data.rename_channels({c:n for c,n in zip(data.ch_names,
        [ch for ch in montage.ch_names if ch.lower() in\
         [x.lower() for x in data.info["ch_names"]]])})
    data.set_montage(montage)
    return data


def _create_montage(ch_names, montage):
    if isinstance(montage, str):
        montage = make_standard_montage(montage)
    elif not isinstance(montage, DigMontage):
        raise ValueError("Unrecognized montage object, should either be a string"
                        "from one of the list in mne.channels.get_builtin_montages()"
                        "or a mne.DigMontage")
    pos = montage.get_positions()['ch_pos']

    montage = make_dig_montage(
        ch_pos={ch: pos[ch] for ch in ch_names},
        coord_frame='head'
    )
    return montage

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
        epoch_data = epoch_data.stack(trial=['recording','epoch']).dropna("trial", how="all")
        for trial in epoch_data.trial:
            trial_dat = epoch_data.sel(trial=trial).data
            # Build fake Epoch mne class and use MNE's dedicated function
            epoch = EpochsArray(np.array([trial_dat.values]), eeg_info)
            epoch = compute_current_source_density(epoch, verbose=False)
            epoch_data['data'].loc[dict(trial=trial)] = epoch.get_data()[0]
        epoch_data = epoch_data.unstack()
        # Set EEG channels to the correct CSD unit

        for ch in eeg_info['chs']:
            ch['unit'] = FIFF.FIFF_UNIT_V_M2

    else:
        raise ValueError(f"Cannot apply CSD on channels with units {info['chs'][0]['unit']}")
    return epoch_data, eeg_info
