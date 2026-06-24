"""Utilities to read data from different format."""
import inspect
import numpy as np
from pandas import DataFrame
import xarray as xr
import os 

from mne import Epochs, create_info
from mne.io import Raw
from mne.epochs import make_metadata, EpochsFIF
from mne.channels import DigMontage, make_standard_montage, make_dig_montage
from warnings import warn
    
def _defaults_check_epoching(kwargs):
    expected = list(inspect.signature(Epochs).parameters)
    critical = {"tmin":-.2,"tmax":2}
    for key, value in critical.items():
        if key not in kwargs:
            warn(f"No '{key}' provided, using default value of {value} seconds \n"
                "use 'epoching_kwargs' to override default values")
            kwargs.setdefault(key, value)
    kwargs.setdefault("proj", False)
    kwargs.setdefault("baseline", (None, 0))
    kwargs.setdefault("detrend", None)
    kwargs.setdefault("on_missing", "warn")
    kwargs.setdefault("event_repeated", "drop")
    kwargs.setdefault("reject_by_annotation", False)
    if not set(expected).issuperset(kwargs):
        raise ValueError("Got unexpected argument for function mne.Epoch: "
                        f"{set(kwargs).difference(expected)} See list of possible arguments"
                        "and defaults at https://mne.tools/stable/generated/mne.Epochs.html")
    return kwargs

def _defaults_check_prep(kwargs, preprocessing_fn):
    expected = ["highpass","lowpass","sfreq","reference","pick_channels"]
    defaults = [0.01, 40, 200, 'average', 'eeg']
    for key, value in zip(expected, defaults):
        if key not in kwargs:
            warn(f"No '{key}' provided, using default value: {value}.\n"
                "use preprocessing_kwargs to override default values")
            kwargs.setdefault(key, value)

    if preprocessing_fn is None and len(set(kwargs).difference(expected)) > 0:
        raise ValueError("Got unexpected argument for preprocessing"
            f"{set(kwargs).difference(expected)} "
            "use 'prepocessing_fn' is further preprocessing steps are needed")
    return kwargs

def _format_trigger_description(stimulus_id, response_id):
    if len(stimulus_id.keys()) == 0:
        raise ValueError('At lease one centering event needs to be provided')
    if any(not k.startswith("stimulus/") for k in stimulus_id.keys()):
        stimulus_id = {f"stimulus/{k}": v for k, v in stimulus_id.items()}    
    if any(not k.startswith("response/") for k in response_id.keys()):
        response_id = {f"response/{k}": v for k, v in response_id.items()}
    return stimulus_id, response_id

def preprocess_data(data: Raw | Epochs,
                   montage: str | DigMontage | None,
                   events: np.ndarray | None,
                   preprocessing_kwargs: dict,
                   verbose: bool) -> (Raw, np.ndarray):
    """
    Apply minimal preprocessing on the raw, continuous, M/EEG data.

    This function:
        1) re-references the data
        2) applies the desired channel montage
        3) Filters and resamples the data

    Parameters
    ----------
    data : mne.Raw
        MNE Raw object
    montage: str or mne.channels.DigMontage
        Either an MNE DigMontage or a string for a bulit-in MNE montage (see 
        mne.channels.get_builtin_montages()) that is applied to all recordings.
    events: np.ndarray
        A 2D numpy array with dimension event (one row per trigger) X description (sample, 0, trigger code)
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
    data.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'}, on_missing='ignore')
    # Select channels
    data = data.pick(preprocessing_kwargs["pick_channels"])

    # Apply the desired montage
    if montage is not None:
        data = _apply_montage(data, montage)

    # Set the reference
    if preprocessing_kwargs["reference"] is not None:
        if preprocessing_kwargs['reference'] == 'REST' and montage is None:
            raise ValueError('Cannot use REST reference without a valid montage')
        data = data.set_eeg_reference(preprocessing_kwargs["reference"])

    # Resample here to fasten preprocessing steps, feed events to avoid
    # timing problem after resampling, if user prefer epoching resample 
    # they can use the 'decim' argument in epoching_kwargs
    data, events = _filtering_resampling(data, preprocessing_kwargs, events, verbose)
    return data, events
    
def _create_montage(ch_names, montage):
    if isinstance(montage, str):
        montage = make_standard_montage(montage)
    elif not isinstance(montage, DigMontage):
        raise ValueError("Unrecognized montage object, should either be a string"
                        "from one of the list in mne.channels.get_builtin_montages()"
                        "or a mne.DigMontage")
    montage
    pos = montage.get_positions()['ch_pos']
    
    montage = make_dig_montage(
        ch_pos={ch: pos[ch] for ch in ch_names},
        coord_frame='head'
    )
    return montage

def _apply_montage(data, montage):
    montage = _create_montage(data.ch_names, montage)
    # Correct for eventual capitalization differences
    data.rename_channels({c:n for c,n in zip(data.ch_names,
        [ch for ch in montage.ch_names if ch.lower() in\
         [x.lower() for x in data.info["ch_names"]]])})
    data.set_montage(montage)
    return data

def _filtering_resampling(data, preprocessing_kwargs, events, verbose):
    lowpass = preprocessing_kwargs['lowpass']
    if preprocessing_kwargs['sfreq'] is not None:
        if preprocessing_kwargs['sfreq'] < data.info["sfreq"]:  # Downsampling
            if lowpass is None:
                lowpass = preprocessing_kwargs['sfreq'] / 3.1
            elif lowpass > preprocessing_kwargs['sfreq'] / 3.1:
                raise ValueError(f"Requested low pass filter of {lowpass}"
                     f"is too high for desired sampling frequency of {preprocessing_kwargs['sfreq']}")
    if preprocessing_kwargs['highpass'] is not None or lowpass is not None:
        data.filter(l_freq=preprocessing_kwargs['highpass'], h_freq=lowpass, verbose=verbose)
    if preprocessing_kwargs['sfreq'] is not None:
        if isinstance(data, EpochsFIF):
            data = data.resample(preprocessing_kwargs['sfreq'])
        else:
            data, events = data.resample(preprocessing_kwargs['sfreq'], events=events)
    return data, events

def _epoching_raw(data, events, stimulus_id, response_id, verbose, epoching_kwargs):
    if len(stimulus_id) == 0:
        raise ValueError("No valid centering_id found in the data "
                         f"detected triggers : {np.unique(events[:,2])}")
    event_id = {**stimulus_id, **response_id}
    stim = list(stimulus_id.keys())

    if len(response_id) > 0:
        keep_first=["response"]
        cols = ["event_name", "response"]
    else:
        keep_first=[]
        cols = ["event_name"]

    metadata_i, meta_events, stimulus_id = make_metadata(
        events=events,
        event_id=event_id,
        tmin=epoching_kwargs['tmin'],
        tmax=epoching_kwargs['tmax'],
        sfreq=data.info["sfreq"],
        row_events=stim,
        keep_first=keep_first,
    )
    if 'first_response' in metadata_i.columns:
        cols.append('first_response')
    metadata_i = metadata_i[cols]  # only keep event_names and rts

    epochs = Epochs(
        data,
        meta_events,
        stimulus_id,
        preload=True,
        metadata=metadata_i,
        **epoching_kwargs
    )
    epochs.metadata.rename({"response": "rt", "first_response":"response"}, axis=1, inplace=True, errors='ignore')

    valid_epoch_index = [x for x, y in enumerate(epochs.drop_log) if len(y) == 0]
    return epochs, valid_epoch_index

def hmp_data_format(
    data: np.ndarray,
    sfreq: float,
    tmin: float,
    tmax: float,
    epochs: list | None = None,
    channel: list | None = None,
    metadata: DataFrame | None = None,
) -> xr.Dataset:

    """
    Convert data to the expected xarray Dataset format.

    This function reshapes a 3D matrix with dimensions trial * channel * sample
    into an xarray Dataset as expected by HMP.

    Parameters
    ----------
    data : np.ndarray
        3D matrix with dimensions trial X channel X sample.
    sfreq : float
        Sampling frequency of the data.
    tmin : float
        Start time (in seconds) relative to stimulus onset for epoching.
    tmax : float
        End time (in seconds) relative to stimulus onset for epoching.
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

def create_info_hmp(ch_names: list[str],
                    montage: str | DigMontage,
                    preprocessing_kwargs: dict,
                    datatype:str):
    '''Create minimal info object for plotting in hmp.visu
    
    Parameters
    ----------
    ch_names: list of str
        List of channels in the data
    montage: str or mne.channels.DigMontage
        Either an MNE DigMontage or a string for a bulit-in MNE montage (see 
        mne.channels.get_builtin_montages()) that is applied to all recordings.
    preprocessing_kwargs: dict
        Dictionnary containing values for sfreq and high/lowpass filters
    datatype: str
        MNE compatible data type in the data (e.g. 'eeg' or 'meg')
    '''
    info = create_info(ch_names=ch_names, sfreq=preprocessing_kwargs['sfreq'],
                       ch_types=np.repeat(datatype, len(ch_names)))
    if montage is not None:
        montage = _create_montage(ch_names, montage)
        info.set_montage(montage)
    return info

def _concat_recordings(epoch_data, recordings, montage, datatype,
                      epoching_kwargs={}, preprocessing_kwargs={}, subj_names=None):
    '''Concatenate list of xr.Datasets into a common xr.Dataset
    '''
    recordings = ["_".join(str(recording.name).split("_")[:-1])
                  for recording in recordings]
    # Data
    epoch_data = xr.concat(
        epoch_data,
        dim=xr.DataArray(recordings, dims="recording"),
        fill_value={"event": "", "data": np.nan},
        join='outer',
        combine_attrs='identical',#Throw error if not the same att
    )
    if subj_names is not None:
        epoch_data = epoch_data.assign_coords({'subject': ("recording", subj_names)})

    # Attributes
    n_trials = (
        (~np.isnan(epoch_data.data[:, :, :, 0].data)).sum(axis=1)[:, 0].sum()
    )  # Compute number of trial based on trial where first sample is nan
    # Creating info, should all be shared
    info = create_info_hmp(list(epoch_data.channel.values),
                       montage, preprocessing_kwargs, datatype)

    epoch_data = epoch_data.assign_attrs(
        **epoching_kwargs,
        **preprocessing_kwargs
    )
    return epoch_data, info

def _check_montage(montage):
    if montage is None:
        warn("No montage was provided, HMP plotting functions cannot be used without a "
            "valid template montage. If using standard channel montage declare one "
            "of MNE's built-in montage (see mne.channels.get_builtin_montages()) in "
            "the 'montage' argument, alternatively provide an mne.DigMontage object")    