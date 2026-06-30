"""Utilities to read data from different format."""
import inspect
from warnings import warn

import numpy as np
import xarray as xr
from mne import Epochs, create_info
from mne.channels import DigMontage
from mne.epochs import make_metadata
from pandas import DataFrame


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

def _defaults_check_prep(kwargs):
    expected = ["highpass","lowpass","sfreq","reference","pick_channels"]
    defaults = [None, None, None, None, None]
    for key, value in zip(expected, defaults):
        if key not in kwargs:
            if value is not None:
                warn(f"No '{key}' provided, using default value: {value}.\n"
                    "use preprocessing_kwargs to override default values")
            kwargs.setdefault(key, value)
    if len(set(kwargs).difference(expected)) > 0:
        raise ValueError("Got unexpected argument for preprocessing"
            f"{set(kwargs).difference(expected)} "
            "use 'prepocessing_fn' is further preprocessing steps are needed")
    return kwargs

def _check_trigger_dicts(trigger_dict):
    triggers = [v for k, v in trigger_dict.items()]
    if len(triggers) != len(set(triggers)):
        raise ValueError(f"Duplicate trigger (value) found in {triggers}. "
            "When providing centering or response IDs one description "
            "should correspond to one trigger")

def _format_trigger_description(stimulus_id, response_id):
    if len(stimulus_id.keys()) == 0:
        raise ValueError('At lease one centering event needs to be provided')
    if any(not k.startswith("stimulus/") for k in stimulus_id.keys()):
        stimulus_id = {f"stimulus/{k}": v for k, v in stimulus_id.items()}
    if any(not k.startswith("response/") for k in response_id.keys()):
        response_id = {f"response/{k}": v for k, v in response_id.items()}
    return stimulus_id, response_id

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
        verbose=verbose,
        **epoching_kwargs
    )
    epochs.metadata.rename({"event_name":"stimulus", "response": "duration",
            "first_response":"response"}, axis=1, inplace=True, errors='ignore')

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
                    montage: DigMontage,
                    preprocessing_kwargs: dict,
                    datatype:str):
    """Create minimal info object for plotting in hmp.visu.

    Parameters
    ----------
    ch_names: list of str
        List of channels in the data
    montage: mne.channels.DigMontage
        An MNE DigMontage that is applied to all recordings.
    sfreq: float
        Sampling frequency of the signal
    datatype: str
        MNE compatible data type in the data (e.g. 'eeg' or 'meg')
    """
    info = create_info(ch_names=ch_names, sfreq=preprocessing_kwargs['sfreq'],
                       ch_types=np.repeat(datatype, len(ch_names)))
    if montage is not None:
        info.set_montage(montage)
    return info

def _concat_recordings(epoch_data, recordings,
                      epoching_kwargs={}, preprocessing_kwargs={}, subj_names=None):
    """Concatenate list of xr.Datasets into a common xr.Dataset."""
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
    (
        (~np.isnan(epoch_data.data[:, :, :, 0].data)).sum(axis=1)[:, 0].sum()
    )  # Compute number of trial based on trial where first sample is nan

    epoch_data = epoch_data.assign_attrs(
        **epoching_kwargs,
        **preprocessing_kwargs
    )
    # Convert tuple and None to string as mostly for info and allows easy saving
    epoch_data.attrs = {
        k: (
            v if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            else str(v)
        )
        for k, v in epoch_data.attrs.items()
    }
    return epoch_data

def _check_montage(montage):
    if montage is None:
        warn("No montage was provided, HMP plotting functions cannot be used without a "
            "valid template montage. If using standard channel montage declare one "
            "of MNE's built-in montage (see mne.channels.get_builtin_montages()) in "
            "the 'montage' argument, alternatively provide an mne.DigMontage object")
