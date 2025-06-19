"""EEG/MEG Data Processing Utilities.

This module provides functions for reading, processing, and saving EEG/MEG data using MNE, xarray, and pandas.
It supports reading raw or epoched data, event/response detection, reaction time trimming, epoch cropping,
metadata handling, and conversion to xarray Datasets for fitting hmp models. Additional utilities are provided
for saving/loading data and models, and exporting event probabilities.
"""

import numpy as np
import xarray as xr
from pandas import DataFrame
from pathlib import Path
import pickle
import os 
import mne
import json

def read_mne_data(
    pfiles: str | list,
    event_id: dict | None = None,
    resp_id: dict | None = None,
    data_format: str = 'raw',
    sfreq: float | None = None,
    subj_name: list | None = None,
    metadata: list | None = None,
    events_provided: np.ndarray | None = None,
    rt_col: str = "rt",
    rts: np.ndarray | None = None,
    verbose: bool = True,
    tmin: float = -0.2,
    tmax: float = 5,
    offset_after_resp: float = 0,
    high_pass: float | None = None,
    low_pass: float | None = None,
    pick_channels: str | list = "eeg",
    baseline: tuple = (None, 0),
    upper_limit_rt: float = np.inf,
    lower_limit_rt: float = 0,
    reject_threshold: float | None = None,
    scale: float = 1,
    reference: str | None = None,
    ignore_rt: bool = False,
    bids_parameters: dict = {}
) -> xr.Dataset:
    """Read EEG/MEG data format (.fif or .bdf) using MNE's integrated function.

    Notes
    -----
    - Only EEG or MEG data are selected (other channel types are discarded).
    - All times are expressed in seconds.
    - If multiple files are provided in ``pfiles``, each participant's data is read and processed sequentially.
    - For non-epoched data: Reaction Times are only computed if the response trigger is in the epoching
      window (determined by ``tmin`` and ``tmax``).

    ## Procedure:
    
    If data is not already epoched:

        - The data is filtered using the specified ``low_pass`` and ``high_pass`` parameters.
        - If no events are provided, events are detected in the stimulus channel and only those with IDs
          in ``event_id`` and ``resp_id`` are kept.
        - Downsampling is performed if ``sfreq`` is lower than the data's sampling frequency.
        - Epochs are created based on stimulus onsets (``event_id``) and the ``tmin``/``tmax`` window.
          Epochs with 'BAD' annotations are removed. Baseline correction is applied from
          ``tmin`` to stimulus onset (time 0).

    Then (or if data is already epoched):

        1. Reaction times (RT) are computed as the time difference between stimulus and response triggers.
           If no response event occurs after a stimulus in the epoch window, or if
           ``RT > upper_limit_rt`` or ``RT < lower_limit_rt``, RT is set to 0.
        2. All non-rejected epochs with positive RTs are cropped from stimulus onset to
           ``stimulus_onset + RT``.

    Parameters
    ----------
    pfiles : str or list of str
        Path(s) to EEG files to read. Can be a single file path or a list of file paths.
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
    rt_col : str, default="rt"
        Column name in metadata containing reaction times.
    rts : np.ndarray, optional
        Array of reaction times. Used if metadata is not provided.
    verbose : bool, default=True
        Whether to display MNE's messages.
    tmin : float, default=-0.2
        Start time (in seconds) relative to stimulus onset for epoching.
    tmax : float, default=5
        End time (in seconds) relative to stimulus onset for epoching.
    offset_after_resp : float, default=0
        Additional time (in seconds) to include after the response onset.
    high_pass : float, optional
        High-pass filter cutoff frequency.
    low_pass : float, optional
        Low-pass filter cutoff frequency.
    pick_channels : str or list, default="eeg"
        Channels to retain. Use "eeg"/"meg" to keep only EEG/MEG channels or provide a list of channel names.
    baseline : tuple, default=(None, 0)
        Time range for baseline correction (start, end) in seconds.
    upper_limit_rt : float, default=np.inf
        Upper limit for reaction times. Longer RTs are discarded.
    lower_limit_rt : float, default=0
        Lower limit for reaction times. Shorter RTs are discarded.
    reject_threshold : float, optional
        Threshold for rejecting epochs based on signal amplitude within the stimulus-response interval.
    scale : float, default=1
        Scaling factor for reaction times (e.g., 1000 for milliseconds).
    reference : str, optional
        Reference to use for EEG data. If None, the existing reference is kept.
    ignore_rt : bool, default=False
        Whether to ignore reaction times and parse epochs up to `tmax`.

    Returns
    -------
    epoch_data : xarray.Dataset
        An xarray Dataset containing the processed EEG/MEG data, events, channels, and participants.
        Metadata and epoch indices are preserved. The chosen sampling frequency is stored as an attribute.
    """

    epoch_data = []
    if isinstance(pfiles, (str, Path)):  # only one participant
        pfiles = [pfiles]
    if not subj_name:
        subj_name = ["S" + str(x) for x in np.arange(len(pfiles))]
    if isinstance(subj_name, str):
        subj_name = [subj_name]
    subj_idx = 0
    if metadata is not None:
        if len(pfiles) > 1 and len(metadata) != len(pfiles):
            raise ValueError(
                f"Incompatible dimension between the provided metadata {len(metadata)} and the "
                f"number of eeg files provided {len(pfiles)}"
            )
    else:
        metadata_i = None
    
    if data_format == 'bids':
        subj_name = pfiles = [d for d in os.listdir(bids_parameters['bids_root']) if d.startswith("sub-") and os.path.isdir(os.path.join(bids_parameters['bids_root'], d))]
        # try:
        event_id, resp_id = _bids_extract_trig(
            bids_parameters['bids_root'],
            bids_parameters['task'],
            bids_parameters['datatype']
        )
        # except:
        #     raise ValueError(f"Wrong BIDS specification {bids_parameters['bids_root']}")
    
    ev_i = 0  # syncing up indexing between event and raw files
    for participant in pfiles:
        print(f"Processing participant {participant}'s {data_format} {pick_channels}")
        if data_format == 'epochs':
            epochs = _read_mne_epochs(participant,
                    sfreq,
                    metadata,
                    high_pass,
                    low_pass,
                    pick_channels,
                    verbose)

        elif data_format == 'raw' or data_format == 'bids':
            epochs = read_raw_and_epoch(participant,
                            pfiles,
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
                            baseline,
                            pick_channels,
                            bids_parameters)
        else:
            raise ValueError(f"Unknown data type {data_format}, should be 'epochs', 'raw' or 'bids'")
        
        if reference is not None:
            epochs = epochs.set_eeg_reference(reference)
        
        epoch_data.append(_epoch_selection(
            epochs,
            metadata,
            pfiles,
            participant,
            subj_idx,
            rt_col,
            scale,
            offset_after_resp,
            sfreq,
            lower_limit_rt,
            upper_limit_rt,
            reject_threshold,
            ignore_rt,
            verbose
        ))
        
        subj_idx += 1
    
    epoch_data = xr.concat(
        epoch_data,
        dim=xr.DataArray(subj_name, dims="participant"),
        fill_value={"event": "", "data": np.nan},
    )
    n_trials = (
        (~np.isnan(epoch_data.data[:, :, :, 0].data)).sum(axis=1)[:, 0].sum()
    )  # Compute number of trial based on trial where first sample is nan
    epoch_data = epoch_data.assign_attrs(
        lowpass=epochs.info["lowpass"],
        highpass=epochs.info["highpass"],
        lower_limit_rt=lower_limit_rt,
        upper_limit_rt=upper_limit_rt,
        reject_threshold=reject_threshold,
        n_trials=n_trials,
    )
    return epoch_data

def _bids_extract_trig(bids_root, task, datatype):

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

def _bids_extract_events(raw, stim_id, resp_id, verbose):
    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    
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
    metadata,
    high_pass,
    low_pass,
    pick_channels,
    verbose
):

    if Path(participant).suffix == ".fif": 
        epochs = mne.read_epochs(participant, preload=True, verbose=verbose)
    else:
        raise ValueError("Incorrect file format")
    
    if high_pass is not None or low_pass is not None:
        epochs.filter(high_pass, low_pass, fir_design="firwin", verbose=verbose)
       
    if sfreq is None:
        sfreq = epochs.info["sfreq"]
    elif sfreq < epochs.info["sfreq"]:
        if verbose:
            print(f"Resampling data at {sfreq}")
        epochs = epochs.resample(sfreq)

    
    if metadata is None:
        try:
            metadata_i = epochs.metadata  # accounts for dropped epochs
        except:
            raise ValueError("Missing metadata in the epoched data")
    elif isinstance(metadata, DataFrame):
        if len(pfiles) > 1:
            metadata_i = metadata[
                y
            ].copy()
        else:
            metadata_i = metadata.copy()
    else:
        raise ValueError(
            "Metadata should be a pandas data-frame as generated by mne or be contained "
            "in the passed epoch data"
        )
    epochs = epochs.pick(pick_channels) 
    return epochs


def read_raw_and_epoch(
    participant,
    pfiles,
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
    baseline,
    pick_channels,
    bids_parameters
):
    if Path(participant).suffix == ".fif":
        data = mne.io.read_raw_fif(participant, preload=True, verbose=verbose)
    elif Path(participant).suffix == ".bdf":
        data = mne.io.read_raw_bdf(participant, preload=True, verbose=verbose)
    elif isinstance(bids_parameters, dict) and len(bids_parameters) > 0:
        import mne_bids
        bids_path = mne_bids.BIDSPath(subject=participant.replace("sub-", ""), task=bids_parameters['task'],
                                      root=bids_parameters['bids_root'],
                                      session = bids_parameters['session'],
                                      datatype=bids_parameters['datatype'])

        data = mne_bids.read_raw_bids(
            bids_path = bids_path,
            verbose=False
        )
        events_provided = _bids_extract_events(data, event_id, resp_id, verbose)
    else:
        raise ValueError(f"Unknown EEG file format for participant {participant}, only '.bdf' and '.fif' or BIDS are accepted")
    if sfreq is None:
        sfreq = data.info["sfreq"]

    if "response" not in list(resp_id.keys())[0]:
        resp_id = {f"response/{k}": v for k, v in resp_id.items()}
    if events_provided is None:
        try:
            events = mne.find_events(
                data, verbose=verbose, min_duration=1 / data.info["sfreq"]
            )
        except:
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
        low_pass = obtained_sfreq / 3.1
    else:
        decim = 1
        if sfreq > data.info["sfreq"] + 1:
            warn(
                f"Requested higher frequency {sfreq} than found in the EEG data, no "
                f"resampling is performed"
            )
    if high_pass is not None or low_pass is not None:
        data.filter(high_pass, low_pass, fir_design="firwin", verbose=verbose)
    combined = {**event_id, **resp_id}  # event_id | resp_id
    stim = list(event_id.keys())

    if verbose:
        print(f"Creating epochs based on following event ID :{np.unique(events[:, 2])}")

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
        metadata_i = metadata_i[["event_name", "response"]]  # only keep event_names and rts
    else:
        metadata_i = metadata[subj_idx]
    epochs = mne.Epochs(
        data,
        meta_events,
        event_id,
        tmin,
        tmax,
        proj=False,
        baseline=baseline,
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
    epochs.metadata.rename({"response": "rt"}, axis=1, inplace=True)
    return epochs

def _epoch_selection(epochs,
                    metadata,
                    pfiles,
                    participant,
                    subj,
                    rt_col,
                    scale,
                    offset_after_resp,
                    sfreq,
                    lower_limit_rt,
                    upper_limit_rt,
                    reject_threshold,
                    ignore_rt,
                    verbose
    ):
    if metadata is None:
        try:
            metadata_i = epochs.metadata  # accounts for dropped epochs
        except:
            raise ValueError("Missing metadata in the epoched data")
    elif isinstance(metadata, DataFrame):
        if len(pfiles) > 1:
            metadata_i = metadata[
                subj
            ].copy()  # TODO, better account for participant's wide provided metadata
        else:
            metadata_i = metadata.copy()
    else:
        raise ValueError(
            "Metadata should be a pandas data-frame as generated by mne or be contained "
            "in the passed epoch data"
        )
    sfreq = epochs.info["sfreq"] if sfreq is None else sfreq
    valid_epoch_index = [x for x, y in enumerate(epochs.drop_log) if len(y) == 0]
    data_epoch = epochs.get_data(copy=False)  # preserves index
    rts = metadata_i[rt_col]
    if isinstance(metadata_i, DataFrame):
        if len(metadata_i) > len(data_epoch):  # assumes metadata contains rejected epochs
            metadata_i = metadata_i.loc[valid_epoch_index]
            rts = metadata_i[rt_col]
        try:
            rts = rts / scale
        except:
            raise ValueError(
                f"Expected column named {rt_col} in the provided metadata file, alternative "
                f"names can be passed through the rt_col parameter"
            )
    elif rts is None:
        raise ValueError("Expected either a metadata Dataframe or an array of Reaction Times")
    rts_arr = np.array(rts)
    triggers = metadata_i.iloc[:, 0].values  # assumes first col is trigger
    offset_after_resp_samples = np.rint(offset_after_resp * sfreq).astype(int)
    
    if not ignore_rt:
        cropped_data_epoch, epochs_idx = _cut_at_rt(
            data_epoch,
            rts_arr,
            triggers,
            offset_after_resp_samples,
            sfreq,
            lower_limit_rt,
            upper_limit_rt,
            epochs,
            reject_threshold,
            valid_epoch_index,
            verbose
        )
    else:
        cropped_data_epoch = data_epoch
        epochs_idx = valid_epoch_index
    print(f"{len(cropped_data_epoch)} trials were retained for participant {participant}")
    if verbose:
        print(f"End sampling frequency is {sfreq} Hz")

    epoch_data = hmp_data_format(
            cropped_data_epoch,
            epochs.info["sfreq"],
            None,
            offset_after_resp_samples,
            epochs=[int(x) for x in epochs_idx],
            channel=epochs.ch_names,
            metadata=metadata_i,
        )
    return epoch_data

def _cut_at_rt(data_epoch, rts, triggers, offset_after_resp_samples, sfreq, lower_limit_rt, upper_limit_rt, epochs, reject_threshold, valid_epoch_index, verbose):
    """
    """
    if upper_limit_rt == np.inf:
        upper_limit_rt = epochs.tmax - (offset_after_resp_samples + 1) / sfreq
    
    if upper_limit_rt < 0 or lower_limit_rt < 0:
        raise ValueError("Limit to RTs cannot be negative")
    rts_arr = np.array(rts)
    if verbose:
        print(
            f"Applying reaction time trim to keep RTs between {lower_limit_rt} and "
            f"{upper_limit_rt} seconds"
        )
    rts_arr[rts_arr > upper_limit_rt] = 0  # removes RT above x sec
    rts_arr[rts_arr < lower_limit_rt] = 0  # removes RT below x sec, determines max events
    rts_arr[np.isnan(rts_arr)] = 0  # too long trial
    rts_arr = np.rint(rts_arr * sfreq).astype(int)
    if verbose:
        print(f"{len(rts_arr[rts_arr > 0])} RTs kept of {len(rts_arr)} clean epochs")
    cropped_data_epoch = np.empty(
        [
            len(rts_arr[rts_arr > 0]),
            len(epochs.ch_names),
            max(rts_arr) + offset_after_resp_samples,
        ]
    )
    cropped_data_epoch[:] = np.nan
    cropped_trigger = []
    epochs_idx = []
    j = 0
    if reject_threshold is None:
        reject_threshold = np.inf
    rej = 0
    time0 = epochs.time_as_index(0)[0]
    for i in range(len(data_epoch)):
        if rts_arr[i] > 0:
            # Crops the epochs to time 0 (stim onset) up to RT
            if (
                np.abs(data_epoch[i, :, time0 : time0 + rts_arr[i] + offset_after_resp_samples])
                < reject_threshold
            ).all():
                cropped_data_epoch[j, :, : rts_arr[i] + offset_after_resp_samples] = data_epoch[
                    i, :, time0 : time0 + rts_arr[i] + offset_after_resp_samples
                ]
                epochs_idx.append(valid_epoch_index[i])  # Keeps trial number
                cropped_trigger.append(triggers[i])
                j += 1
            else:
                rej += 1
                rts_arr[i] = 0
    while np.isnan(cropped_data_epoch[-1]).all():  # Remove excluded epochs based on rejection
        cropped_data_epoch = cropped_data_epoch[:-1]

    if ~np.isinf(reject_threshold):
        print(f"{rej} trials rejected based on threshold of {reject_threshold}")
        
    return cropped_data_epoch, epochs_idx


def hmp_data_format(
    data: np.ndarray,
    sfreq: float,
    events: np.ndarray | None = None,
    offset: float = 0,
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
    offset : float, default=0
        Offset in seconds to apply to the data.
    participants : list, optional
        List of participant indices.
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
    if n_subj < 2:
        data = xr.Dataset(
            {
                "data": (["epoch", "channel", "sample"], data),
            },
            coords={"epoch": epochs, "channel": channel, "sample": np.arange(n_samples)},
            attrs={"sfreq": sfreq, "offset": offset},
        )
    else:
        data = xr.Dataset(
            {
                "data": (["participant", "epoch", "channel", "sample"], data),
            },
            coords={
                "participant": participants,
                "epoch": epoch,
                "channel": channel,
                "sample": np.arange(n_samples),
            },
            attrs={"sfreq": sfreq, "offset": offset},
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
            coords={"participant": participants, "epoch": epoch},
        )
        data = data.set_coords("events")
    return data

def save_xr(data, filename):
    """Save fit."""
    data = data.copy()
    attributes = data.attrs.copy()
    for attr in attributes:
        if isinstance(data.attrs[attr], np.ndarray):
            del data.attrs[attr]
    data.unstack().to_netcdf(filename)

    print(f"{filename} saved")


def load_xr(filename):
    """Load fit or data."""
    with xr.open_dataset(filename) as data:
        data.load()
    if "epoch" in data:
        data = data.stack(trial=["participant", "epoch"]).dropna(
            dim="trial", how="all"
        )

    # Ensures correct order of dimensions for later index use
    if "iteration" in data:
        data = data.transpose(
            "iteration", "trial", "sample", "event"
        )
    elif "trial" in data:
        data = data.transpose(
            "trial", "sample", "event"
        )
    return data.to_dataarray().drop_vars('variable').squeeze()


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
