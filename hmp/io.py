import numpy as np
import xarray as xr
from pandas import DataFrame
from pathlib import Path
import pickle

def read_mne_data(
    pfiles: str | list,
    event_id: dict | None = None,
    resp_id: dict | None = None,
    epoched: bool = False,
    sfreq: float | None = None,
    subj_idx: list | None = None,
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
) -> xr.Dataset:
    """Read EEG/MEG data format (.fif or .bdf) using MNE's integrated function.

    Notes
    -----
    - Only EEG or MEG data are selected (other channel types are discarded).
    - All times are expressed on the second scale.
    - If multiple files in `pfiles`, the data of the group is read and sequentially processed.
    - For non-epoched data: Reaction Times are only computed if the response trigger is in the epoch
      window (determined by `tmin` and `tmax`).

    Procedure:
    If data is not already epoched:
    0.1) The data is filtered with filters specified in `low_pass` and `high_pass`.
         Parameters of the filter are determined by MNE's filter function.
    0.2) If no events are provided, detect events in the stimulus channel and keep events with IDs
         in `event_id` and `resp_id`.
    0.3) Eventual downsampling is performed if `sfreq` is lower than the data's sampling frequency.
         The event structure is passed to the `resample()` function of MNE to ensure that events
         are appropriately timed after downsampling.
    0.4) Epochs are created based on stimulus onsets (`event_id`) and `tmin` and `tmax`. Epoching
         removes any epoch where a 'BAD' annotation is present. Epochs are baseline corrected from
        `tmin` to stimulus onset (time 0).

    1) Reaction times (RT) are computed based on the sample difference between the onset of the
       stimulus and response triggers. If no response event happens after a stimulus in the epochs window
        or if RT > `upper_limit_rt` or RT < `lower_limit_rt`, RT is set to 0.
    2) All the non-rejected epochs with positive RTs are cropped to stimulus onset to
       stimulus_onset + RT.    Parameters
    ----------
    pfiles : str or list
        List of EEG files to read. Can be a single file path or a list of file paths.
    event_id : dict, optional
        Dictionary mapping condition names [keys] to event codes [values].
    resp_id : dict, optional
        Dictionary mapping response names [keys] to event codes [values].
    epoched : bool, default=False
        Whether the data is already epoched.
    sfreq : float, optional
        Desired sampling frequency for downsampling.
    subj_idx : list, optional
        List of subject identifiers. If not provided, defaults to "S0", "S1", etc.
    metadata : list, optional
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
        Threshold for rejecting epochs based on signal amplitude within the interval stimulus response
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

    import mne
    dict_datatype = {False: "continuous", True: "epoched"}
    epoch_data = []
    if isinstance(pfiles, (str, Path)):  # only one participant
        pfiles = [pfiles]
    if not subj_idx:
        subj_idx = ["S" + str(x) for x in np.arange(len(pfiles))]
    if isinstance(subj_idx, str):
        subj_idx = [subj_idx]
    if upper_limit_rt < 0 or lower_limit_rt < 0:
        raise ValueError("Limit to RTs cannot be negative")
    y = 0
    if metadata is not None:
        if len(pfiles) > 1 and len(metadata) != len(pfiles):
            raise ValueError(
                f"Incompatible dimension between the provided metadata {len(metadata)} and the "
                f"number of eeg files provided {len(pfiles)}"
            )
    else:
        metadata_i = None
    
    ev_i = 0  # syncing up indexing between event and raw files
    for participant in pfiles:
        print(f"Processing participant {participant}'s {dict_datatype[epoched]} {pick_channels}")

        # loading data
        if epoched is False:  # performs epoching on raw data
            if Path(participant).suffix == ".fif":
                data = mne.io.read_raw_fif(participant, preload=True, verbose=verbose)
            elif Path(participant).suffix == ".bdf":
                data = mne.io.read_raw_bdf(participant, preload=True, verbose=verbose)
            else:
                raise ValueError(f"Unknown EEG file format for participant {participant}")
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
            elif len(events_provided[0]) == 3:
                events_provided = events_provided[np.newaxis]
                events = events_provided[y]
            else:  # assumes stacked event files
                events = events_provided[ev_i]
                ev_i += 1
            if reference is not None:
                data = data.set_eeg_reference(reference)
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
                metadata_i = metadata[y]
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
            metadata_i = epochs.metadata
        else:
            if Path(participant).suffix == ".fif":
                epochs = mne.read_epochs(participant, preload=True, verbose=verbose)
                if high_pass is not None or low_pass is not None:
                    epochs.filter(high_pass, low_pass, fir_design="firwin", verbose=verbose)
                if sfreq is None:
                    sfreq = epochs.info["sfreq"]
                elif sfreq < epochs.info["sfreq"]:
                    if verbose:
                        print(f"Resampling data at {sfreq}")
                    epochs = epochs.resample(sfreq)
            else:
                raise ValueError("Incorrect file format")
            if reference is not None:
                epochs = epochs.set_eeg_reference(reference)
            epochs = epochs.pick(pick_channels) 

            if metadata is None:
                try:
                    metadata_i = epochs.metadata  # accounts for dropped epochs
                except:
                    raise ValueError("Missing metadata in the epoched data")
            elif isinstance(metadata, DataFrame):
                if len(pfiles) > 1:
                    metadata_i = metadata[
                        y
                    ].copy()  # TODO, better account for participant's wide provided metadata
                else:
                    metadata_i = metadata.copy()
            else:
                raise ValueError(
                    "Metadata should be a pandas data-frame as generated by mne or be contained "
                    "in the passed epoch data"
                )
        if upper_limit_rt == np.inf:
            upper_limit_rt = epochs.tmax - offset_after_resp + 1 * (1 / sfreq)
        if ignore_rt:
            metadata_i[rt_col] = epochs.tmax
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
        offset_after_resp_samples = np.rint(offset_after_resp * sfreq).astype(int)
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
        triggers = metadata_i.iloc[:, 0].values  # assumes first col is trigger
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
        print(f"{len(cropped_data_epoch)} trials were retained for participant {participant}")
        if verbose:
            print(f"End sampling frequency is {sfreq} Hz")

        epoch_data.append(
            hmp_data_format(
                cropped_data_epoch,
                epochs.info["sfreq"],
                None,
                offset_after_resp_samples,
                epochs=[int(x) for x in epochs_idx],
                channel=epochs.ch_names,
                metadata=metadata_i,
            )
        )

        y += 1
    epoch_data = xr.concat(
        epoch_data,
        dim=xr.DataArray(subj_idx, dims="participant"),
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

def save_model(model, filename):
    with open(filename, 'wb') as output:
        pickle.dump(model, output)

def load_model(filename):
    with open(filename, 'rb') as pkl_file:
        model = pickle.load(pkl_file)
    return model

def save_eventprobs_csv(estimates, filename):
    """Save eventprobs to filename csv file."""
    estimates = estimates.unstack()
    estimates.to_dataframe('eventprobs').to_csv(filename)
    print(f"Saved at {filename}")
