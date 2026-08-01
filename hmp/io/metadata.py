"""Read metadata provided as a dataframe.

This module provides functions for reading metadata
"""

import pandas as pd
import xarray as xr


def add_metadata(epoch_data: xr.Dataset,
                  df: pd.DataFrame,
                 recording_id: str = 'recording',
                 epoch_id: str = 'epoch'
                 ):
    """Append metadata to a dataset obtained from the io module.

    This functions adds any column in the dataframe to epoch_data.
    The correspondance between epochs in the epoch_data and the dataframe
    cannot be guaranteed and should be inspected

    Parameters
    ----------
    epoch_data: xr.Dataset
        Dataset obtained from either `read_bids_raw`, `read_mne_raw` or `read_mne_epochs`
    df: pd.DataFrame
        Metadata that needs to be added to `epoch_data` to be used in subsequent HMP fits
        Note: there must be an exact match between participant and trials
    recording_id: str
        Column of the dataframe containing the recording (typically subjects) identifier
    trial_id: str
        Column of the dataframe containing the epoch identifier (typically trial)

    Returns
    -------
    epoch_data: Raw
        Dataset with metadata contained in the passed dataset added.
    """
    df = df.copy()
    if df[epoch_id].min() == 1: #Assume this means 1-based indexing
        df[epoch_id] -= 1

    # Checks:
    if len(set(epoch_data.recording.values)) != len(set(df[recording_id].values)):
        raise ValueError("Cannot align epoch_data and metadata. "
                         "Mismatch between recording dimension :"
                         f"{set(epoch_data.recording.values)} in epoch data, vs "
                         f"{set(df[recording_id].values)} in metadata")

    if len(set(epoch_data.epoch.values)) != len(set(df[epoch_id].values)):
        raise ValueError("Cannot align epoch_data and metadata. "
                         "Mismatch between recording dimension :"
                         f"{set(epoch_data.epoch.values)} in epoch data, vs "
                         f"{set(df[recording_id].values)} in metadata")
    if not set(df[epoch_id]).issuperset(epoch_data.epoch.values):
        raise ValueError(f"DataFrame {epoch_id} is not a superset of epoch_data['epoch']. "
                         "Did not find epochs "
                         f"{set(df[epoch_id]).difference(epoch_data.epoch.values)}"
                        )
    if not set(df[recording_id]).issuperset(epoch_data.recording.values):
        raise ValueError(f"DataFrame {recording_id} is not a superset of epoch_data['recording']. "
                         f"Did not find recodings "
                         f"{set(df[recording_id]).difference(epoch_data.recording.values)}"
                        )

    df = df.rename(columns={recording_id:'recording', epoch_id:'epoch'})
    df['recording'] = df['recording'].astype(str)
    df = df.set_index(['recording','epoch'])
    coords = xr.Dataset.from_dataframe(df)
    coords = coords.set_coords(coords.data_vars)
    for coord in coords.coords:
        if coord not in epoch_data.coords:
            epoch_data = epoch_data.assign_coords({
                coord: coords[coord]
            })
    return epoch_data
