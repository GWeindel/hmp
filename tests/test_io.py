from pathlib import Path

import numpy as np
import mne_bids
import shutil
import platform
import os
import requests
from zipfile import ZipFile


import hmp
from hmp import io
from hmp import simulations
from hmp import utils
from hmp.patterndata import PatternData
from hmp.models import EventModel


DATA_DIR = Path("tests", "gen_data")
DATA_DIR_A = DATA_DIR / "dataset_a"
DATA_DIR_B = DATA_DIR / "dataset_b"
DATA_DIR_C = DATA_DIR / "dataset_c"
DATA_DIR_D = DATA_DIR / "dataset_d"

sfreq = 100
n_events = 3
centering_id = {'stimulus/0':1}#trigger 1 = stimulus
event_id = {'response/0':5}

def preprocessing_fn(data, montage, events, 
                   verbose, preprocessing_kwargs):
    return data, events

def init_data():
    # Also tests data_format 'raw'
    """ Initialize all data and model related info."""
    events = []
    raws = [DATA_DIR_A / 'dataset_a_raw_raw.fif', DATA_DIR_B / 'dataset_b_raw_raw.fif']
    event_files = [DATA_DIR_A / 'dataset_a_raw_raw_generating_events.npy',
                   DATA_DIR_B / 'dataset_b_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_a = events[0]
    event_b = events[1]
    # Data reading
    montage = simulations.sim_info().get_montage()
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, 
                event_id=event_id, events_provided=events,
                subj_name=['a','b'], montage=montage,
                cpus=2)
    epoch_data = epoch_data.assign_coords({'condition': ('recording', epoch_data.subject.data)})
    # subsample channels for speed
    epoch_data = epoch_data.sel(channel=epoch_data.channel[::3])
    positions = simulations.positions()[::3]
    return event_b, event_a, epoch_data, positions, sfreq, n_events

def init_data_large():
    # all electrode and more trials
    """ Initialize all data and model related info."""
    events = []
    raws = [DATA_DIR_C / 'dataset_c_raw_raw.fif']
    event_files = [DATA_DIR_C / 'dataset_c_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_c = events[0]
    # Data reading
    montage = simulations.sim_info().get_montage()
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, 
                event_id=event_id, events_provided=events,
                subj_name=['c'], montage=montage, 
                preprocessing_fn=preprocessing_fn)
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, event_id=event_id,
            events_provided=events, verbose=True, subj_name=['c'])
    epoch_data = epoch_data.assign_coords({'condition': ('recording', epoch_data.subject.data)})
    return event_c, epoch_data, info, sfreq, n_events

def init_data_short():
    # events separated by a very short interval to trigger invalid LL
    """ Initialize all data and model related info."""
    events = []
    raws = [DATA_DIR_D / 'dataset_d_raw_raw.fif']
    event_files = [DATA_DIR_D / 'dataset_d_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_d = events[0]
    # Data reading
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, event_id=event_id,
            events_provided=events, verbose=True, subj_name=['d'], 
            preprocessing_fn=preprocessing_fn)
    epoch_data = epoch_data.assign_coords({'condition': ('recording', epoch_data.recording.data)})
    # subsample channels for speed
    epoch_data = epoch_data.sel(channel=epoch_data.channel[::10])
    positions = simulations.positions()[::10]
    return event_d, epoch_data, positions, sfreq, n_events

def test_bids():
    bids_path = DATA_DIR / "pseudo_BIDS"
    bids_kwargs = dict(
        root = bids_path, #Mandatory path to the data
        datatypes=['eeg'],#Mandatory, type of data we want to model, could be `meg`
        tasks=["X"],# Here the task we want to analyze 
    )
    montage = simulations.sim_info().get_montage()
    preprocessing = dict(sfreq=120, #Test upsampling
                         lowpass=40,
                         highpass=.1,
                         reference='average',
                         pick_channels='eeg' 
                        )

    epoching = dict(tmin=-.1, tmax=2)
    
    # Reading the data
    epoch_data, info = io.read_bids_raw(
        bids_kwargs=bids_kwargs,
        epoching_kwargs=epoching,
        montage=montage,
        preprocessing_kwargs=preprocessing,
        preprocessing_fn=preprocessing_fn,
        centering_id = centering_id,
        event_id = event_id,
        cpus=1,
    )
    epoch_data, info = io.read_bids_raw(
        bids_kwargs=bids_kwargs,
        epoching_kwargs=epoching,
        montage=montage,
        preprocessing_kwargs=preprocessing,
        centering_id = centering_id,
        event_id = event_id,
        cpus=2,
    )

def test_epochs():
    # Declaring path where the EEG data will be stored
    epoch_data_path = os.path.join('sample_data', 'eeg')
    os.makedirs(epoch_data_path, exist_ok=True)
    
    # URLs of the first participant
    file_urls = [
        "https://osf.io/download/67cffa85f67af67e7a92f0a6/",
    ]
    
    # Download and save each file if not already in folder
    for i, url in enumerate(file_urls, start=1):
        file_path = os.path.join(epoch_data_path, f'S{i}_epo.fif')
        if not os.path.exists(file_path):
            response = requests.get(url)
            with open(file_path, 'wb') as f:
                f.write(response.content)
    subj_files = [os.path.join(epoch_data_path, f) for f in os.listdir(epoch_data_path) if f.endswith('.fif')]  # Create a list of files with full paths
    epoch_data, info = io.read_mne_epochs(subj_files,
                                montage='biosemi64',
                                verbose=True)
    epoch_data, info = io.read_mne_epochs(subj_files,
                                montage='biosemi64',
                                verbose=True, cpus=2, preprocessing_fn=preprocessing_fn)
    epoch_data.drop_vars(['stim'])
    metadata = epoch_data['RT'].to_dataframe().reset_index().iloc[:,:4]
    metadata['recording'] = 'S1'
    io.add_metadata(epoch_data,metadata)


def test_fixed_csd():
    """ test CSD computation"""
    event_c, epoch_data, info, sfreq, n_events = init_data_large()
    epoch_data, info = io.preprocessing.compute_csd(epoch_data, info)
