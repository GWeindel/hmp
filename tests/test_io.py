from pathlib import Path

import numpy as np
import mne_bids
import shutil
import platform
import os
import requests
from zipfile import ZipFile


from hmp import io
from hmp import simulations
from hmp import utils
from hmp import transformers
from hmp.patterndata import PatternData
from hmp.models import EventModel


DATA_DIR = Path("tests", "gen_data")
DATA_DIR_A = DATA_DIR / "dataset_a"
DATA_DIR_B = DATA_DIR / "dataset_b"
DATA_DIR_C = DATA_DIR / "dataset_c"
DATA_DIR_D = DATA_DIR / "dataset_d"


def init_data():
    # Also tests data_format 'raw'
    """ Initialize all data and model related info."""
    sfreq = 100
    n_events = 3
    events = []
    event_id = {'stimulus':1}#trigger 1 = stimulus
    resp_id = {'response':5}
    raws = [DATA_DIR_A / 'dataset_a_raw_raw.fif', DATA_DIR_B / 'dataset_b_raw_raw.fif']
    event_files = [DATA_DIR_A / 'dataset_a_raw_raw_generating_events.npy',
                   DATA_DIR_B / 'dataset_b_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_a = events[0]
    event_b = events[1]
    # Data reading
    epoch_data = io.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq, pick_channels='eeg',
            events_provided=events, verbose=True, subj_name=['a','b'], tmin=-.01, tmax=1)
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    # subsample channels for speed
    epoch_data = epoch_data.sel(channel=epoch_data.channel[::3])
    positions = simulations.positions()[::3]
    return event_b, event_a, epoch_data, positions, sfreq, n_events

def init_data_large():
    # all electrode and more trials
    """ Initialize all data and model related info."""
    sfreq = 100
    n_events = 3
    events = []
    event_id = {'stimulus':1}#trigger 1 = stimulus
    resp_id = {'response':5}
    raws = [DATA_DIR_C / 'dataset_c_raw_raw.fif']
    event_files = [DATA_DIR_C / 'dataset_c_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_c = events[0]
    # Data reading
    epoch_data = io.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq, pick_channels='eeg',
            events_provided=events, verbose=True, subj_name=['c'], tmin=-.01, tmax=1)
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    info = simulations.sim_info()
    return event_c, epoch_data, info, sfreq, n_events

def init_data_short():
    # events separated by a very short interval to trigger invalid LL
    """ Initialize all data and model related info."""
    sfreq = 100
    n_events = 3
    events = []
    event_id = {'stimulus':1}#trigger 1 = stimulus
    resp_id = {'response':5}
    raws = [DATA_DIR_D / 'dataset_d_raw_raw.fif']
    event_files = [DATA_DIR_D / 'dataset_d_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_d = events[0]
    # Data reading
    epoch_data = io.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq, pick_channels='eeg',
            events_provided=events, verbose=True, subj_name=['d'], tmin=-.01, tmax=1)
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    # subsample channels for speed
    epoch_data = epoch_data.sel(channel=epoch_data.channel[::10])
    positions = simulations.positions()[::10]
    return event_d, epoch_data, positions, sfreq, n_events

    
def test_epochs():
    # Declaring path where the EEG data will be stored
    epoch_data_path = os.path.join('sample_data', 'eeg')
    os.makedirs(epoch_data_path, exist_ok=True)
    
    # URLs of the first two participants
    file_urls = [
        "https://osf.io/download/67cffa85f67af67e7a92f0a6/",
    ]
    
    # Download and save each file if not already in folder
    for i, url in enumerate(file_urls, start=1):
        file_path = os.path.join(epoch_data_path, f'participant{i}_epo.fif')
        if not os.path.exists(file_path):
            response = requests.get(url)
            with open(file_path, 'wb') as f:
                f.write(response.content)
    sfreq = 100 #at what sampling rate we want the data, downsampling to 100Hz is computationally less intensive for hmp instances
    
    # Recovering individual files and participant names
    subj_files = [os.path.join(epoch_data_path, f) for f in os.listdir(epoch_data_path) if f.endswith('.fif')]  # Create a list of files with full paths
    subj_names = [os.path.splitext(f)[0] for f in os.listdir(epoch_data_path) if f.endswith('.fif')]  # Extract subject names based on file names
    
    # Then we read the data (see more in Tutorial 1)
    epoch_data = io.read_mne_data(subj_files, sfreq=sfreq, data_format='epochs',
                            verbose=False, subj_name=subj_names)#Turning verbose off for the documentation but it is recommended to leave it on as some output from MNE might be useful

# def test_bids():
#     # Testing on small bids dataset (1.8 GB)
#     erp_core_url = "https://osf.io/download/3zk6n/"
#     zip_path = "ERP_CORE_P3.zip"
#     extract_dir = "./"
    
#     if not os.path.exists(zip_path):
#         print("Downloading ERP CORE dataset...")
#         r = requests.get(erp_core_url, stream=True)
#         with open(zip_path, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#     print("Extracting ERP CORE dataset...")
#     with ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extract_dir)
    
#     sfreq = 250 
#     tmin, tmax = -.2, .8
#     epoch_data = io.read_mne_data([], 
#                                       data_format='bids',
#                                       tmin=tmin, tmax=tmax, 
#                                       sfreq=sfreq,
#                                       bids_parameters={
#                                           'bids_root': 'ERP_CORE',
#                                           'task': 'P3',
#                                           'datatype': 'eeg',
#                                           'session': 'P3'
#                                       },
#                                       reference='average',
#                                       verbose=False
#                                       )
    

def test_save_dat():
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = transformers.ProjPCA(epoch_data, n_comp=2,)
    data_b = utils.participant_selection(hmp_data.data, 'b')
    model = EventModel(n_events=n_events)
    _, estimates = model.fit_transform(data_b)

    io.save_eventprobs_csv(estimates, 'test')
    
    