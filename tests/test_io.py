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
from hmp import preprocessors
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
    centering_id = {'stimulus':1}#trigger 1 = stimulus
    response_id = {'response':5}
    raws = [DATA_DIR_A / 'dataset_a_raw_raw.fif', DATA_DIR_B / 'dataset_b_raw_raw.fif']
    event_files = [DATA_DIR_A / 'dataset_a_raw_raw_generating_events.npy',
                   DATA_DIR_B / 'dataset_b_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_a = events[0]
    event_b = events[1]
    # Data reading
    preprocessing_kwargs = dict(sfreq = sfreq)
    montage = simulations.sim_info().get_montage()
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, 
                response_id=response_id, events_provided=events,
                subj_name=['a','b'], montage=montage,
                preprocessing_kwargs=preprocessing_kwargs)
    epoch_data = epoch_data.assign_coords({'condition': ('recording', epoch_data.subject.data)})
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
    centering_id = {'stimulus':1}#trigger 1 = stimulus
    response_id = {'response':5}
    raws = [DATA_DIR_C / 'dataset_c_raw_raw.fif']
    event_files = [DATA_DIR_C / 'dataset_c_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_c = events[0]
    # Data reading
    preprocessing_kwargs = dict(sfreq = sfreq)
    montage = simulations.sim_info().get_montage()
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, 
                response_id=response_id, events_provided=events,
                subj_name=['c'], montage=montage,
                preprocessing_kwargs=preprocessing_kwargs)
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, response_id=response_id,
            events_provided=events, verbose=True, subj_name=['c'])
    epoch_data = epoch_data.assign_coords({'condition': ('recording', epoch_data.subject.data)})
    return event_c, epoch_data, info, sfreq, n_events

def init_data_short():
    # events separated by a very short interval to trigger invalid LL
    """ Initialize all data and model related info."""
    sfreq = 100
    n_events = 3
    events = []
    centering_id = {'stimulus':1}#trigger 1 = stimulus
    response_id = {'response':5}
    raws = [DATA_DIR_D / 'dataset_d_raw_raw.fif']
    event_files = [DATA_DIR_D / 'dataset_d_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_d = events[0]
    # Data reading
    preprocessing_kwargs = dict(sfreq = sfreq)
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, response_id=response_id,
            events_provided=events, verbose=True, subj_name=['d'],
            preprocessing_kwargs=preprocessing_kwargs)
    epoch_data = epoch_data.assign_coords({'condition': ('recording', epoch_data.recording.data)})
    # subsample channels for speed
    epoch_data = epoch_data.sel(channel=epoch_data.channel[::10])
    positions = simulations.positions()[::10]
    return event_d, epoch_data, positions, sfreq, n_events

#### SAVE AS BIDS READ WITH BIDS AND CHECK IT:S THE SAME
#### Same for EPOCH

def test_bids():
    centering_id = {"stimulus":1}
    response_id = {"response":5}
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

    epoching = dict()
    
    # Reading the data
    epoch_data, info = io.read_bids_raw(
        bids_kwargs=bids_kwargs,
        epoching_kwargs=epoching,
        montage=montage,
        preprocessing_kwargs=preprocessing,
        centering_id = centering_id,
        response_id = response_id,
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
    preprocessing = dict(sfreq = 100)
    subj_files = [os.path.join(epoch_data_path, f) for f in os.listdir(epoch_data_path) if f.endswith('.fif')]  # Create a list of files with full paths
    epoch_data, info = io.read_mne_epochs(subj_files,
                                preprocessing_kwargs=preprocessing,
                                montage='biosemi64',
                                verbose=True)


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
    
#     # Which triggers to center the epochs on
#     centering_id = {
#            "compatible/left" : 11,
#     	   "compatible/right" : 12,
#     	   "incompatible/left" : 21,
#     	   "incompatible/right" : 22,
#     }
    
#     # Which triggers indicate the response/the end of the duration to be modelled
#     response_id = {
#     	   "left/compatible/correct":111,
#     	   "left/compatible/incorrect":112,
#     	   "left/incompatible/correct":121,
#     	   "left/incompatible/incorrect":122,
#     	   "right/compatible/incorrect":211,
#     	   "right/compatible/correct":212,
#     	   "right/incompatible/incorrect":221,
#     	   "right/incompatible/correct":222
#     }
    
#     # BIDS argument, root and selection (datatype, session, task, ...)
#     bids_kwargs = dict(
#         root = "../../data/Flanker", #Mandatory
#         datatypes=['eeg'],#Mandatory
#         tasks=['LRP'],#Mandatory
#         # subjects=['001','002'],
#     )
    
#     # Epoching arguments, anything that can be used by mne.Epoch
#     epoching_kwargs = dict(
#         tmin = -0.2,
#         tmax = 2,
#     )
    
#     # Preprocessing arguments, can have low_pass, high_pass, sfreq, etc.
#     preprocessing_kwargs = dict(
#         sfreq = sfreq,
#         # pick_channels = ['C3','C4', 'Fp1', 'Fp2']
#     )
    
#     # Reading the data
#     epoch_data, info = hmp.io.read_bids_raw(
#         bids_kwargs=bids_kwargs,
#         epoching_kwargs=epoching_kwargs,
#         preprocessing_kwargs=preprocessing_kwargs,
#         centering_id = centering_id,
#         response_id = response_id,
#         # preprocessing_fn=preprocessing, # users can now also use their own preprocessing functions
#         montage='biosemi64',
#         cpus=-2, #Can also read data in parallel
#     )


# def test_save_dat():
#     event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
#     hmp_data = preprocessors.ProjPCA(epoch_data, n_comp=2,)
#     data_b = utils.participant_selection(hmp_data.data, 'b')
#     model = EventModel(n_events=n_events)
#     _, estimates = model.fit_transform(data_b)

#     io.save_eventprobs_csv(estimates, 'test')
    
    