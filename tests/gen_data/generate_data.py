import os
import shutil
import numpy as np
from scipy.stats import gamma

from hmp import simulations

def create_data():
    sfreq = 100
    n_events = 3
    # Data creation/reading
    ## Simulation parameters
    n_trials = 2
    cpus=1
    times_a = np.array([[100, 100, 200, 100],
             [100, 100, 200, 100],], dtype='float64')
    times_b = times_a.copy()
    times_b[:,2] *= 2
    print(times_a)
    print(times_b)
    names = ['bankssts-rh','bankssts-lh','caudalanteriorcingulate-rh','bankssts-lh']
    sources = []
    for cur_name in names:
        sources.append([cur_name, 10., 2.5e-8, gamma(2, scale=1)])
    raw_a, event_a = simulations.simulate(sources, n_trials, cpus, 'dataset_a_raw', overwrite=True,
        sfreq=sfreq, times=times_a, noise=True, seed=1, path='dataset_a')
    raw_b, event_b = simulations.simulate(sources, n_trials, cpus, 'dataset_b_raw', overwrite=True,
        sfreq=sfreq, times=times_b, noise=False, seed=1, path='dataset_b')
    
    # For SVD/CSD test 
    n_trials = 62 #cannot easily subset electrodes from info so increase n_trials for SVD
    sources = []
    for cur_name in names:
        sources.append([cur_name, 10., 2.5e-8, gamma(2, scale=50)])
    raw_c, event_c = simulations.simulate(sources, n_trials, cpus, 'dataset_c_raw', overwrite=True,
        sfreq=sfreq, noise=False, seed=1, path='dataset_c')

    # For corner time case, small time distance between events
    # Will drive EM to too short values

    n_trials = 20
    sources = []
    scales = [50,15,50,50]
    for cur_name, scale in zip(names,scales):
        sources.append([cur_name, 15., 2.5e-5, gamma(2, scale=scale)])
    raw_d, event_d = simulations.simulate(sources, n_trials, cpus, 'dataset_d_raw', overwrite=True,
         sfreq=sfreq, noise=True, seed=1, path='dataset_d')

    # For BIDS
    os.makedirs("pseudo_BIDS/sub-a/eeg", exist_ok=True)
    shutil.move(
        "dataset_a/dataset_a_raw_raw.fif",
        "pseudo_BIDS/sub-a/eeg/sub-a_task-X_eeg.fif",
    )

if __name__ == "__main__":
    create_data()
