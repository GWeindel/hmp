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
        sources.append([cur_name, 10., 2.5e-8, None])
    files_a = simulations.simulate(sources, n_trials, cpus, 'dataset_a_raw', overwrite=True,
        sfreq=sfreq, times=times_a, noise=True, seed=1, path='dataset_a')
    files_b = simulations.simulate(sources, n_trials, cpus, 'dataset_b_raw', overwrite=True,
        sfreq=sfreq, times=times_b, noise=False, seed=1, path='dataset_b')
    raw_a, event_a = [files_a[0][0]], files_a[0][1]
    raw_b, event_b = [files_b[0][0]], files_b[0][1]
    # For SVD/CSD test 
    n_trials = 62 #cannot easily subset electrodes from info so increase n_trials for SVD
    sources = []
    for cur_name in names:
        sources.append([cur_name, 10., 2.5e-8, gamma(2, scale=50)])
    files_c = simulations.simulate(sources, n_trials, cpus, 'dataset_c_raw', overwrite=True,
        sfreq=sfreq, noise=False, seed=1, path='dataset_c')
    raw_c, event_c = [files_c[0][0]], files_c[0][1]
    # For corner time case, small time distance between events
    # Will drive EM to too short values

    n_trials = 20
    sources = []
    scales = [50,15,50,50]
    for cur_name, scale in zip(names,scales):
        sources.append([cur_name, 15., 2.5e-5, gamma(2, scale=scale)])
    files_d = simulations.simulate(sources, n_trials, cpus, 'dataset_d_raw', overwrite=True,
         sfreq=sfreq, noise=True, seed=1, path='dataset_d')
    raw_d, event_d = [files_d[0][0]], files_d[0][1]
    # For BIDS
    os.makedirs("pseudo_BIDS/sub-a/eeg", exist_ok=True)
    shutil.copy(
        "dataset_a/dataset_a_raw_raw.fif",
        "pseudo_BIDS/sub-a/eeg/sub-a_task-X_eeg.fif",
    )

if __name__ == "__main__":
    create_data()
