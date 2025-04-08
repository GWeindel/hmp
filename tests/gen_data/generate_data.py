import os
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
        sfreq=sfreq, times=times_a, noise=True, seed=1, path='dataset_a/')
    raw_b, event_b = simulations.simulate(sources, n_trials, cpus, 'dataset_b_raw', overwrite=True,
        sfreq=sfreq, times=times_b, noise=False, seed=1, path='dataset_b/')

if __name__ == "__main__":
    create_data()