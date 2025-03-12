import os
import numpy as np
from scipy.stats import gamma

from hmp import simulations

def create_data():
    sfreq = 100
    n_events = 3 
    # Testing isolated functions in simulations
    assert simulations.simulation_sfreq() == 600.614990234375    
    epoch_data, sim_source_times, info = simulations.demo(1, 1)
    # Data creation/reading
    ## Simulation parameters
    n_trials = 2
    cpus=1
    times_a = np.array([[50, 50, 150, 50],
             [50, 100, 100, 50],])
    names = ['bankssts-rh','bankssts-lh','bankssts-rh','bankssts-lh']
    sources = []
    for cur_name in names:
        sources.append([cur_name, 10., 3.1e-8, gamma(2, scale=1)])
    raw_a, event_a,_ = simulations.simulate(sources, n_trials, cpus, 'dataset_a_raw', overwrite=True,
                                          sfreq=sfreq, times=times_a, noise=True, seed=1, save_snr=True, path='dataset_a/')
    means = np.array([50, 100, 100, 50])/2
    sources = []
    for cur_name, cur_mean in zip(names, means):
        sources.append([cur_name, 10., 1, gamma(2, scale=cur_mean)])
    raw_b, event_b = simulations.simulate(sources, n_trials, cpus, 'dataset_b_raw', seed=1, overwrite=True, 
        sfreq=sfreq, verbose=True, proportions=[.99,1,1,1], noise=False, path='dataset_b/')

if __name__ == "__main__":
    create_data()