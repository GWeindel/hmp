## Importing these packages is specific for this simulation case
import os

import numpy as np
import pandas as pd
from scipy.stats import gamma
from pathlib import Path
import gc

import hmp
from hmp import simulations
from hmp.models import FixedEventModel
from hmp.models.base import EventProperties
from hmp.trialdata import TrialData


real_data = os.path.join("tutorials", "sample_data", "eeg", "processed_0022_epo.fif")
real_metadata = os.path.join("tutorials", "sample_data", "eeg", "0022.csv")


def test_fixed():

    # Testing isolated functions in simulations
    assert simulations.simulation_sfreq() == 600.614990234375    
    epoch_data, sim_source_times, info = simulations.demo(1, 1)
    # Data creation/reading
    ## Simulation parameters
    sfreq = 102
    n_events = 3
    n_trials = 2
    cpus=1
    times_a = np.array([[50, 50, 150, 50],
             [50, 100, 100, 50],])
    names = ['bankssts-rh','bankssts-lh','bankssts-rh','bankssts-lh']
    sources = []
    for cur_name in names:
        sources.append([cur_name, 10., 3.1e-8, gamma(2, scale=1)])
    raw_a, event_a,_ = simulations.simulate(sources, n_trials, cpus, 'dataset_a_raw', overwrite=True,
                                          sfreq=sfreq, times=times_a, noise=True, seed=1, save_snr=True)
    means = np.array([50, 100, 100, 50])/2
    sources = []
    for cur_name, cur_mean in zip(names, means):
        sources.append([cur_name, 10., 1, gamma(2, scale=cur_mean)])
    raw_b, event_b = simulations.simulate(sources, n_trials, cpus, 'dataset_b_raw', seed=1, overwrite=True, 
        sfreq=sfreq, verbose=True, proportions=[.99,1,1,1], noise=False)
    
    events = []
    raws = []
    event_id = {'stimulus':1}#trigger 1 = stimulus
    resp_id = {'response':5}
    
    for raw_file, event_file in zip([raw_a, raw_b],
                    [event_a, event_b]):
        events.append(np.load(event_file))
        raws.append(raw_file)
    sfreq = 100
    
    # Data reading
    epoch_data = hmp.utils.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq,
            events_provided=events, verbose=True, reference='average', high_pass=1, low_pass=45,
                subj_idx=['a','b'],pick_channels='eeg', lower_limit_rt=0.01, upper_limit_rt=2 ) 
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    positions = simulations.simulation_positions()

    hmp_data = hmp.utils.transform_data(epoch_data, n_comp=2,)

        
   # Comparing to simulated data, asserting that results are the one simulated
    events_a = np.load(event_a)
    data_a = hmp.utils.participant_selection(hmp_data, 'a')
    event_properties = EventProperties.create_expected(sfreq=data_a.sfreq)
    trial_data = TrialData.from_standard_data(data=data_a, template=event_properties.template)
    init_sim = FixedEventModel(event_properties, n_events=n_events, maximization=False,)
    sim_source_times, true_pars, true_magnitudes, _ = simulations.simulated_times_and_parameters(events_a, init_sim, trial_data)
    true_loglikelihood, true_estimates = init_sim.fit_transform(trial_data, parameters = np.array([true_pars]), magnitudes=np.array([true_magnitudes]),  verbose=True)
    true_topos = hmp.utils.event_topo(epoch_data, true_estimates.squeeze(), mean=True)
    init_sim = FixedEventModel(event_properties, n_events=n_events, maximization=True,)
    likelihood, estimates = init_sim.fit_transform(trial_data, verbose=True)
    test_topos = hmp.utils.event_topo(epoch_data, estimates.squeeze(), mean=True)
    assert (np.array(simulations.classification_true(true_topos,test_topos)) == np.array(([0,1,2],[0,1,2]))).all()
    assert np.sum(np.abs(true_topos.data - test_topos.data)) < 2.65e-05
    assert np.round(likelihood,4) > np.array(-1)
