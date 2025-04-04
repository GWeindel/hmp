## Importing these packages is specific for this simulation case
import os

import numpy as np
import pandas as pd
from pathlib import Path
import gc
from pytest import mark

import hmp
from hmp import simulations
from hmp.models import FixedEventModel
from hmp.models.base import EventProperties
from hmp.trialdata import TrialData

DATA_DIR = Path("tests", "gen_data")
DATA_DIR_A = DATA_DIR / "dataset_a"
DATA_DIR_B = DATA_DIR / "dataset_b"

def init_data():
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
    epoch_data = hmp.utils.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq,
            events_provided=events, verbose=True, reference='average', subj_idx=['a','b'], tmin=-.01)
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    positions = simulations.simulation_positions()
    hmp_data = hmp.utils.transform_data(epoch_data, n_comp=2,)
    return event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events

def test_fixed_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events = init_data()
    # Data b is without noise, recovery should be perfect
    data_b = hmp.utils.participant_selection(hmp_data, 'b')
    event_properties = EventProperties.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_standard_data(data=data_b, template=event_properties.template)
    model = FixedEventModel(event_properties, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_b, model, trial_data_b)
    # Fixing true parameter in model
    model.parameters = np.array([true_pars])
    model.magnitudes = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = model.transform(trial_data_b)
    true_topos = hmp.utils.event_topo(epoch_data, true_estimates, mean=True)
    true_topos = hmp.utils.event_topo(epoch_data, true_estimates, mean=True)
    #Estimate
    lkh_b, estimates_b = model.fit_transform(trial_data_b, verbose=True)
    test_topos = hmp.utils.event_topo(epoch_data, estimates_b, mean=True)
    test_topos = hmp.utils.event_topo(epoch_data, estimates_b, mean=True)
    
    # Test if events found are classified as true
    assert (np.array(simulations.classification_true(true_topos.squeeze().T,test_topos.squeeze().T)) == np.array(([0,1,2],[0,1,2]))).all()
    # test the difference between electrode values at event times
    assert np.isclose(np.sum(np.abs(true_topos.data - test_topos.data)), 0, atol=1e-4, rtol=0)
    # Test whether likelihood is the expected one
    assert np.isclose(lkh_b, np.array(28.13143873), atol=1e-4, rtol=0)
    
    # testing recovery of attributes
    model.xrlikelihoods
    model.xrmags
    model.xrparams
    model.param_dev
    model.xrtraces

def test_fixed_multilevel():
    _, event_a, epoch_data, hmp_data, positions, sfreq, n_events = init_data()

    # testing multilevel model
    mags_map = np.array([[0, 0, 0],
                         [0, 0, 0]])
    pars_map = np.array([[0, 0, 0, 0],
                         [0, 0, 1, 0],])
    level_dict = {'condition': ['a', 'b']}
    
    event_properties = EventProperties.create_expected(sfreq=hmp_data.sfreq)
    hmp_data_a = hmp.utils.participant_selection(hmp_data, 'a')
    hmp_data_b = hmp.utils.participant_selection(hmp_data, 'b')
    trial_data = TrialData.from_standard_data(data=hmp_data, template=event_properties.template)
    trial_data_a = TrialData.from_standard_data(data=hmp_data_a, template=event_properties.template)
    trial_data_b = TrialData.from_standard_data(data=hmp_data_b, template=event_properties.template)

    model = FixedEventModel(event_properties, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_a, model, trial_data_a)
    # Fixing true parameter in model
    model.parameters = np.array([true_pars])
    model.magnitudes = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = model.transform(trial_data_a)
    true_topos = hmp.utils.event_topo(epoch_data, true_estimates.squeeze(), mean=True)
    
    # Perform a fit on a (should be too noisy)
    lkh_a, estimates_a = model.fit_transform(trial_data_a)

    # Fit model on both conditions (noiseless b should help estimate a)
    trial_data = TrialData.from_standard_data(data=hmp_data, template=event_properties.template)
    lkh_comb, estimates_comb = model.fit_transform(trial_data, pars_map=pars_map, mags_map=mags_map, level_dict=level_dict)
    lkh_a_level, estimates_a_level = model.transform(trial_data_a)

    # a_level should be closer to ground truth 
    test_topos_a = hmp.utils.event_topo(epoch_data, estimates_a, mean=True)
    test_topos_a_level = hmp.utils.event_topo(epoch_data, estimates_a_level, mean=True)
    assert np.sum(np.abs(true_topos.data - test_topos_a.data)) > np.sum(np.abs(true_topos.data - test_topos_a_level.data))

    # Testing one event less in one condition
    mags_map = np.array([[0, 0, 0],
                         [0, 0, -1]])
    pars_map = np.array([[0, 0, 0, 0],
                         [0, 0, -1, 0],])
    lkh_comb, estimates_comb = model.fit_transform(trial_data, pars_map=pars_map, mags_map=mags_map, level_dict=level_dict)
    
def test_starting_points():
    _, _, epoch_data, hmp_data, positions, sfreq, n_events = init_data()
    event_properties = EventProperties.create_expected(sfreq=hmp_data.sfreq)
    trial_data = TrialData.from_standard_data(data=hmp_data, template=event_properties.template)
    # Testing starting points
    model_sp = FixedEventModel(event_properties, n_events=n_events, starting_points=2, max_scale=21)
    model_sp.fit(trial_data, verbose=True)


