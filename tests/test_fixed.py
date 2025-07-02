## Importing these packages is specific for this simulation case

import numpy as np

import hmp
from hmp import simulations
from hmp.models import EventModel
from hmp.patterns import HalfSine
from hmp.distributions import Gamma
from hmp.trialdata import TrialData
from hmp import preprocessing


from test_io import init_data

def data():
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.preprocessing.Standard(epoch_data, n_comp=2,)
    return event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events

def test_fixed_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events = data()
    # Data b is without noise, recovery should be perfect
    data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_preprocessed(preprocessed=data_b, pattern=event_properties.template)
    time_distribution = Gamma()
    model = EventModel(event_properties, time_distribution, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_b, model, trial_data_b)
    # Fixing true parameter in model
    model.time_pars = np.array([true_pars])
    model.channel_pars = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = model.transform(trial_data_b)
    true_topos = hmp.utils.event_channels(epoch_data, true_estimates, mean=True)
    true_topos = hmp.utils.event_channels(epoch_data, true_estimates, mean=True)
    #Estimate
    lkh_b, estimates_b = model.fit_transform(trial_data_b, verbose=True)
    test_topos = hmp.utils.event_channels(epoch_data, estimates_b, mean=True)
    test_topos = hmp.utils.event_channels(epoch_data, estimates_b, mean=True)
    
    # Test if events found are classified as true
    assert (np.array(simulations.classification_true(true_topos.squeeze().T,test_topos.squeeze().T)) == np.array(([0,1,2],[0,1,2]))).all()
    # test the difference between electrode values at event times
    assert np.isclose(np.sum(np.abs(true_topos.data - test_topos.data)), 0, atol=1e-4, rtol=0)
    # Test whether likelihood is the expected one
    assert np.isclose(lkh_b, np.array(30.57338794), atol=1e-4, rtol=0)
    
    # testing recovery of attributes
    model.xrlikelihoods
    model.xrchannel_pars
    model.xrtime_pars
    model.xrtime_pars_dev
    model.xrtraces

def test_fixed_grouping():
    _, event_a, epoch_data, hmp_data, positions, sfreq, n_events = data()

    # testing grouping model
    channel_map = np.array([[0, 0, 0],
                         [0, 0, 0]])
    time_map = np.array([[0, 0, 0, 0],
                         [0, 0, 1, 0],])
    grouping_dict = {'condition': ['a', 'b']}
    
    hmp_data_a = hmp.utils.participant_selection(hmp_data.data, 'a')
    hmp_data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=epoch_data.sfreq)
    trial_data = TrialData.from_preprocessed(preprocessed=hmp_data, pattern=event_properties.template)
    trial_data_a = TrialData.from_preprocessed(preprocessed=hmp_data_a, pattern=event_properties.template)
    trial_data_b = TrialData.from_preprocessed(preprocessed=hmp_data_b, pattern=event_properties.template)

    model = EventModel(event_properties, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_a, model, trial_data_a)
    # Fixing true parameter in model
    model.time_pars = np.array([true_pars])
    model.channel_pars = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = model.transform(trial_data_a)
    true_topos = hmp.utils.event_channels(epoch_data, true_estimates.squeeze(), mean=True)
    
    # Perform a fit on a (should be too noisy)
    lkh_a, estimates_a = model.fit_transform(trial_data_a)

    # Fit model on both conditions (noiseless b should help estimate a)
    trial_data = TrialData.from_preprocessed(preprocessed=hmp_data, pattern=event_properties.template)
    lkh_comb, estimates_comb = model.fit_transform(trial_data, time_map=time_map, channel_map=channel_map, grouping_dict=grouping_dict)
    lkh_a_group, estimates_a_group = model.transform(trial_data_a)

    # a_group should be closer to ground truth 
    test_topos_a = hmp.utils.event_channels(epoch_data, estimates_a, mean=True)
    test_topos_a_group = hmp.utils.event_channels(epoch_data, estimates_a_group, mean=True)
    assert np.sum(np.abs(true_topos.data - test_topos_a.data)) > np.sum(np.abs(true_topos.data - test_topos_a_group.data))

    # Testing one event less in one condition
    channel_map = np.array([[0, 0, 0],
                         [0, 0, -1]])
    time_map = np.array([[0, 0, 0, 0],
                         [0, 0, -1, 0],])
    lkh_comb, estimates_comb = model.fit_transform(trial_data, time_map=time_map, channel_map=channel_map, grouping_dict=grouping_dict)
    
def test_starting_points():
    _, _, epoch_data, hmp_data, positions, sfreq, n_events = data()
    event_properties = HalfSine.create_expected(sfreq=epoch_data.sfreq)
    trial_data = TrialData.from_preprocessed(preprocessed=hmp_data, pattern=event_properties.template)
    # Testing starting points
    model_sp = EventModel(event_properties, n_events=n_events, starting_points=2, max_scale=21)
    model_sp.fit(trial_data, verbose=True)