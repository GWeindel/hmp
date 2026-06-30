## Importing these packages is specific for this simulation case

import numpy as np

import hmp
from hmp import simulations
from hmp.models import EventModel
from hmp.patterns import HalfSine
from hmp.distributions import Gamma
from hmp.patterndata import PatternData


from test_io import init_data, init_data_large, init_data_short

def data():
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.basedata.BaseData.from_io_all_pca(epoch_data, n_comp=5)
    return event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events

def test_fixed_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events = data()
    # Data b is without noise, recovery should be perfect
    data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    pattern = HalfSine()
    pdata_b = PatternData.from_basedata(data_b, pattern=pattern)
    time_distribution = Gamma()
    model = EventModel(distribution=time_distribution, pattern=pattern, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_b, model, pdata_b)
    # Fixing true parameter in model
    model.time_pars = np.array([true_pars])
    model.channel_pars = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = model.transform(pdata_b)
    true_topos = hmp.utils.event_channels(epoch_data, true_estimates, mean=True)
    true_topos = hmp.utils.event_channels(epoch_data, true_estimates, mean=True)
    #Estimate
    lkh_b, estimates_b = model.fit_transform(pdata_b, verbose=True)
    test_topos = hmp.utils.event_channels(epoch_data, estimates_b, mean=True)
    test_topos = hmp.utils.event_channels(epoch_data, estimates_b, mean=True)
    # Test if events found are classified as true
    assert (np.array(simulations.classification_true(true_topos.squeeze().T,test_topos.squeeze().T)) == np.array(([0,1,2],[0,1,2]))).all()
    # test the difference between electrode values at event times
    assert np.isclose(np.sum(np.abs(true_topos.data - test_topos.data)), 0, atol=1e-4, rtol=0)
    # Test whether likelihood is the expected one
    assert np.isclose(lkh_b, np.array(101.39), atol=1e-2, rtol=0)

    #locations
    locations = np.zeros(n_events+1, dtype=int)
    model = EventModel(n_events=n_events, location=locations)
    noloc_loglikelihood, noloc_estimates = model.fit_transform(data_b)
    model = EventModel(n_events=n_events, location=25)
    noloc_loglikelihood, noloc_estimates = model.fit_transform(data_b,)
    assert np.isclose(noloc_loglikelihood, np.array(101.39), atol=1e-2, rtol=0)

    # testing recovery of attributes
    model.xrlikelihoods
    model.xrchannel_pars
    model.xrtime_pars
    model.xrtime_pars_dev
    model.xrtraces

def test_fixed_csd():
    """ test CSD computation"""
    event_c, epoch_data, info, sfreq, n_events = init_data_large()
    epoch_data, info = hmp.utils.compute_csd(epoch_data, info)

def test_fixed_short():
    """ test very short latencies """
    event_d, epoch_data, positions, sfreq, n_events = init_data_short()
    hmp_data = hmp.basedata.BaseData.from_io(epoch_data, crop=True,
                                                reject = True, apply_variance=True)
    model = EventModel(n_events=n_events)

    #Estimate
    lkh, estimates = model.fit_transform(hmp_data)
    assert ~np.isnan(lkh), "nan likelihood"

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
    pdata = PatternData.from_basedata(hmp_data)
    pdata_a = PatternData.from_basedata(hmp_data_a)
    pdata_b = PatternData.from_basedata(hmp_data_b)

    model = EventModel(n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_a, model, pdata_a)
    # Fixing true parameter in model
    model.time_pars = np.array([true_pars])
    model.channel_pars = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = model.transform(pdata_a)
    true_topos = hmp.utils.event_channels(epoch_data, true_estimates.squeeze(), mean=True)
    
    # Perform a fit on a (should be too noisy)
    lkh_a, estimates_a = model.fit_transform(pdata_a)

    # Fit model on both conditions (noiseless b should help estimate a)
    model = EventModel(n_events=n_events,time_map=time_map, \
                channel_map=channel_map, grouping_dict=grouping_dict)
    model_a = EventModel(n_events=n_events)

    # Fixing true parameter in model
    model.time_pars = np.array([true_pars])
    model.channel_pars = np.array([true_magnitudes])
    model_a.time_pars = np.array([true_pars])
    model_a.channel_pars = np.array([true_magnitudes])

    lkh_comb, estimates_comb = model.fit_transform(pdata)
    lkh_a_group, estimates_a_group = model_a.transform(pdata_a)

    # a_group should be closer to ground truth 
    test_topos_a = hmp.utils.event_channels(epoch_data, estimates_a, mean=True)
    test_topos_a_group = hmp.utils.event_channels(epoch_data, estimates_a_group, mean=True)
    assert np.sum(np.abs(true_topos.data - test_topos_a.data)) > np.sum(np.abs(true_topos.data - test_topos_a_group.data))

    # Testing one event less in one condition
    channel_map = np.array([[0, 0, 0],
                         [0, 0, -1]])
    time_map = np.array([[0, 0, 0, 0],
                         [0, 0, -1, 0],])
    model = EventModel(n_events=n_events,time_map=time_map, \
        channel_map=channel_map, grouping_dict=grouping_dict)

    # Fixing true parameter in model
    model.time_pars = np.array([true_pars])
    model.channel_pars = np.array([true_magnitudes])

    lkh_comb, estimates_comb = model.fit_transform(pdata)
    
def test_starting_points():
    _, _, epoch_data, hmp_data, positions, sfreq, n_events = data()
    pattern = HalfSine()
    pdata = PatternData.from_basedata(hmp_data, pattern=pattern)
    # Testing starting points
    model_sp = EventModel(pattern=pattern, n_events=n_events, starting_points=2, max_duration=1000)
    model_sp.fit(pdata, verbose=True, cpus=2)