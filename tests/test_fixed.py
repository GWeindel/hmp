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
    hmp_data = hmp.basedata.default(epoch_data, n_comp=3, center=True, duration_id = 'response_time')
    return event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events

def test_fixed_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events = data()
    # Data b is without noise, recovery should be perfect
    data_b = hmp_data.select_coord('b', 'subject')
    pdata_b = PatternData.from_basedata(data_b)
    model = EventModel(n_events=n_events)
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
    expected_lkh = np.array(-8.005786)
    assert np.isclose(lkh_b, expected_lkh, atol=1e-2, rtol=0)

    #locations
    locations = np.zeros(n_events+1, dtype=int)
    locations[1:-1] += 1 #(gamma shift)
    model = EventModel(n_events=n_events, location=locations)
    noloc_loglikelihood, noloc_estimates = model.fit_transform(data_b)
    model = EventModel(n_events=n_events, location=25)
    noloc_loglikelihood, noloc_estimates = model.fit_transform(data_b,)
    expected_lkh = np.array(-7.9525814)
    assert np.isclose(noloc_loglikelihood, expected_lkh, atol=1e-2, rtol=0)

    # testing recovery of attributes
    model.xrlikelihoods
    model.xrchannel_pars
    model.xrtime_pars
    model.xrtime_pars_dev
    model.xrtraces
    estimates_b.sfreq

def test_fixed_short():
    """ test very short latencies """
    event_d, epoch_data, positions, sfreq, n_events = init_data_short()
    hmp_data = hmp.basedata.default(epoch_data, n_comp=.999)
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
    
    hmp_data_a = hmp_data.select_coord('a', 'subject')
    hmp_data_b = hmp_data.select_coord('b', 'subject')

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

def test_grouping_absent_group():
    """A declared group without any trial should not shift the results of the other groups."""
    _, _, _, hmp_data, _, _, n_events = data()
    pdata = PatternData.from_basedata(hmp_data)
    n_trials = len(pdata.durations)
    n_dims = pdata.cross_corr.shape[1]

    def grouping_model(groups_declared):
        # Every group gets its own parameters so that groups cannot influence each other
        channel_map = np.array([np.arange(n_events) + g * n_events for g in groups_declared])
        time_map = np.array([np.arange(n_events + 1) + g * (n_events + 1) \
            for g in groups_declared])
        model = EventModel(n_events=n_events, time_map=time_map, channel_map=channel_map,
            grouping_dict={'condition': [str(g) for g in groups_declared]})
        model.n_dims = n_dims
        return model

    # Same two sets of trials, coded 0 and 2 with the declared group 1 left without any trial,
    # and coded 0 and 1 with only the two groups that do occur declared
    groups = np.where(np.arange(n_trials) < n_trials // 2, 0, 2).astype(np.int8)
    reference_groups = np.where(np.arange(n_trials) < n_trials // 2, 0, 1).astype(np.int8)
    model = grouping_model([0, 1, 2])
    reference_model = grouping_model([0, 2])

    # Distinct starting points per group, so that swapping two groups changes the estimates
    mean_stage = pdata.durations.values.mean() / (n_events + 1)
    channel_pars = np.zeros((3, n_events, n_dims))
    time_pars = np.array([np.tile([model.distribution.shape,
        model.distribution.mean_to_scale(mean_stage * scaling)], (n_events + 1, 1))
        for scaling in [.8, 1., 1.2]])

    lkh, group_channel_pars, group_time_pars, _, traces_group, _ = \
        model.EM(pdata, channel_pars, time_pars, groups)
    reference_lkh, reference_channel_pars, reference_time_pars, _, _, _ = \
        reference_model.EM(pdata, channel_pars[[0, 2]], time_pars[[0, 2]], reference_groups)

    assert np.isclose(lkh, reference_lkh)
    assert np.allclose(group_channel_pars[[0, 2]], reference_channel_pars)
    assert np.allclose(group_time_pars[[0, 2]], reference_time_pars)
    # Group likelihoods are reported for every declared group, nan for the one without trials
    assert traces_group.shape[1] == 3
    assert np.isnan(traces_group[:, 1]).all()

def test_starting_points():
    _, _, epoch_data, hmp_data, positions, sfreq, n_events = data()
    pattern = HalfSine()
    pdata = PatternData.from_basedata(hmp_data, pattern=pattern)
    # Testing starting points
    model_sp = EventModel(pattern=pattern, n_events=n_events, starting_points=2, max_duration=1000)
    model_sp.fit(pdata, verbose=True, cpus=2)
