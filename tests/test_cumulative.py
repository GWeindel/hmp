import numpy as np
import xarray as xr

import hmp
from hmp import simulations
from hmp.models import CumulativeMethod, EventModel
from hmp.patterns import HalfSine
from hmp.distributions import Gamma
from hmp.trialdata import TrialData
from hmp import preprocessing

from test_fixed import init_data


from test_io import init_data

def test_cumulative_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.preprocessing.Standard(epoch_data, n_comp=5, apply_zscore=False).data
    # Data b is without noise, recovery should be perfect
    data_b = hmp.utils.participant_selection(hmp_data, 'b')
    event_properties = HalfSine.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_preprocessed(preprocessed=data_b, pattern=event_properties.template)
    time_distribution = Gamma()

    true_model = EventModel(event_properties, time_distribution, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_b, true_model, trial_data_b)
    # Fixing true parameter in model
    true_model.time_pars = np.array([true_pars])
    true_model.channel_pars = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = true_model.transform(trial_data_b)

    # Cumulative estimation
    model = CumulativeMethod(event_properties, step=25)
    model.fit(trial_data_b)
    estimates = model.transform(trial_data_b)

    # testing if bacward identifies the 3 real events
    assert np.isclose(model.final_model.channel_pars, true_model.channel_pars, atol=1).all()

    # testing recovery of attributes
    assert isinstance(model.xrlikelihoods, xr.DataArray)
    assert isinstance(model.xrchannel_pars, xr.DataArray)
    assert isinstance(model.xrtime_pars, xr.DataArray)
    assert isinstance(model.xrtime_pars_dev, xr.DataArray)
    assert isinstance(model.xrtraces, xr.DataArray)