import numpy as np
import xarray as xr

import hmp
from hmp import simulations
from hmp.models import CumulativeMethod, EventModel
from hmp.patterns import HalfSine
from hmp.distributions import Gamma
from hmp.trialdata import TrialData

from test_fixed import init_data_large


from test_io import init_data

def test_cumulative_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_c, epoch_data, positions, sfreq, n_events = init_data_large()
    hmp_data = hmp.transformers.ProjPCA(epoch_data, n_comp=2).data
    # Data b is without noise, recovery should be perfect
    data_c = hmp.utils.participant_selection(hmp_data, 'c')
    event_properties = HalfSine.create_expected(sfreq=data_c.sfreq)
    trial_data_c = TrialData.from_transformer(data_c, pattern=event_properties.template)
    time_distribution = Gamma()

    true_model = EventModel(event_properties, time_distribution, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_c, true_model, trial_data_c)
    # Fixing true parameter in model
    true_model.time_pars = np.array([true_pars])
    true_model.channel_pars = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = true_model.transform(trial_data_c)
    
    #Try k-fold 
    model = CumulativeMethod(event_properties)
    model.fit(trial_data_c, kfold=2)

    # Cumulative estimation
    model = CumulativeMethod(event_properties)
    model.fit(trial_data_c)

    # Testing estimates
    estimates = model.transform(trial_data_c)
    # testing if bacward identifies the 3 real events
    assert np.isclose(model.submodels[-1].channel_pars, true_model.channel_pars, atol=1).all()

    # testing recovery of attributes
    assert isinstance(model.xrlikelihoods, xr.DataArray)
    assert isinstance(model.xrchannel_pars, xr.DataArray)
    assert isinstance(model.xrtime_pars, xr.DataArray)
    assert isinstance(model.xrtime_pars_dev, xr.DataArray)
    assert isinstance(model.xrtraces, xr.DataArray)