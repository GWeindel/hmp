import numpy as np
import xarray as xr

import hmp
from hmp import simulations
from hmp.models import EliminativeMethod, EventModel
from hmp.patterndata import PatternData


from test_io import init_data

def test_backward_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.basedata.BaseData.from_io_all_pca(epoch_data, n_comp=3,).data
    # Data b is without noise, recovery should be perfect
    data_b = hmp.utils.participant_selection(hmp_data, 'b')
    pdata_b = PatternData.from_basedata(data_b)
    true_model = EventModel(n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_b, true_model, pdata_b)
    # Fixing true parameter in model
    true_model.time_pars = np.array([true_pars])
    true_model.channel_pars = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = true_model.transform(pdata_b)

    # Backward estimation
    model = EliminativeMethod()
    # fit the model
    model.fit(pdata_b)
    # Transform the data
    estimates = model.transform(pdata_b)

    # testing if bacward identifies the 3 real events
    assert np.isclose(model.submodels[3].channel_pars, true_model.channel_pars, atol=1).all()

    # testing recovery of attributes
    assert isinstance(model.xrlikelihoods, xr.DataArray)
    assert isinstance(model.xrchannel_pars, xr.DataArray)
    assert isinstance(model.xrtime_pars, xr.DataArray)
    assert isinstance(model.xrtime_pars_dev, xr.DataArray)
    assert isinstance(model.xrtraces, xr.DataArray)