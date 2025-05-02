import numpy as np
import xarray as xr

import hmp
from hmp import simulations
from hmp.models import CumulativeEstimationModel, FixedEventModel
from hmp.models.base import EventProperties
from hmp.trialdata import TrialData

from test_io import init_data

def test_cumulative_simple():
    """ test a simple fit_transform on perfect data and compare to ground truth."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.utils.transform_data(epoch_data, n_comp=5, apply_zscore=False)
    # Data b is without noise, recovery should be perfect
    data_b = hmp.utils.participant_selection(hmp_data, 'b')
    event_properties = EventProperties.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_standard_data(data=data_b, template=event_properties.template)

    true_model = FixedEventModel(event_properties, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_b, true_model, trial_data_b)
    # Fixing true parameter in model
    true_model.parameters = np.array([true_pars])
    true_model.magnitudes = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = true_model.transform(trial_data_b)

    # Cumulative estimation
    model = CumulativeEstimationModel(event_properties, step=25)
    model.fit(trial_data_b)
    estimates = model.transform(trial_data_b)

    # testing if bacward identifies the 3 real events
    assert np.isclose(model.fitted_model.magnitudes, true_model.magnitudes, atol=1).all()

    # testing recovery of attributes
    assert isinstance(model.xrlikelihoods, xr.DataArray)
    assert isinstance(model.xrmags, xr.DataArray)
    assert isinstance(model.xrparams, xr.DataArray)
    assert isinstance(model.xrparam_dev, xr.DataArray)
    assert isinstance(model.xrtraces, xr.DataArray)
