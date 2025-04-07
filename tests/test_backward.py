## Importing these packages is specific for this simulation case
from pathlib import Path

import numpy as np
import xarray as xr

import hmp
from hmp import simulations
from hmp.models import BackwardEstimationModel, FixedEventModel
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

    true_model = FixedEventModel(event_properties, n_events=n_events)
    # Recover generating parameters
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_b, true_model, trial_data_b)
    # Fixing true parameter in model
    true_model.parameters = np.array([true_pars])
    true_model.magnitudes = np.array([true_magnitudes])
    # Ground truth
    true_loglikelihood, true_estimates = true_model.transform(trial_data_b)

    # Backward estimation
    model = BackwardEstimationModel(event_properties)
    # fit the model
    model.fit(trial_data_b)
    # Transform the data
    estimates = model.transform(trial_data_b)

    # testing if bacward identifies the 3 real events
    assert np.isclose(model.submodels[3].magnitudes, true_model.magnitudes, atol=1).all()

    # testing recovery of attributes
    assert isinstance(model.xrlikelihoods, xr.DataArray)
    assert isinstance(model.xrmags, xr.DataArray)
    assert isinstance(model.xrparams, xr.DataArray)
    assert isinstance(model.xrparam_dev, xr.DataArray)
    assert isinstance(model.xrtraces, xr.DataArray)
