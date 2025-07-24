from pathlib import Path

import numpy as np

from hmp import io
from hmp import simulations
from hmp import utils
from hmp.patterns import HalfSine
from hmp.distributions import Gamma
from hmp import preprocessing
from hmp.trialdata import TrialData
from hmp.models import EventModel


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
    epoch_data = io.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq,
            events_provided=events, verbose=True, reference='average', subj_name=['a','b'], tmin=-.01)
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    positions = simulations.positions()
    return event_b, event_a, epoch_data, positions, sfreq, n_events


def test_save_dat():
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = preprocessing.Standard(epoch_data, n_comp=2,)
    data_b = utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=epoch_data.sfreq)
    trial_data_b = TrialData.from_preprocessed(preprocessed=data_b, pattern=event_properties.template)
    model = EventModel(event_properties, n_events=n_events)
    _, estimates = model.fit_transform(trial_data_b)

    io.save_eventprobs_csv(estimates, 'test')
    
    