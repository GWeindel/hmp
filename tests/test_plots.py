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
from hmp.visu import plot_topo_timecourse

from test_fixed import init_data


DATA_DIR = Path("tests", "gen_data")
DATA_DIR_A = DATA_DIR / "dataset_a"
DATA_DIR_B = DATA_DIR / "dataset_b"

def test_plot():
    _, _, epoch_data, hmp_data, positions, sfreq, n_events = init_data()
    
    # Testing one event less in one condition
    mags_map = np.array([[0, 0, -1],
                         [0, 0, 0]])
    pars_map = np.array([[0, 0, -1, 0],
                         [0, 0, 0, 0],])
    level_dict = {'condition': ['a', 'b']}
    
    event_properties = EventProperties.create_expected(sfreq=hmp_data.sfreq)
    hmp_data_b = hmp.utils.participant_selection(hmp_data, 'a')
    trial_data = TrialData.from_standard_data(data=hmp_data, template=event_properties.template)
    trial_data_b = TrialData.from_standard_data(data=hmp_data_b, template=event_properties.template)

    model = FixedEventModel(event_properties, n_events=n_events)
    
    # Perform a fit on a (should be too noisy)
    lkh_b, estimates_b = model.fit_transform(trial_data_b)

    # Fit model on both conditions (noiseless b should help estimate a)
    trial_data = TrialData.from_standard_data(data=hmp_data, template=event_properties.template)
    lkh_comb, estimates_comb = model.fit_transform(trial_data, pars_map=pars_map, mags_map=mags_map, level_dict=level_dict)
    lkh_b_level, estimates_b_level = model.transform(trial_data_b)

    plot_topo_timecourse(epoch_data, estimates_comb, positions, as_time=True, colorbar=False, )
    plot_topo_timecourse(epoch_data, estimates_b, positions, as_time=True, 
                       max_time=500, colorbar=False, )
