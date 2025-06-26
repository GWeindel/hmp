import numpy as np
import hmp
from hmp.models import EventModel
from hmp.patterns import HalfSine
from hmp.trialdata import TrialData
from hmp.visu import plot_topo_timecourse
from hmp import preprocessing


from test_io import init_data


def test_plot():
    _, _, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = preprocessing.Standard(epoch_data, n_comp=2,)
    # Testing one event less in one condition
    channel_map = np.array([[0, 0, -1],
                         [0, 0, 0]])
    time_map = np.array([[0, 0, -1, 0],
                         [0, 0, 0, 0],])
    group_dict = {'condition': ['a', 'b']}
    
    event_properties = HalfSine.create_expected(sfreq=epoch_data.sfreq)
    hmp_data_b = hmp.utils.participant_selection(hmp_data.data, 'a')
    trial_data = TrialData.from_preprocessed(preprocessed=hmp_data.data, pattern=event_properties.template)
    trial_data_b = TrialData.from_preprocessed(preprocessed=hmp_data_b, pattern=event_properties.template)

    model = EventModel(event_properties, n_events=n_events)
    
    # Perform a fit on a (should be too noisy)
    lkh_b, estimates_b = model.fit_transform(trial_data_b)

    # Fit model on both conditions (noiseless b should help estimate a)

    trial_data = TrialData.from_preprocessed(preprocessed=hmp_data.data, pattern=event_properties.template)
    lkh_comb, estimates_comb = model.fit_transform(trial_data, time_map=time_map, channel_map=channel_map, grouping_dict=group_dict)
    lkh_b_group, estimates_b_group = model.transform(trial_data_b)

    plot_topo_timecourse(epoch_data, estimates_comb, positions, as_time=True, colorbar=False, )
    plot_topo_timecourse(epoch_data, estimates_b, positions, as_time=True, 
                       max_time=500, colorbar=False, )