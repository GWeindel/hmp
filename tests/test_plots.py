import numpy as np
import hmp
from hmp.models import EventModel
from hmp.patterndata import PatternData
from hmp.visu import plot_topo_timecourse
from hmp import preprocessors


from test_io import init_data


def test_plot():
    _, _, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = preprocessors.ProjPCA(epoch_data, n_comp=2,)
    # Testing one event less in one condition
    channel_map = np.array([[0, 0, -1],
                         [0, 0, 0]])
    time_map = np.array([[0, 0, -1, 0],
                         [0, 0, 0, 0],])
    group_dict = {'condition': ['a', 'b']}
    
    hmp_data_a = hmp.utils.participant_selection(hmp_data.data, 'a')
    
    model = EventModel(n_events=n_events)
    
    # Perform a fit on a (should be too noisy)
    lkh_a, estimates_a = model.fit_transform(hmp_data_a)

    # Fit model on both conditions (noiseless b should help estimate a)
    model = EventModel(n_events=n_events, \
        channel_map=channel_map, time_map=time_map, grouping_dict=group_dict)
    model_a = EventModel(n_events=n_events)   
    lkh_comb, estimates_comb = model.fit_transform(hmp_data)
    lkh_a_group, estimates_a_group = model_a.fit_transform(hmp_data_a)

    plot_topo_timecourse(epoch_data, estimates_comb, positions, as_time=False, colorbar=False, )
    plot_topo_timecourse(epoch_data, estimates_a, positions, as_time=True, 
                       max_time=500, colorbar=False, )