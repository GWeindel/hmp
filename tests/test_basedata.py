import pytest
from pathlib import Path
from hmp import io
from hmp import simulations
import numpy as np
import hmp

DATA_DIR = Path("tests", "gen_data")
DATA_DIR_A = DATA_DIR / "dataset_a"
DATA_DIR_B = DATA_DIR / "dataset_b"

@pytest.fixture
def init_data():
    """ Initialize all data and model related info."""
    sfreq = 100
    n_events = 3
    events = []
    centering_id = {'stimulus/0':1}#trigger 1 = stimulus
    event_id = {'response/0':5}
    raws = [DATA_DIR_A / 'dataset_a_raw_raw.fif', DATA_DIR_B / 'dataset_b_raw_raw.fif']
    event_files = [DATA_DIR_A / 'dataset_a_raw_raw_generating_events.npy',
                   DATA_DIR_B / 'dataset_b_raw_raw_generating_events.npy']
    for file in event_files:
        events.append(np.load(file))
    event_a = events[0]
    event_b = events[1]
    # Data reading
    preprocessing_kwargs = dict(sfreq = sfreq)
    montage = simulations.sim_info().get_montage()
    epoch_data, info = io.read_mne_raw(raws, centering_id=centering_id, 
                event_id=event_id, events_provided=events,
                subj_name=['a','b'], montage=montage,
                preprocessing_kwargs=preprocessing_kwargs)
    epoch_data = epoch_data.assign_coords({'condition': ('recording', epoch_data.subject.data)})
    epoch_data = epoch_data.sel(channel=epoch_data.channel[::4])
    print(epoch_data.channel)
    positions = simulations.positions()
    return event_b, event_a, epoch_data, positions, sfreq, n_events

@pytest.mark.parametrize("n_comp,center,whiten,reject_threshold,min_duration,max_duration", [
    (5, True, True, None, None, None),
    (10, False, True, 0.1, None, None),
    (1, True, False, None, 0.05, 1.5),
    (2, False, False, 0.2, 0.1, 2.0),
])
def test_proj_pca_custom_variants(init_data, n_comp, center, whiten, reject_threshold, min_duration, max_duration):
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data
    pca = hmp.basedata.BaseData.from_io_all_pca(epoch_data, n_comp=n_comp, center=center, whiten=whiten, interval_id = 'response_time')
    assert pca.data.shape[1] == n_comp
    if whiten:
        assert np.allclose(pca.data.var(dim=['trial','sample']), 1, atol=0.05)
    custom = hmp.basedata.BaseData.from_io(epoch_data, weights=pca.weights,
            projection_kwargs={'center': center}, crop=True, reject=True,
            apply_variance=True, variance_kwargs={'whiten': whiten},
            projection_type='custom', interval_id = 'response_time')
    assert custom.data.shape[1] == n_comp
    if whiten:
        assert np.allclose(custom.data.var(dim=['trial','sample']), 1, atol=0.05)

@pytest.mark.parametrize("center,whiten,reject_threshold,min_duration,max_duration", [
    (True, True, None, None, None),
    (False, True, 0.1, None, None),
    (True, False, None, 0.05, 1.5),
    (False, False, 0.2, 0.1, 2.0),
])
def test_proj_identity_variants(init_data, center, whiten, reject_threshold, min_duration, max_duration):
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data
    identity = hmp.basedata.BaseData.from_io(epoch_data, crop=True, reject=True,
                                             apply_variance=True, interval_id = 'response_time'
                                             projection_kwargs={'center': center},
                                             variance_kwargs={'whiten': whiten})
    assert identity.data.shape[1] == epoch_data.sizes['channel']
    if whiten:
        assert np.allclose(identity.data.var(dim=['trial','sample']), 1, atol=0.05)

@pytest.mark.parametrize("n_comp,center,whiten,reject_threshold,min_duration,max_duration", [
    (5, True, True, None, None, None),
    (10, False, True, 0.1, None, None),
    (1, True, False, None, 0.05, 1.5),
    (2, False, False, 0.2, 0.1, 2.0),
])
def test_proj_pca_custom_variants_newbasedata(init_data, n_comp, center, whiten,
                                 reject_threshold, min_duration, max_duration):
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data
    #pca = hmp.basedata.BaseData.from_io_all_pca(epoch_data, n_comp=n_comp, center=center, whiten=whiten)
    pca = hmp.basedata.BaseData.from_io(epoch_data)
    pca.crop_epochs()
    pca.reject_epochs()
    pca.pca_and_variance(n_comp=n_comp,center=center,whiten=whiten)

    assert pca.data.shape[1] == n_comp
    if whiten:
        assert np.allclose(pca.data.var(dim=['trial','sample']), 1, atol=0.05)

    custom = hmp.basedata.BaseData.from_io(epoch_data)
    custom.crop_epochs()
    custom.reject_epochs()
    custom.project(projection_type='custom',center=center, weights=pca.weights)
    custom.apply_variance_ops(whiten=whiten)
    data = hmp.basedata.BaseData.remove_participant(custom, 'a')
    data = hmp.basedata.BaseData.get_participants(custom, ['a'])

    #and pca pca based
    pca = hmp.basedata.BaseData.from_io(epoch_data)
    pca.crop_epochs()
    pca.reject_epochs()
    pca.pca_and_variance(n_comp=n_comp,center=center,
                         whiten=whiten, method_pca='pca')

    assert custom.data.shape[1] == n_comp
    if whiten:
        assert np.allclose(custom.data.var(dim=['trial','sample']), 1, atol=0.05)
