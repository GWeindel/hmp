import pytest
from pathlib import Path
from hmp import io
from hmp import simulations
import numpy as np
import hmp

from test_io import init_data

@pytest.fixture
def fixt_init_data():
    return init_data()

@pytest.mark.parametrize("n_comp,whiten,reject_threshold,min_duration,max_duration", [
    (5, True, None, None, None),
    (10, True, 0.1, None, None),
    (1, False, None, 0.05, 1.5),
    (1, False, 0.2, 0.1, 2.0),
])
def test_proj_pca_custom_variants(fixt_init_data, n_comp, whiten, reject_threshold, min_duration, max_duration):
    event_b, event_a, epoch_data, positions, sfreq, n_events = fixt_init_data
    pca = hmp.basedata.from_io(epoch_data)
    pca.crop_reject_epochs()
    pca.project(hmp.projectors.PCA(n_comp=n_comp))
    pca.apply_variance_ops(whiten=whiten)
    custom = hmp.basedata.from_io(epoch_data)
    custom.project(hmp.projectors.Custom(weights=pca.projector.weights))
    custom.apply_variance_ops()
    if isinstance(n_comp, int):
        assert pca.data.shape[1] == n_comp
        assert custom.data.shape[1] == n_comp
    if whiten:
        assert np.allclose(pca.data.var(dim=['trial','sample']), 1, atol=0.05)
        assert np.allclose(custom.data.var(dim=['trial','sample']), 1, atol=0.05)
        
    if n_comp == 5: #Only needs to run once
        # Testing default shortcut
        pca = hmp.basedata.default(epoch_data, n_comp=n_comp, whiten=whiten,
                               reject_amplitude=reject_threshold,
                               center=False,
                               min_duration =min_duration,
                               max_duration = max_duration,
                               duration_id = 'response_time')
        #testing identity
        identity = hmp.basedata.from_io(epoch_data)
        identity.project(hmp.projectors.Identity())
        identity.apply_variance_ops(whiten=whiten)
        assert identity.data.shape[1] == epoch_data.sizes['channel']
        if whiten:
            assert np.allclose(identity.data.var(dim=['trial','sample']), 1, atol=0.05)
        #Testing selection
        # Selecting participant
        # print(pca)
        a = pca.select_coord(value='a', variable='subject', method=np.equal)
        print(a)
        assert a.data.sizes['trial'] == 2
        # Selecting participant list
        not_a = pca.select_coord(value='a', variable='subject', method=np.not_equal)
        print(not_a)
        assert pca.data.sizes['trial'] == a.data.sizes['trial'] + not_a.data.sizes['trial']
        #Testing attributes preservation
        a.data.attrs['sfreq']
        pdata_a = hmp.patterndata.PatternData.from_basedata(a)
        pdata_a.durations.attrs['sfreq']

        # Testing shortcut
        pca = hmp.basedata.from_io(epoch_data)
        pca = pca.pca_and_variance(n_comp=1)