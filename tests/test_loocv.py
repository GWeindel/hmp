import hmp
import copy
import pytest
import numpy as np

from test_io import init_data

def example_simple_func(data, n_events, channel_pars=None, time_pars=None, verbose=False):
    ev_model = hmp.models.EventModel(n_events=n_events)
    ev_model.fit(data, channel_pars=channel_pars, time_pars=time_pars, verbose=verbose)
    return ev_model

def test_loocv():
    cpus=1 #In tests with small data, mp overhead actually slows test

    _, _, epoch_data, info, sfreq, n_events = init_data()

    base_data_noPCA = hmp.basedata.from_io(epoch_data)
    base_data_noPCA.crop_reject_epochs(duration_id='response_time')

    base_data = copy.deepcopy(base_data_noPCA)
    base_data.project(hmp.projectors.PCA(n_comp=2))
    base_data.apply_variance_ops()

    pattern_data = hmp.patterndata.PatternData.from_basedata(base_data)

    #basic inclusing average & plot
    model = hmp.models.EventModel(n_events=n_events)
    loocv = hmp.loocv.LOOCV(model, informed=False, pca_cv=False)
    lkh_loocv, modelfits_loocv = loocv.fit(pattern_data)
    
    assert lkh_loocv is not None
    assert modelfits_loocv is not None
    assert len(lkh_loocv) > 0
    assert len(modelfits_loocv) > 0
    
    average_estimate, average_eventprobs = hmp.loocv.get_average_fit_eventprobs(modelfits_loocv, pattern_data)
    hmp.visu.plot_model(epoch_data, average_eventprobs, info, as_time=True)

    # Testing to apply pca on unprojected data without pca_cv
    model = hmp.models.EventModel(n_events=n_events)
    loocv = hmp.loocv.LOOCV(model, informed=False, pca_cv=False, pca_kwargs={'n_comp': 2})
    lkh_loocv_pca, modelfits_loocv = loocv.fit(base_data_noPCA)
    assert np.isclose(np.sum(lkh_loocv_pca.values), np.sum(lkh_loocv.values), atol=1)
    
    # Testing trial dimension
    model = hmp.models.EventModel(n_events=n_events)
    loocv = hmp.loocv.LOOCV(model, informed=False, dimension='epoch', pca_cv=False)
    lkh_loocv, modelfits_loocv = loocv.fit(pattern_data)
    assert "epoch" in lkh_loocv.dims


    #base_data and informed
    model_3ev_fit = hmp.models.EventModel(n_events=3)
    model_3ev_fit.fit(data=pattern_data,cpus=cpus)
    loocv3ev_informed = hmp.loocv.LOOCV(model_3ev_fit,informed=True, pca_cv=False)
    # also testing mp
    lkh_loocv_informed, modelfits_loocv_informed = loocv3ev_informed.fit(pattern_data, cpus_cv=2, cpus_model=1)

    #function vs informed (also included mp comparison)
    loocv_simple_func = hmp.loocv.LOOCV(example_simple_func, function_kwargs={'n_events' : 3}, pca_cv=False)
    lkh_simple_func, modelfits_simple_func = loocv_simple_func.fit(pattern_data)
    assert lkh_simple_func is not None
    assert modelfits_simple_func is not None
    assert len(lkh_simple_func) > 0
    assert len(modelfits_simple_func) > 0
    assert np.isclose(np.sum(lkh_simple_func.values), np.sum(lkh_loocv_informed.values), atol=1)

    #including PCA
    model_3ev = hmp.models.EventModel(n_events=3)
    loocv3ev_pca = hmp.loocv.LOOCV(model_3ev, pca_cv = True, pca_kwargs={'n_comp': 3})
    lkh_loocv_pca, modelfits_loocv_pca = loocv3ev_pca.fit(base_data_noPCA)
    assert lkh_loocv_pca is not None
    assert modelfits_loocv_pca is not None
    assert len(lkh_loocv_pca) > 0
    assert len(modelfits_loocv_pca) > 0

    #eliminative with more plots
    model_elim = hmp.models.EliminativeMethod(max_events=3)
    # Without informed model
    loocv_elim = hmp.loocv.LOOCV(model_elim, informed=False, pca_kwargs={'n_comp': 3})
    lkh_loocv_elim, modelfits_loocv_elim = loocv_elim.fit(base_data_noPCA)
    # with
    model_elim.fit(pattern_data)
    loocv_elim = hmp.loocv.LOOCV(model_elim, informed=True, pca_cv=False)
    lkh_loocv_elim_inf, modelfits_loocv_elim_inf = loocv_elim.fit(pattern_data)
    assert lkh_loocv_elim_inf is not None
    assert modelfits_loocv_elim_inf is not None
    assert len(lkh_loocv_elim_inf) > 0
    assert len(modelfits_loocv_elim_inf) > 0
    
    average_estimate, average_eventprobs = hmp.loocv.get_average_fit_eventprobs(modelfits_loocv_elim, pattern_data)
    hmp.visu.plot_model(epoch_data, average_eventprobs, info, as_time=True)
    hmp.visu.plot_loocv(lkh_loocv_elim,pvals=True, test="sign",mean=True)

    #cumulative
    model_cumulative = hmp.models.CumulativeMethod()
    loocv_cumu = hmp.loocv.LOOCV(model_cumulative, pca_cv=False)
    lkh_loocv_cumu, modelfits_loocv_cumu = loocv_cumu.fit(pattern_data)
    assert lkh_loocv_cumu is not None
    assert modelfits_loocv_cumu is not None
    assert len(lkh_loocv_cumu) > 0
    assert len(modelfits_loocv_cumu) > 0