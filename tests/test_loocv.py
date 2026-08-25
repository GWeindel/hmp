import hmp
import copy

from test_io import init_data

def example_simple_func(data, n_events, channel_pars=None, time_pars=None, verbose=False):
    ev_model = hmp.models.EventModel(n_events=n_events)
    ev_model.fit(data, channel_pars=channel_pars, time_pars=time_pars, verbose=verbose)
    return ev_model

def test_loocv():
    cpus=2 #useful for loocv with 2 subjects

    _, _, epoch_data, info, sfreq, n_events = init_data()

    base_data_noPCA = hmp.basedata.from_io(epoch_data)
    base_data_noPCA.crop_reject_epochs(duration_id='response_time')

    base_data = copy.deepcopy(base_data_noPCA)
    base_data.project(hmp.projectors.PCA(n_comp=3))
    base_data.apply_variance_ops()

    pattern_data = hmp.patterndata.PatternData.from_basedata(base_data)

    #basic inclusing average & plot
    model = hmp.models.EventModel(n_events=n_events)
    loocv = hmp.loocv.LOOCV(model, quick=False)
    lkh_loocv, modelfits_loocv = loocv.fit(pattern_data, cpus_cv=cpus, cpus_model=1)

    average_estimate, average_eventprobs = hmp.loocv.get_average_fit_eventprobs(modelfits_loocv, pattern_data)
    hmp.visu.plot_model(epoch_data, average_eventprobs, info, as_time=True)

    #base_data and quick
    model_3ev_fit = hmp.models.EventModel(n_events=3)
    model_3ev_fit.fit(data=pattern_data,cpus=cpus)
    loocv3ev_quick = hmp.loocv.LOOCV(model_3ev_fit,quick=True)
    lkh_loocv_quick, modelfits_loocv_quick = loocv3ev_quick.fit(base_data, cpus_cv=cpus, cpus_model=1)

    #including PCA
    model_3ev = hmp.models.EventModel(n_events=3)
    loocv3ev_pca = hmp.loocv.LOOCV(model_3ev, pca_cv = True, pca_kwargs={'n_comp': 3})
    lkh_loocv_pca, modelfits_loocv_pca = loocv3ev_pca.fit(base_data_noPCA, cpus_cv=cpus, cpus_model=1)

    #eliminative with more plots
    model_elim = hmp.models.EliminativeMethod(max_events=3)
    loocv_elim = hmp.loocv.LOOCV(model_elim, quick=False)
    lkh_loocv_elim, modelfits_loocv_elim = loocv_elim.fit(pattern_data, cpus_cv=cpus, cpus_model=1)

    average_estimate, average_eventprobs = hmp.loocv.get_average_fit_eventprobs(modelfits_loocv_elim, pattern_data)
    hmp.visu.plot_model(epoch_data, average_eventprobs, info, as_time=True)
    hmp.visu.plot_loocv(lkh_loocv_elim,pvals=True, test="sign",mean=True)

    #cumulative
    model_cumulative = hmp.models.CumulativeMethod()
    loocv_cumu = hmp.loocv.LOOCV(model_cumulative, quick=False)
    lkh_loocv_cumu, modelfits_loocv_cumu = loocv_cumu.fit(pattern_data, cpus_cv=cpus, cpus_model=1)

    #function
    loocv_simple_func = hmp.loocv.LOOCV(example_simple_func, function_kwargs={'n_events' : 3})
    lkh_simple_func, modelfits_simple_func = loocv_simple_func.fit(pattern_data, cpus_cv=cpus, cpus_model=1)
