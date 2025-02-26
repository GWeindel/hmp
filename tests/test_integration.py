## Importing these packages is specific for this simulation case
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma
from pathlib import Path
import gc

import hmp
from hmp import simulations
from hmp.models import FixedEventModel, CumulativeEstimationModel, BackwardEstimationModel
from hmp.models.base import EventProperties
from hmp.trialdata import TrialData


real_data = os.path.join("tutorials", "sample_data", "eeg", "processed_0022_epo.fif")
real_metadata = os.path.join("tutorials", "sample_data", "eeg", "0022.csv")


def test_integration():

    # Testing isolated functions in simulations
    assert simulations.simulation_sfreq() == 600.614990234375    
    epoch_data, sim_source_times, info = simulations.demo(1, 1)
    # Data creation/reading
    ## Simulation parameters
    sfreq = 102
    n_events = 3
    n_trials = 2
    cpus=1
    times_a = np.array([[50, 50, 150, 50],
             [50, 100, 100, 50],])
    names = ['bankssts-rh','bankssts-lh','bankssts-rh','bankssts-lh']
    sources = []
    for cur_name in names:
        sources.append([cur_name, 10., 3.1e-8, gamma(2, scale=1)])
    raw_a, event_a,_ = simulations.simulate(sources, n_trials, cpus, 'dataset_a_raw', overwrite=True,
                                          sfreq=sfreq, times=times_a, noise=True, seed=1, save_snr=True)
    means = np.array([50, 100, 100, 50])/2
    sources = []
    for cur_name, cur_mean in zip(names, means):
        sources.append([cur_name, 10., 1, gamma(2, scale=cur_mean)])
    raw_b, event_b = simulations.simulate(sources, n_trials, cpus, 'dataset_b_raw', seed=1, overwrite=True, 
        sfreq=sfreq, verbose=True, proportions=[.99,1,1,1], noise=False)
    
    events = []
    raws = []
    event_id = {'stimulus':1}#trigger 1 = stimulus
    resp_id = {'response':5}
    
    for raw_file, event_file in zip([raw_a, raw_b],
                    [event_a, event_b]):
        events.append(np.load(event_file))
        raws.append(raw_file)
    sfreq = 100
    
    # Data reading
    df_real_metadata = pd.read_csv(real_metadata) 
    epoch_data = hmp.utils.read_mne_data(real_data,  epoched=True, sfreq=10, verbose=True, high_pass=1, low_pass=45,
                                         reference='average',ignore_rt=True, reject_threshold=1e1, 
                                         metadata=df_real_metadata, pick_channels=['Cz'])
    epoch_data = hmp.utils.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq,
            events_provided=events, verbose=True, reference='average', high_pass=1, low_pass=45,
                subj_idx=['a','b'],pick_channels='eeg', lower_limit_rt=0.01, upper_limit_rt=2 )
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    positions = simulations.simulation_positions()
    
    
    # Testing transform data
    hmp_data_sim = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=2, method=None, apply_zscore=True, centering=False, copy=True)
    hmp_data_sim = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=2, apply_zscore=False, centering=False, zscore_across_pcs=True, copy=True)
    hmp_data_sim = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=2, bandfilter=(1,40),cov=False,apply_zscore='all', centering=False, copy=True)
    hmp_data_sim = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=2, bandfilter=(1,40),cov=False, averaged=True, apply_zscore='participant', centering=False, copy=True)
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=True, n_comp=2, method='mcca', apply_zscore='all', bandfilter=(1,40), zscore_across_pcs=True, centering=False, copy=True)
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=True, n_comp=2, method='mcca', cov=False, apply_zscore='participant', mcca_reg=1, zscore_across_pcs=True, centering=False, copy=True)
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=True, n_comp=2, method='mcca', cov=False, apply_zscore='participant', mcca_reg=1, zscore_across_pcs=True, centering=False, averaged=True, copy=True)
    hmp_data = hmp.utils.transform_data(epoch_data, n_comp=2, copy=True,)

    # Testing condition selection functions and methods      
    hmp.utils.condition_selection(epoch_data, 'a', variable='condition', method='equal')
    hmp.utils.condition_selection(epoch_data, 'a', variable='condition', method='contains')
    hmp.utils.condition_selection_epoch(epoch_data, 'a', variable='condition', method='equal')
    hmp.utils.condition_selection_epoch(epoch_data, 'a', variable='condition', method='contains')
    hmp_speed_data = hmp_data

        
   # Comparing to simulated data, asserting that results are the one simulated
    events_a = np.load(event_a)
    data_a = hmp.utils.participant_selection(hmp_data, 'a')
    event_properties = EventProperties.create_expected(sfreq=data_a.sfreq)
    trial_data = TrialData.from_standard_data(data=data_a, template=event_properties.template)
    init_sim = FixedEventModel(event_properties, n_events=n_events, maximization=False,)
    sim_source_times, true_pars, true_magnitudes, _ = simulations.simulated_times_and_parameters(events_a, init_sim, trial_data)
    true_loglikelihood, true_estimates = init_sim.fit_transform(trial_data, parameters = np.array([true_pars]), magnitudes=np.array([true_magnitudes]),  verbose=True)
    true_topos = hmp.utils.event_topo(epoch_data, true_estimates.squeeze(), mean=True)
    init_sim = FixedEventModel(event_properties, n_events=n_events, maximization=True,)
    likelihood, estimates = init_sim.fit_transform(trial_data, verbose=True)
    test_topos = hmp.utils.event_topo(epoch_data, estimates.squeeze(), mean=True)
    assert (np.array(simulations.classification_true(true_topos,test_topos)) == np.array(([0,1,2],[0,1,2]))).all()
    assert np.sum(np.abs(true_topos.data - test_topos.data)) < 2.65e-05
    assert np.round(likelihood,4) > np.array(-1)


    # Initializing models
    ## Testing different distribution implementation
    event_properties = EventProperties.create_expected(hmp_data_sim.sfreq, width=50, shape=2)
    trial_data = TrialData.from_standard_data(hmp_data_sim, event_properties.template)
    for distribution in  ['lognormal','wald','weibull','gamma']:
        fixed_sim_model = FixedEventModel(trial_data, event_properties, distribution=distribution)
        estimates = fixed_sim_model.fit(n_events, verbose=True)

    ## different init parameters
    _events = EventProperties.create_expected(sfreq=epoch_data.sfreq)
    _trial_data = TrialData.from_standard_data(hmp_data, _events.template)
    fixed_sim_model = FixedEventModel(_trial_data, _events)

    _events_speed = EventProperties.create_expected(sfreq=epoch_data.sfreq)
    _trial_data_speed = TrialData.from_standard_data(hmp_speed_data, _events.template)
    fixed_speedsim_model = FixedEventModel(_trial_data, _events)
    
    
    # init = hmp.models.HMP(hmp_data, sfreq=epoch_data.sfreq)
    # init_speed = hmp.models.HMP(hmp_speed_data, sfreq=epoch_data.sfreq)

    # Testing fit
    ## fit_n tests
    selected = fixed_sim_model.fit(n_events, starting_points=2,
                                   return_max=False,verbose=True)#funct
    selected = fixed_sim_model.fit(3, parameters=estimates.parameters, magnitudes=estimates.magnitudes)#funct
    selected = fixed_sim_model.fit(3, parameters=estimates.parameters, magnitudes=estimates.magnitudes, cpus=2)#funct
    # ## Fit function
    estimates_speed, _ = CumulativeEstimationModel(_trial_data_speed, _events_speed).fit(tolerance=1e-1, step=10, diagnostic=True, by_sample=True, pval=1, return_estimates=True)
    estimates_speed, _ = CumulativeEstimationModel(_trial_data_speed, _events_speed).fit(tolerance=1e-1, step=10, diagnostic=True, by_sample=True,return_estimates=True)
    ## Backward function 
    backward_speed = BackwardEstimationModel(_trial_data_speed, _events_speed).fit(base_fit=estimates_speed, tolerance=1e-1, max_events=2)
    ## Condition fit
    mags_map = np.array([[1, -1],
                    [1, 0]])
    pars_map = np.array([[0, -1, 1],
                        [0, 0, 1]])
    conds = {'condition': ['a', 'b']} #dictionary with conditions to analyze as well as the levels.
    mags4 = backward_speed.sel(n_events=2).magnitudes.dropna('event').data
    pars4 = backward_speed.sel(n_events=2).parameters.dropna('stage').data
    model_stage_removed = FixedEventModel(_trial_data, _events).fit(magnitudes=mags4, parameters=pars4, pars_map=pars_map, mags_map=mags_map, level_dict=conds,  cpus=1, tolerance=1e-1, verbose=True)
    
    # LOOCV
    loocv_model_speed = hmp.loocv.loocv(init_speed, hmp_speed_data, backward_speed, print_warning=True, verbose=True, cpus=1)
    correct_loocv_model = hmp.loocv.loocv_backward(init, hmp_data, max_events=2)
    hmp.loocv.loocv_fit_backward(init_speed, hmp_speed_data, by_sample=False, min_events=2, cpus=1, verbose=True)
    hmp.loocv.loocv_func(init_speed, hmp_speed_data, hmp.loocv.example_fit_n_func, func_args=[1])
    hmp.loocv.loocv_func(init_speed, hmp_speed_data, hmp.loocv.example_complex_fit_n_func, func_args=[2])
    # loocv_combined = hmp.loocv.loocv(init_speed, hmp_speed_data, model_stage_removed, print_warning=True)
    # hmp.loocv.loocv_func(init_speed, hmp_speed_data, init_speed.fit(), func_args=[5], cpus=1, verbose=True) 
    # testing plot_topo
    fig,ax = plt.subplots(2)
    hmp.visu.plot_topo_timecourse(epoch_data, estimates, positions, 
                                  sensors=True, times_to_display = np.array([1,2]), ax=ax[0], topo_size_scaling=True, max_time=2)
    hmp.visu.plot_topo_timecourse(epoch_data, estimates, positions, sensors=False, 
                                  times_to_display = None, title='a', max_time=None)
    hmp.visu.plot_topo_timecourse(epoch_data, estimates_speed, info, 
                                  as_time=True, contours=False, event_lines=None, colorbar=False, ax=ax[0])
    hmp.visu.plot_topo_timecourse(epoch_data, backward_speed, info, ax=ax[0])
    
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax[0],) 
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax[0], times_to_display=[1,2]) 
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax[0], times_to_display=[[1,2]]) 
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax[0], times_to_display=np.array([1,2]))
    
    # Testing comoput_times
    hmp.utils.event_times(backward_speed.sel(n_events=1), duration=False, fill_value=None, mean=False, add_rt=False, as_time=False, errorbars='se', center_measure='mean',estimate_method='max')
    hmp.utils.event_times(backward_speed, duration=False, fill_value=None, mean=False, add_rt=True, as_time=False, errorbars='se', center_measure='mean')
    hmp.utils.event_times(model_stage_removed, duration=True, fill_value=None, mean=False, add_rt=True, as_time=True, errorbars='std', center_measure='median',estimate_method='mean')
    
    
    # Testing diverse plotting functions
    hmp.visu.plot_latencies(estimates_speed, init=init_speed, labels=[], 
        figsize=False, errs='std', kind='bar', legend=False, max_time=None, as_time=False)
    hmp.visu.plot_latencies(estimates_speed, init=init_speed, labels=[], 
        figsize=False, errs='std', kind='bar', legend=False, max_time=None, as_time=True)
    hmp.visu.plot_latencies(model_stage_removed, init=init_speed, labels=[], 
    figsize=False, errs='std', kind='bar', legend=False, max_time=None, as_time=False)
    
    hmp.visu.plot_loocv(loocv_model_speed, pvals=True, test='t-test', indiv=False,  mean=False, 
                        additional_points=None)
    hmp.visu.plot_loocv(loocv_model_speed, pvals=True, test='sign', indiv=True, mean=True, 
                        additional_points = (3.5, loocv_model_speed[0]), ax=ax)
    hmp.visu.plot_loocv(loocv_model_speed, pvals=False, test='sign', figsize=(1,1), indiv=True, mean=True, 
                        additional_points = (3.5, loocv_model_speed[0]), ax=ax)

    # testing save functions
    hmp.utils.save(selected, 'selected.nc')
    hmp.utils.load('selected.nc')
    hmp.utils.save(estimates_speed, 'estimates_speed.nc')
    hmp.utils.load('estimates_speed.nc')
    hmp.visu.save_model_topos(epoch_data, estimates, positions, 
                              fname='topo', figsize=None, dpi=300, cmap='Spectral_r', 
                vmin=None, vmax=None, sensors=False, contours=6, colorbar=True)
    hmp.visu.save_model_topos(epoch_data, backward_speed, info, fname='topo', figsize=None, dpi=300, cmap='Spectral_r', 
                vmin=None, vmax=None, sensors=False, contours=6, colorbar=True)
    hmp.visu.save_model_topos(epoch_data, model_stage_removed, info, fname='topo', figsize=None, dpi=300, cmap='Spectral_r', 
                vmin=None, vmax=None, sensors=False, contours=6, colorbar=True)
    hmp.utils.save_eventprobs(selected.eventprobs, 'selected_eventprobs.csv')
    
    # Testing parallelized func 
    init_speed = hmp.models.HMP(hmp_speed_data, sfreq=epoch_data.sfreq, cpus=2)
    backward_speed = init_speed.backward_estimation(max_fit=estimates_speed, tolerance=1e-1, max_events=2)
    
    # LOOCV
    loocv_model_speed = hmp.loocv.loocv(init_speed, hmp_speed_data, backward_speed, print_warning=True, verbose=True)
    correct_loocv_model = hmp.loocv.loocv_backward(init, hmp_data, max_events=2,cpus=2)
    hmp.loocv.loocv_fit_backward(init_speed, hmp_speed_data, by_sample=False, min_events=2, cpus=2, verbose=True)
    # loocv_combined = hmp.loocv.loocv(init_speed, hmp_speed_data, model_stage_removed, print_warning=True,cpus=2)
    
    # Testing electrode activity plots
    ## ERP plot
    fig, ax = plt.subplots(1,1, figsize=(1,1), sharey=True, sharex=True)
    fakefit = init_sim.fit_n(2, maximization=False, verbose=True)#Just to get the stim ERP in the same format
    BRP_times = hmp.utils.event_times(fakefit, fill_value=0, add_rt=True)
    times = BRP_times.sel(event=[0,1])#Stim and response only
    times['event'] = [0,1]
    erp_data = hmp.visu.erp_data(epoch_data.stack(trial_x_participant=["participant","epochs"]), times, 'EEG 031')
    hmp.visu.plot_erp(times, erp_data, ax=ax, upsample=2, label='EEG 031', bootstrap=2,minmax_lines=[1,2])

    data = epoch_data.stack({'trial_x_participant':['participant','epochs']}).data.dropna('trial_x_participant', how="all")
    times = hmp.utils.event_times(estimates, fill_value=0, add_rt=True)
    centered = hmp.utils.centered_activity(data, times, ['EEG 031'], event=1, baseline=1, n_samples=None, cut_before_event=1, cut_after_event=1)
    centered = hmp.utils.centered_activity(data, times, ['EEG 031'], event=0, baseline=1, center=False, cut_before_event=0, cut_after_event=0)
    
    # Remove temporary files
    os.remove("dataset_a_raw_raw_generating_events.npy")
    os.remove("dataset_a_raw_raw.fif")
    os.remove("dataset_a_raw_raw_snr.npy")
    os.remove("dataset_b_raw_raw_generating_events.npy")
    os.remove("dataset_b_raw_raw.fif")
    os.remove("selected_eventprobs.csv")
    os.remove("selected.nc")
    for filename in os.listdir():
        if filename.endswith('.png'):
            os.remove(filename)
    
    #close all plots
    plt.cla()
    plt.close()
    gc.collect()
