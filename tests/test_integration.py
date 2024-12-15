  
## Importing these packages is specific for this simulation case
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mne.io import read_info
from scipy.stats import gamma
from pathlib import Path
import gc

import hmp
from hmp import simulations

epoch_data_file = Path("tutorials", "sample_data", "sample_data.nc")
info_data_file = os.path.join("tutorials", "sample_data", "eeg", "processed_0022_epo.fif")
raw_data_file = os.path.join("tutorials", "sample_data", "eeg", "processed_0023_epo.fif")

def test_integration():
    epoch_data, sim_source_times, info = simulations.demo(1, 1)
    fig, ax = plt.subplots(1)#captures plots

        ## REading data
    # print(info_data_file2 )
    epoch_data = hmp.utils.read_mne_data([info_data_file, raw_data_file] , epoched=True, sfreq=81,
                                verbose=True, pick_channels='eeg', lower_limit_RT=0.2,  
                                         upper_limit_RT=2, )
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=True, n_comp=2, method='mcca', apply_zscore='all')
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=True, n_comp=2, method='mcca', cov=False, apply_zscore='participant', mcca_reg=1)

    
    n_trials = 2 #Mini for testing
    sfreq = 110
    n_events = 2
    frequency = 10. #Frequency of the event defining its duration, half-sine of 10Hz = 50ms
    amplitude = 1 #Amplitude of the event in nAm, defining signal to noise ratio
    shape = 2 #shape of the gamma distribution
    means = np.array([60, 150, 80])/shape #Mean duration of the between event times in ms
    names = ['inferiortemporal-lh','caudalanteriorcingulate-rh','bankssts-lh']#Which source to activate for each event (see atlas when calling simulations.available_sources())

    sources = []
    for cur_name, cur_mean in zip(names, means): #One source = one frequency, one amplitude and a given by-trial variability distribution
        sources.append([cur_name, frequency, amplitude, gamma(shape, scale=cur_mean)])
    simulations.simulation_sfreq()
    # Function used to generate the data
    raw, events = simulations.simulate(sources, n_trials, 1, 'dataset_raw', overwrite=True, sfreq=sfreq, seed=1)
    #load electrode position, specific to the simulations
    positions = simulations.simulation_positions()
    # Reading the data
    events = np.load(events)
    resp_trigger = int(np.max(np.unique(events[:,2])))#Resp trigger is the last source in each trial
    event_id = {'stimulus':1}#trigger 1 = stimulus
    resp_id = {'response':resp_trigger}
    sfreq = 100
    # epoch_data = hmp.utils.read_mne_data(raw , event_id=event_id, resp_id=resp_id, sfreq=sfreq, 
    #                             verbose=True, pick_channels=['Cz', 'Pz'])
    epoch_data = hmp.utils.read_mne_data(raw, event_id=event_id, resp_id=resp_id, sfreq=sfreq, 
                events_provided=events, verbose=True, subj_idx='S0', reference='average', high_pass=1, low_pass=45,)
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=False, method=None, apply_zscore=False)
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=2, bandfilter=(1,40))
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=2, bandfilter=(1,40),cov=False)
    hmp_data = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=2, bandfilter=(1,40),cov=False, averaged=True)
    for distribution in  ['lognormal','wald','weibull','gamma']:
        init = hmp.models.hmp(data=hmp_data, epoch_data=epoch_data, 
                            event_width=50, distribution=distribution, shape=2)
        sim_source_times, true_pars, true_magnitudes, _ = simulations.simulated_times_and_parameters(events, init)
    
        true_estimates = init.fit_single(n_events, parameters = true_pars, magnitudes=true_magnitudes, maximization=False, verbose=False)
    estimates = init.fit_single(n_events, verbose=False)
    simulations.classification_true(init.compute_topographies(epoch_data, true_estimates, init, mean=True),init.compute_topographies(epoch_data, estimates, init, mean=True))
    selected = init.fit_single(n_events, method='random', starting_points=2,
                            return_max=False,verbose=False)#funct
    hmp.utils.save(selected, 'selected.nc')
    hmp.utils.load('selected.nc')

### testing plot_topo
    
    hmp.visu.plot_topo_timecourse(epoch_data, estimates, positions, init, magnify=1, sensors=True, times_to_display = np.mean(np.cumsum(sim_source_times,axis=1),axis=0),ax=ax, topo_size_scaling=True, max_time=2)
    hmp.visu.save_model_topos(epoch_data, estimates, positions, init, fname='topo', figsize=None, dpi=300, cmap='Spectral_r', 
                vmin=None, vmax=None, sensors=False, contours=6, colorbar=True)
    p = hmp.visu.plot_topo_timecourse(epoch_data, estimates, positions, init, sensors=False, 
                                      times_to_display = None, title='a', max_time=None)
    plt.close()
    assert np.abs(np.sum(init.compute_times(epoch_data,true_estimates) - init.compute_times(epoch_data,estimates)))<1.1 #If error is reasonable
    #fig, ax = plt.subplots(1)#captures plots
    #hmp.visu.plot_distribution(estimates.eventprobs.sel(trial_x_participant=('S0',1)), 
                                #xlims=(0,np.percentile(sim_source_times.sum(axis=1), q=90)), ax=ax);

    fig, ax = plt.subplots(1,2, figsize=(6,2), sharey=True, sharex=True)
    colors = iter([plt.cm.tab10(i) for i in range(10)])

    for channel in  ['EEG 031', 'EEG 039']:
        c = next(colors)
        fakefit = init.fit_single(2, maximization=False, verbose=False)#Just to get the stim ERP in the same format
        BRP_times = init.compute_times(init, fakefit, fill_value=0, add_rt=True)
        times = BRP_times.sel(event=[0,3])#Stim and response only
        times['event'] = [0,1]
        erp_data = hmp.visu.erp_data(epoch_data.stack(trial_x_participant=["participant","epochs"]), times, channel)
        hmp.visu.plot_erp(times, erp_data, c, ax[0], upsample=2, label=channel, bootstrap=2,minmax_lines=[1,2])
    ev_colors = iter(['red', 'purple','brown','black',])
    sim_event_times_cs = np.cumsum(sim_source_times, axis=1)
    for event in range(2):
        c = next(ev_colors)
        ax[1].vlines(sim_event_times_cs[:,event].mean()*2, ymin=-3e-6, ymax=3e-6, color=c, alpha=.75)
    ax[0].set_xlabel('Time (ms) from stimulus')
    ax[1].set_xlabel('(Resampled) Time (ms) from stimulus')
    plt.xlim(0,80)
    ax[0].legend(bbox_to_anchor=(2.9,.85))
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(9,2), sharey=True, sharex=True)
    colors = iter([plt.cm.tab10(i) for i in range(10)])

    data = epoch_data.stack({'trial_x_participant':['participant','epochs']}).data.dropna('trial_x_participant', how="all")
    times = init.compute_times(init, estimates, fill_value=0, add_rt=True)

    # Plotting the single trial aligned events
    baseline, n_samples = -5, 5#Take 50 samples on both sides, i.e. 100ms in a 500Hz signal
    ev_colors = iter(['red', 'purple','brown','black',])
    for i, event in enumerate(times.event[1:3]):
        c = next(ev_colors)
        centered = hmp.utils.centered_activity(data, times, ['EEG 031',  'EEG 040', 'EEG 048'], event=event, baseline=baseline, n_samples=None, cut_before_event=1, cut_after_event=0)
        ax[i].plot(centered.samples*2, centered.data.unstack().mean(['trials', 'channel', 'participant']).data, color=c)
        ax[i].set(title=f"Event {event.values}", ylim=(-5.5e-6, 5.5e-6), xlabel=f'Time (ms) around {event.values}')
        if i == 0:
            ax[i].set_ylabel("Voltage")

    plt.xlim(-10,10);
    plt.close()
    # EEG data
    epoch_data = xr.load_dataset(epoch_data_file)
    epoch_data = epoch_data.sel(participant=['processed_0025_epo', 'processed_0023_epo',], epochs=range(5))
    # channel information
    info = read_info(info_data_file, verbose=False)
    # select the data
    fig, ax = plt.subplots(1)#captures plots
    
    hmp.utils.condition_selection_epoch(epoch_data, 'SP', variable='cue', method='equal')
    hmp.utils.condition_selection_epoch(epoch_data, 'SP', variable='cue', method='contains')
    hmp_data = hmp.utils.transform_data(epoch_data, apply_zscore='trial', n_comp=2)
    hmp.utils.participant_selection(hmp_data, 'processed_0025_epo')
    hmp_speed_data = hmp.utils.condition_selection(hmp_data, 'SP', variable='cue') # select the conditions where participants needs to be fast
    hmp.utils.condition_selection(hmp_data, 'SP', variable='cue', method='contains')
    init_speed = hmp.models.hmp(hmp_speed_data, epoch_data, sfreq=epoch_data.sfreq, cpus=1)
    estimates_speed = init_speed.fit(tolerance=1e-1, step=50)
    hmp.visu.plot_topo_timecourse(epoch_data, estimates_speed, info, init_speed, as_time=True, sensors=False, contours=False, event_lines=None, colorbar=False, ax=ax)
    backward_speed = init_speed.backward_estimation(max_fit=estimates_speed, tolerance=1e-1, max_events=2)
    fig, ax = plt.subplots(1)#captures plots
    hmp.visu.plot_topo_timecourse(epoch_data, backward_speed, info, init_speed, ax=ax)
    #LOOCV
    loocv_model_speed = hmp.loocv.loocv(init_speed, hmp_speed_data, backward_speed, print_warning=False, verbose=False)
    #Same but testing multiproc.
    init_speed = hmp.models.hmp(hmp_speed_data, epoch_data, sfreq=epoch_data.sfreq, cpus=2)
    backward_speed = init_speed.backward_estimation(max_fit=estimates_speed, tolerance=1e-1, max_events=2)
    #LOOCV
    hmp.visu.plot_components_sensor(hmp_data, info)
    times = init.compute_times(init, estimates).values
    hmp.visu.plot_distribution(times, xlims=False, figsize=(8, 3), survival=False)
    hmp.visu.plot_distribution(times, xlims=[0,1], figsize=(8, 3), survival=True)
    hmp.visu.plot_latencies(times, init=init, labels=[], 
    figsize=False, errs=None, kind='point', legend=True, max_time=None, as_time=True)
    hmp.visu.plot_latencies(times, init=init, labels=[], 
    figsize=False, errs=None, kind='point', legend=True, max_time=None, as_time=True)
    hmp.visu.plot_latencies(estimates, init=init, labels=[], 
    figsize=False, errs='std', kind='bar', legend=False, max_time=None, as_time=False)
    hmp.visu.plot_latencies(estimates, init=init, labels=[], 
    figsize=False, errs='std', kind='bar', legend=False, max_time=None, as_time=True)
    hmp.visu.save_model_topos(epoch_data, backward_speed, info, init, fname='topo', figsize=None, dpi=300, cmap='Spectral_r', 
                vmin=None, vmax=None, sensors=False, contours=6, colorbar=True)
    correct_loocv_model = hmp.loocv.loocv_backward(init, hmp_data, max_events=2)

    loocv_model_speed = hmp.loocv.loocv(init_speed, hmp_speed_data, backward_speed, print_warning=False, verbose=False)
    
    hmp.visu.plot_loocv(loocv_model_speed, pvals=True, test='t-test', figsize=(16,5), indiv=True,  mean=False, additional_points=None)
    fig, ax = plt.subplots(2)#captures plots
    hmp.visu.plot_loocv(loocv_model_speed, pvals=True, test='sign', figsize=(16,5), indiv=True, mean=True, additional_points = (3.5, loocv_model_speed[0]), ax=ax)
    hmp.visu.plot_loocv(loocv_model_speed, pvals=False, test='sign', figsize=(16,5), indiv=True, mean=True, additional_points = (3.5, loocv_model_speed[0]), ax=ax)
    # the magnitudes map indicates the events. We have two conditions, speed and accuracy, hence two rows. For speed the third event is missing, indicated by the -1.
    mags_map = np.array([[0, -1],
                        [0, 0]])

    # the parameters maps indicates the stages. Here, we indicate that stage 3 is missing for the speed condition. Obviously, this has to be congruent with magnitudes map.
    pars_map = np.array([[0, -1, 0],
                        [0, 0, 0]])

    # finally, we have to define the conditions we want to analyze:
    conds = {'cue': ['SP', 'AC']} #dictionary with conditions to analyze as well as the levels.
    # we take the starting parameters from the accuracy model (you could also take the average)
    mags4 = backward_speed.sel(n_events=2).magnitudes.dropna('event').data
    pars4 = backward_speed.sel(n_events=2).parameters.dropna('stage').data

    fig, ax = plt.subplots(1)#captures plots
    init = hmp.models.hmp(hmp_data, epoch_data, sfreq=epoch_data.sfreq, cpus=1)
    #fit the model - note that we use the full data again
    
    model_stage_removed = init.fit_single_conds(magnitudes=mags4, parameters=pars4, pars_map=pars_map, mags_map=mags_map, conds=conds,  cpus=1, tolerance=1e-1, verbose=False)
    hmp.visu.plot_latencies(model_stage_removed, init=init, labels=[], 
    figsize=False, errs='std', kind='bar', legend=False, max_time=None, as_time=False)
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, init, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax,) 
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, init, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax, times_to_display=[1,2]) 
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, init, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax, times_to_display=[[1,2]]) 
    hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, init, magnify=1, sensors=False, as_time=True, xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event",ax=ax, times_to_display=np.array([1,2]))

    hmp.visu.save_model_topos(epoch_data, model_stage_removed, info, init, fname='topo', figsize=None, dpi=300, cmap='Spectral_r', 
                vmin=None, vmax=None, sensors=False, contours=6, colorbar=True)
    correct_loocv_model = hmp.loocv.loocv_backward(init, hmp_data, max_events=2)
    hmp.utils.save_eventprobs(selected.eventprobs, 'selected_eventprobs.csv')
    _loocv_combined = hmp.loocv.loocv(init, hmp_data, model_stage_removed, print_warning=False)

    # Remove temporary files
    os.remove("dataset_raw_raw_generating_events.npy")
    os.remove("dataset_raw_raw.fif")
    os.remove("selected_eventprobs.csv")
    os.remove("selected.nc")
    for filename in os.listdir():
        if filename.endswith('.png'):
            os.remove(filename)

    #close all plots
    plt.cla()
    plt.close()
    gc.collect()
