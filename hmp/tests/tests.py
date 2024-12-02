%load_ext autoreload
%autoreload 2
    
## Importing these packages is specific for this simulation case
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import os
from scipy.stats import gamma
from mne.io import read_info
import pandas as pd


## Importing HMP
import hmp
from hmp import simulations

cpus = 1 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM

n_trials = 2 #Mini for testing
sfreq = 100
n_events = 2
frequency = 10. #Frequency of the event defining its duration, half-sine of 10Hz = 50ms
amplitude = 1 #Amplitude of the event in nAm, defining signal to noise ratio
shape = 2 #shape of the gamma distribution
means = np.array([60, 150, 80])/shape #Mean duration of the between event times in ms
names = ['inferiortemporal-lh','caudalanteriorcingulate-rh','bankssts-lh']#Which source to activate for each event (see atlas when calling simulations.available_sources())

sources = []
for source in zip(names, means): #One source = one frequency, one amplitude and a given by-trial variability distribution
    sources.append([source[0], frequency, amplitude, gamma(shape, scale=source[1])])

# Function used to generate the data
file = simulations.simulate(sources, n_trials, cpus, 'dataset_raw', overwrite=False, sfreq=sfreq, seed=1)
#load electrode position, specific to the simulations
positions = simulations.simulation_positions()

# Reading the data
events = np.load(file[1])
resp_trigger = int(np.max(np.unique(events[:,2])))#Resp trigger is the last source in each trial
event_id = {'stimulus':1}#trigger 1 = stimulus
resp_id = {'response':resp_trigger}
eeg_data = hmp.utils.read_mne_data(file[0], event_id=event_id, resp_id=resp_id, sfreq=sfreq, 
            events_provided=events, verbose=False)
hmp_data = hmp.utils.transform_data(eeg_data, apply_standard=False, n_comp=4)
init = hmp.models.hmp(data=hmp_data, epoch_data=eeg_data, sfreq=eeg_data.sfreq,
                      event_width=50, distribution='gamma', shape=2)
sim_source_times, true_pars, true_magnitudes, _ = simulations.simulated_times_and_parameters(events, init)

true_estimates = init.fit_single(n_events, parameters = true_pars, magnitudes=true_magnitudes, maximization=False, verbose=False)
estimates = init.fit_single(n_events, verbose=False)
selected = init.fit_single(n_events, method='random', starting_points=2,
                           return_max=False,verbose=False)#funct
hmp.visu.plot_topo_timecourse(eeg_data, estimates, positions, init, magnify=1, sensors=True, times_to_display = np.mean(np.cumsum(sim_source_times,axis=1),axis=0))
assert np.abs(np.sum(init.compute_times(eeg_data,true_estimates) - init.compute_times(eeg_data,estimates)))<1.1 #If error is reasonable
hmp.visu.plot_distribution(estimates.eventprobs.sel(trial_x_participant=('S0',1)), 
                            xlims=(0,np.percentile(sim_source_times.sum(axis=1), q=90)));

fig, ax = plt.subplots(1,2, figsize=(6,2), sharey=True, sharex=True)
colors = iter([plt.cm.tab10(i) for i in range(10)])

for channel in  ['EEG 031', 'EEG 039', 'EEG 040', 'EEG 048']:
    c = next(colors)
    fakefit = init.fit_single(2, maximization=False, verbose=False)#Just to get the stim ERP in the same format
    BRP_times = init.compute_times(init, fakefit, fill_value=0, add_rt=True)
    times = BRP_times.sel(event=[0,3])#Stim and response only
    times['event'] = [0,1]
    erp_data = hmp.visu.erp_data(eeg_data.stack(trial_x_participant=["participant","epochs"]), times, channel)
    hmp.visu.plot_erp(times, erp_data, c, ax[0], upsample=2, label=channel)
    BRP_times = init.compute_times(init, estimates, fill_value=0, add_rt=True)#Real estimate
    erp_data = hmp.visu.erp_data(eeg_data.stack(trial_x_participant=["participant","epochs"]), BRP_times, channel,100)
    hmp.visu.plot_erp(BRP_times, erp_data, c, ax[1], upsample=2)
ev_colors = iter(['red', 'purple','brown','black',])
sim_event_times_cs = np.cumsum(sim_source_times, axis=1)
for event in range(2):
    c = next(ev_colors)
    ax[1].vlines(sim_event_times_cs[:,event].mean()*2, ymin=-3e-6, ymax=3e-6, color=c, alpha=.75)
ax[0].set_xlabel('Time (ms) from stimulus')
ax[1].set_xlabel('(Resampled) Time (ms) from stimulus')
plt.xlim(0,80)
ax[0].legend(bbox_to_anchor=(2.9,.85))
plt.show()

fig, ax = plt.subplots(1,2, figsize=(9,2), sharey=True, sharex=True)
colors = iter([plt.cm.tab10(i) for i in range(10)])

data = eeg_data.stack({'trial_x_participant':['participant','epochs']}).data.dropna('trial_x_participant', how="all")
times = init.compute_times(init, estimates, fill_value=0, add_rt=True)

# Plotting the single trial aligned events
baseline, n_samples = -50, 50#Take 50 samples on both sides, i.e. 100ms in a 500Hz signal
ev_colors = iter(['red', 'purple','brown','black',])
for i, event in enumerate(times.event[1:-1]):
    c = next(ev_colors)
    centered = hmp.utils.centered_activity(data, times, ['EEG 031',  'EEG 040', 'EEG 048'], event=event, baseline=baseline, n_samples=n_samples)
    ax[i].plot(centered.samples*2, centered.data.unstack().mean(['trials', 'channel', 'participant']).data, color=c)
    ax[i].set(title=f"Event {event.values}", ylim=(-5.5e-6, 5.5e-6), xlabel=f'Time (ms) around {event.values}')
    if i == 0:
        ax[i].set_ylabel("Voltage")

plt.xlim(-10,10);

# EEG data
epoch_data = xr.load_dataset(os.path.join('../../tutorials/sample_data/sample_data.nc'))
epoch_data = epoch_data.sel(participant=['processed_0025_epo', 'processed_0023_epo',])
# channel information
info = read_info(os.path.join('../../tutorials/sample_data/eeg/processed_0022_epo.fif'), verbose=False)
# select the data

hmp_data = hmp.utils.transform_data(epoch_data, apply_zscore='trial', n_comp=4)
hmp_speed_data = hmp.utils.condition_selection(hmp_data, epoch_data, 'SP', variable='cue') # select the conditions where participants needs to be fast
init_speed = hmp.models.hmp(hmp_speed_data, epoch_data, sfreq=epoch_data.sfreq, cpus=cpus)
estimates_speed = init_speed.fit(tolerance=1e-1, step=20)
hmp.visu.plot_topo_timecourse(epoch_data, estimates_speed, info, init_speed, as_time=True, sensors=False, contours=False, event_lines=None, colorbar=False)
backward_speed = init_speed.backward_estimation(max_fit=estimates_speed, tolerance=1e-1)
hmp.visu.plot_topo_timecourse(epoch_data, backward_speed, info, init_speed)
#LOOCV
loocv_model_speed = hmp.loocv.loocv(init_speed, hmp_speed_data, backward_speed, print_warning=False, verbose=False)
hmp.visu.plot_loocv(loocv_model_speed, pvals=True, test='sign', indiv=True, mean=True)

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

init = hmp.models.hmp(hmp_data, epoch_data, sfreq=epoch_data.sfreq, cpus=cpus)
#fit the model - note that we use the full data again
model_stage_removed = init.fit_single_conds(magnitudes=mags4, parameters=pars4, pars_map=pars_map, mags_map=mags_map, conds=conds,  cpus=1, tolerance=1e-1, verbose=False)
hmp.visu.plot_topo_timecourse(epoch_data, model_stage_removed, info, init, magnify=1, sensors=False, time_step=1000/init.sfreq,xlabel='Time (ms)', event_lines=True, colorbar=True, title="Remove one event") 
