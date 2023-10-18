HsMM MVpy
==========

![](plots/general_illustration.png)

HsMM MVpy is an open-source Python package to estimate Hidden Semi-Markov Models in a Multivariate Pattern Analysis (HMP) of neural time-series (e.g. EEG) based on the HsMM-MVPA method developed by Anderson, Zhang, Borst, & Walsh  ([2016](https://psycnet.apa.org/doi/10.1037/rev0000030); see Borst & Anderson, [2021](http://jelmerborst.nl/pubs/ACTR_HMP_MVPA_BorstAnderson_preprint.pdf), for an accessible introduction). HMP is described in Weindel, van Maanen & Borst (in preparation).

As a summary of the method, an HMP model parses the reaction time into a number of successive stages determined based on patterns in a neural time-serie. Hence any reaction time can then be described by a number of stages and their duration estimated using HMP. The important aspect of HMP is that it is a whole-brain analysis (or whole scalp analysis) that estimates the onset of stages on a single-trial basis. These by-trial estimates allow you then to further dig into any aspect you are interested in a signal:
- Describing an experiment or a clinical sample in terms of stages detected in the EEG signal
- Describing experimental effects based on a particular stage duration
- Estimating the effect of trial-wise manipuations on the identified stages (e.g. the by-trial variation of stimulus strength or the effect of time-on-task)
- Time-lock EEG signal to the onset of a given stage and perform classical ERPs or time-frequency analysis based on the onset of a new stage
- And many more (e.g. evidence accumulation models on decision stage, classification based on the number of transition events in the signal,...)


# Documentation

The package is available through *pip*. 
A recommended way of using the package is to use a conda environment (see [anaconda](https://www.anaconda.com/products/distribution>) for how to install conda):

    $ conda create -n hmp 
    $ conda activate hmp
    $ conda install pip #if not already installed
    $ pip install hsmm_mvpy

Then import hsmm-mvpy in your favorite python IDE through:

```python
    import hsmm_mvpy as hmp
```

For the cutting edge version you can clone the repository using *git*

Open a terminal and type:

    $ git clone https://github.com/gweindel/hsmm_mvpy.git
   
Then move to the clone repository and run 
    
    $ pip install -e .


## To get started
To get started with the code:
- Check the demo below 
- Inspect the tutorials in the tutorials repository
    - Tutorial 1: Loading EEG data 
    - Tutorial 2: Estimating a HMP with given number of stages
    - Tutorial 3: Test for the number of stages that best explains the data
    - Tutorial 4: Testing differences across conditions

To further learn about the oriignal HsMM-MVPA method be sure to check the paper by Anderson, Zhang, Borst, & Walsh  ([2016](https://psycnet.apa.org/doi/10.1037/rev0000030)) as well as the book chapter by Borst & Anderson ([2021](http://jelmerborst.nl/pubs/ACTR_HMP_MVPA_BorstAnderson_preprint.pdf)). The following list contains a non-exhaustive list of papers published using the original HsMM-MVPA method:
- Anderson, J.R., Borst, J.P., Fincham, J.M., Ghuman, A.S., Tenison, C., & Zhang, Q. (2018). The Common Time Course of Memory Processes Revealed. Psychological Science 29(9), pp. 1463-1474. [link](https://doi.org/10.1177/0956797618774526)
- Berberyan, H.S., Van Rijn, H., & Borst, J.P. (2021). Discovering the brain stages of lexical decision: Behavioral effects originate from a single neural decision process. Brain and Cognition 153: 105786. [link](https://www.sciencedirect.com/science/article/pii/S0278262621001068)
- Berberyan, H. S., van Maanen, L., van Rijn, H., & Borst, J. (2021). EEG-based identification of evidence accumulation stages in decision-making. Journal of Cognitive Neuroscience, 33(3), 510-527. [link](https://doi.org/10.1162/jocn_a_01663)
- Van Maanen, L., Portoles, O., & Borst, J. P. (2021). The discovery and interpretation of evidence accumulation stages. Computational brain & behavior, 4(4), 395-415. [link](https://link.springer.com/article/10.1007/s42113-021-00105-2)
- Portoles, O., Blesa, M., van Vugt, M., Cao, M., & Borst, J. P. (2022). Thalamic bursts modulate cortical synchrony locally to switch between states of global functional connectivity in a cognitive task. PLoS computational biology, 18(3), e1009407. [link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009407)
- Portoles, O., Borst, J.P., & Van Vugt, M.K. (2018). Characterizing synchrony patterns across cognitive task stages of associative recognition memory. European Journal of Neuroscience 48, pp. 2759-2769. [link](http://onlinelibrary.wiley.com/doi/10.1111/ejn.13817/epdf)
- Zhang, Q., van Vugt, M.K., Borst, J.P., & Anderson, J.R. (2018). Mapping Working Memory Retrieval in Space and in Time: A Combined Electroencephalography and Electrocorticography Approach. NeuroImage 174, pp. 472-484. [link](https://www.sciencedirect.com/science/article/pii/S1053811918302490)



## Demo on simulated data

The following section will quickly walk you through an example usage in simulated data (using [MNE](https://mne.tools/dev/auto_examples/simulation/index.html)'s simulation functions and tutorials)

First we load the packages necessary for the demo on simulated data

### Importing libraries



```python
## Importing these packages is specific for this simulation case
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma

## Importing HMP
import hsmm_mvpy as hmp
from hsmm_mvpy import simulations
```



### Simulating data

In the following code block we simulate 50 trials from four known sources, this is not code you would need for your own analysis except if you'd want to simulate and test properties of HMP models. All four sources are defined by a location, an activation amplitude and a distribution in time (here a gamma with shape and scale parameters) for the onsets of the stages on each trial. The simulation functions are based on this [MNE tutorial ](https://mne.tools/stable/auto_examples/simulation/simulated_raw_data_using_subject_anatomy.html).

_If you're running this for the first time a 1.65 G file will be downloaded in order to perform the simulation but this will be done only once (alternatively you can just download the corresponding simulation file and place it in the same folder from where you are running this notebook)_




```python
cpus = 5 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM

n_trials = 50 #Number of trials to simulate

##### Here we define the sources of the brain activity (event) for each trial
sfreq = 500#sampling frequency of the signal
n_events = 4
n_stages = n_events+1
frequency = 10.#Frequency of the event defining its duration, half-sine of 10Hz = 50ms
amplitude = .4e-6 #Amplitude of the event in nAm, defining signal to noise ratio
shape = 2 #shape of the gamma distribution
means = np.array([60, 150, 200, 100, 80])/shape #Mean duration of the stages in ms
names = simulations.available_sources()[:n_stages+1]#Which source to activate at each stage (see atlas when calling simulations.available_sources())

sources = []
for source in zip(names, means):#One source = one frequency, one amplitude and a given by-trial variability distribution
    sources.append([source[0], frequency, amplitude, gamma(shape, scale=source[1])])

# Function used to generate the data
file = simulations.simulate(sources, n_trials, cpus, 'dataset_README',  overwrite=False, sfreq=sfreq, seed=123)
#load electrode position, specific to the simulations
positions = simulations.simulation_positions()
```

    /home/gweindel/owncloud/projects/RUGUU/main_hmp/hsmm_mvpy/src/hsmm_mvpy/simulations.py:165: UserWarning: ./dataset_README_raw.fif exists no new simulation performed
      warn(f'{subj_file} exists no new simulation performed', UserWarning)


### Creating the event structure and plotting the raw data

To recover the data we need to create the event structure based on the triggers sent during simulation. This is the same as analyzing real EEG data and recovering events in the stimulus channel. In our case 1 signal the onset of the stimulus and 6 the moment of the response. Hence a trial is defined as the times occuring between the triggers 1 and 6.


```python
#Recovering the events to epoch the data (in the number of trials defined above)
generating_events = np.load(file[1])
resp_trigger = int(np.max(np.unique(generating_events[:,2])))#Resp trigger is the last source in each trial
event_id = {'stimulus':1}#trigger 1 = stimulus
resp_id = {'response':resp_trigger}
#Keeping only stimulus and response triggers
events = generating_events[(generating_events[:,2] == 1) | (generating_events[:,2] == resp_trigger)]#only retain stimulus and response triggers

#Visualising the raw simulated EEG data
import mne
raw = mne.io.read_raw_fif(file[0], preload=False, verbose=False)
raw.pick_types(eeg=True).plot(scalings=dict(eeg=1e-5), events=events, block=True);
```

    NOTE: pick_types() is a legacy function. New code should use inst.pick(...).
    Using qt as 2D backend.
    Channels marked as bad:
    none


![png](README_files/README_7_1.png)

### Recovering number of stages as well as actual by-trial variation

To see how well HMP does at recovering by-trial stages below, here we get the ground truth from our simulation. Unfortunately, with an actual dataset you don’t have access to this, of course. 


```python
%matplotlib inline
#Recover the actual time of the simulated events
sim_event_times = np.reshape(np.ediff1d(generating_events[:,0],to_begin=0)[generating_events[:,2] > 1], \
           (n_trials, n_stages))
```

## Demo for a single participant in a single condition based on the simulated data

First we read the EEG data as we would for a single participant


```python
# Reading the data
eeg_data = hmp.utils.read_mne_data(file[0], event_id=event_id, resp_id=resp_id, sfreq=sfreq, 
            events_provided=events, verbose=False)

```

    Processing participant ./dataset_README_raw.fif's continuous eeg
    Reading 0 ... 199372  =      0.000 ...   398.744 secs...
    50 trials were retained for participant ./dataset_README_raw.fif



HMP uses [xarray](https://docs.xarray.dev/en/stable/) named dimension matrices, allowing to directly manipulate the data using the name of the dimensions:




```python
#example of usage of xarray
print(eeg_data)
eeg_data.sel(channels=['EEG 001','EEG 002','EEG 003'], samples=range(400))\
    .data.groupby('samples').mean(['participant','epochs']).plot.line(hue='channels');
```

    <xarray.Dataset>
    Dimensions:      (participant: 1, epochs: 50, channels: 59, samples: 570)
    Coordinates:
      * epochs       (epochs) int64 0 1 2 3 4 5 6 7 8 ... 41 42 43 44 45 46 47 48 49
      * channels     (channels) <U7 'EEG 001' 'EEG 002' ... 'EEG 059' 'EEG 060'
      * samples      (samples) int64 0 1 2 3 4 5 6 7 ... 563 564 565 566 567 568 569
        event_name   (epochs) object 'stimulus' 'stimulus' ... 'stimulus' 'stimulus'
        rt           (epochs) float64 0.582 0.594 0.474 0.538 ... 0.572 0.364 0.398
      * participant  (participant) <U2 'S0'
    Data variables:
        data         (participant, epochs, channels, samples) float64 -1.768e-06 ...
    Attributes:
        sfreq:    500.0
        offset:   0



    
![png](README_files/README_12_1.png)
    


Next we transform the data as in Anderson, Zhang, Borst, & Walsh  ([2016](https://psycnet.apa.org/doi/10.1037/rev0000030)) including standardization of individual variances (not in this case as we have only one simulated participant), z-scoring and spatial principal components analysis (PCA). 

Note that if the number of components to retain is not specified, the scree plot of the PCA is displayed and a prompt ask how many PCs should be retained (in this case we specify it as building the README does not allow for prompts)


```python
hmp_data = hmp.utils.transform_data(eeg_data, apply_standard=False, n_comp=4)
```

# HMP model

Once the data is in the expected format, we can initialize an HMP object; note that no estimation is performed yet.



```python
init = hmp.models.hmp(hmp_data, eeg_data, event_width=50, cpus=cpus)#Initialization of the model
```


We are looking for stages in the data (**Hidden Markov**) and assume that transitions between stages are signaled by a template in the data (**pattern analysis**). By default we use the same template as Anderson, Zhang, Borst, & Walsh  ([2016](https://psycnet'.apa.org/doi/10.1037/rev0000030)), a 10 Hz half-sine, resulting in a 50ms duration bump-like shape:



```python
plt.plot(init.template,'x');
```


    
![png](README_files/README_18_0.png)
    


This pattern is assumed to be present across several electrodes (**multivariate**). In order to find it, we apply a cross-correlation between that shape and the (normalized) EEG data:



```python
epoch = 0 #illustrating the first trial
hmp_data.unstack().sel(component=[0,1,2], epochs=epoch).squeeze().plot.line(hue='component');
plt.vlines(sim_event_times[epoch,:-1].cumsum()-1, -3, 3, 'k');#overlaying the simulated stage transition times
```


    
![png](README_files/README_20_0.png)
    


The by-trial onset of this transition pattern event is assumed to be captured by a probability distribution (**semi-Markov**), e.g. in this application a gamma with a shape of 2:



```python
T = 350
plt.plot(np.linspace(0,T,1001),gamma.pdf(np.linspace(0,T,1001), 2, scale=50)) 
plt.xlabel('t');
```


    
![png](README_files/README_22_0.png)
    


And this is then the full explanation of an HMP model: Looking for a transition event across several electrodes and trials that signals a transition to the next stage and which onset is expected to follow a probability distribution (by default a gamma distribution).

# Estimating an HMP model

We can directly fit an HMP model without giving any info on the number of stages (see tutorial 2 for a detailed explanation of the following cell)





```python
estimates = init.fit()
```


      0%|          | 0/280 [00:00<?, ?it/s]


    Transition event 1 found around sample 46
    Transition event 2 found around sample 119
    Transition event 3 found around sample 206
    Transition event 4 found around sample 251
    Estimating 4 events model
    parameters estimated for 4 events model


### Visualizing results of the fit

In the previous cell we initiated an HMP model looking for default 50ms bumps – the transition events – in the EEG signal. The method discovered four events, and therefore five gamma-distributed stages with a fixed shape of 2 and an estimated scale. We can now inspect the results of the fit.

We can directly take a look to the topologies and latencies of the events by calling ```hmp.visu.plot_topo_timecourse```




```python
hmp.visu.plot_topo_timecourse(eeg_data, estimates, #Data and estimations 
                               positions, init,#position of the electrodes and initialized model
                               magnify=1, sensors=False, #time_step=1000/init.sfreq,#Display control parameters
                               times_to_display = np.mean(init.ends - init.starts))#plot reaction times
```


    
![png](README_files/README_26_0.png)
    


This shows us the electrode activity on the scalp as well as the average time of occurence of the events based on the stage distributions.

As we are estimating the event onsets on a by-trial basis we can look at the by-trial variation in stage duration.



```python
ax = hmp.visu.plot_latencies(estimates, init, errs='ci')
ax.set_ylabel('your label here');
```


    
![png](README_files/README_29_0.png)
    


For the same reason we can also inspect the probability distribution of event onsets:



```python
hmp.visu.plot_distribution(estimates.eventprobs.mean(dim=['trial_x_participant']))
hmp.visu.plot_distribution(estimates.eventprobs.mean(dim=['trial_x_participant']), survival=True)
```


    
![png](README_files/README_31_1.png)
    



    
![png](README_files/README_31_2.png)
    


As HMP estimated stage onset per trial we can also look at the predicted stage onsets for a given trial.



```python
hmp.visu.plot_distribution(estimates.eventprobs.sel(trial_x_participant=('S0', 0)), 
                            xlims=(0,np.percentile(sim_event_times.sum(axis=1), q=90)))
```

    
![png](README_files/README_33_1.png)
    



This then shows the likeliest stage onset location in time for the first trial!

## Comparing with ground truth

As we simulated the data we have access to the ground truth of the underlying generative events. We can then compare the average stage durations compared to the one estimated by HMP. Perhaps to state the obvious, this code is specific to the case where you simulate data.



```python
colors = sns.color_palette(None, n_stages)
plt.scatter(np.mean(sim_event_times, axis=0), estimates.parameters.prod(axis=1), color=colors,s=50)
plt.plot([np.min(np.mean(sim_event_times,axis=0)),np.max(np.mean(sim_event_times,axis=0))],
         [np.min(np.mean(sim_event_times,axis=0)),np.max(np.mean(sim_event_times,axis=0))],'--');
plt.title('Actual vs estimated stage durations')
plt.xlabel('Simulated stage duration')
plt.ylabel('Estimated stage duration')
plt.show()
```


    
![png](README_files/README_35_0.png)
    



We can also overlay actual bumps onset with predicted one



```python
hmp.visu.plot_topo_timecourse(eeg_data, estimates, positions, init, magnify=1, sensors=False, figsize=(13,1), title='Actual vs estimated event onsets',
        times_to_display = np.mean(np.cumsum(sim_event_times,axis=1),axis=0))
```


    
![png](README_files/README_37_0.png)
    



We see that the HMP recovered the exact average location of the bumps defined in the simulated data.

We can further test how well the package did by comparing the generated single trial onsets with those estimated from the HMP model



```python
fig, ax = plt.subplots(n_stages,2, figsize=(10,2*n_stages), dpi=100, sharex=True)
i = 0
ax[0,0].set_title('point-wise')
ax[0,1].set_title('Distribution')
ax[-1,0].set_xlabel(f'Simulated by-trial stage {i+1}')
estimated_times = init.compute_times(init, estimates, duration=True, mean=False, add_rt=True).T
for event in estimated_times:
    sns.regplot(x=sim_event_times[:,i].T, y=event, ax=ax[i,0], color=colors[i])
    ax[i,0].plot([np.min(event), np.max(event)], [np.min(event), np.max(event)],'--', color='k')
    ax[i,0].set_ylabel(f'Estimated by-trial stage {i+1}')
    sns.kdeplot(event, ax=ax[i,1], color='orange')
    sns.kdeplot(sim_event_times[:,i].T, ax=ax[i,1], color='k')
    i+= 1

plt.xlim(0,250)
plt.tight_layout();
```
    
![png](README_files/README_39_1.png)
    



We see that every stage gets nicely recovered even on a by-trial basis!

# Beyond summary statistics for EEG analysis

Now the purpose, apart from determining the number and time course of important EEG events in the reaction time, is also to use the by-trial information.

We illustrate this by first plotting the traditional event-related potentials (i.e. taking the average of given electrodes across the different time points) with cherry-picked electrodes.



```python
import seaborn as sns
import pandas as pd

data = eeg_data.stack({'trial_x_participant':['participant','epochs']}).data.dropna('trial_x_participant', how="all")
fig, ax = plt.subplots(1,1, figsize=(20,5), sharey=True)

for electrode in zip(['EEG 016','EEG 025','EEG 020'],#Cherry-picked electrodes
                     ['darkgreen','darkred','darkblue']):
    df = pd.DataFrame(data.sel(channels=electrode[0]).squeeze().T).melt(var_name='Time')
    sns.lineplot(x='Time', y='value',data=df, color=electrode[1])

plt.vlines(np.cumsum(sim_event_times.mean(axis=0)), -3e-6, 3e-6)
plt.ylabel('Volt')
plt.xlim(0,500)
plt.ylim(-3e-6,3e-6);

```

    
![png](README_files/README_41_1.png)
    


Given how variable and serial each of these stages are, the more you progress in the chain of events the less clear the signal gets. This is similar to traditional ERPs that have a very clear signal in the beginning of a trial (as events are more in phase) and summed and blurred further away from the stimulus (as events are off phase).

Now things can get better if we first parse, by-trial, the signal into the different stages based on the HMP estimates:




```python
BRP_times = init.compute_times(init, estimates.dropna('event'), fill_value=0, add_rt=True)

fig, ax = plt.subplots(1,n_stages, figsize=(20,5), sharey=True, sharex=True)
ax[0].set_ylabel('Volt')
for stage in range(n_stages):
    BRP = hmp.utils.event_times(data, BRP_times,'EEG 016',stage=stage)
    df = pd.DataFrame(BRP).melt(var_name='Time')
    sns.lineplot(x="Time", y="value", data=df,ax=ax[stage], color='darkgreen')
    BRP = hmp.utils.event_times(data, BRP_times,'EEG 025',stage=stage)
    df = pd.DataFrame(BRP).melt(var_name='Time')
    sns.lineplot(x="Time", y="value", data=df,ax=ax[stage], color='darkred')
    BRP = hmp.utils.event_times(data, BRP_times,'EEG 032',stage=stage)
    df = pd.DataFrame(BRP).melt(var_name='Time')
    sns.lineplot(x="Time", y="value", data=df,ax=ax[stage], color='darkblue') 
    plt.xlim(0,100)
```
    
![png](README_files/README_43_1.png)
    




In this case we clearly see at each stage the half-sine (of 50ms hence ~ 30 samples for the sampling frequency used) we simulated in the first step and the preceding period of silence (hence the first stage doesn't contain such a half-sine). Note that the end of the stages tend to be noisier because fewer trials are defining the average signal.


### Follow-up

For examples on how to use the package when the number of transition events/stages is unkown, or to compare stage durations across conditions see the tutorial notebooks:
- Load EEG data (tutorial 1)
- Estimating a given number of events (tutorial 2)
- Test for the number of events that best explains the data (tutorial 3)
- Testing differences across conditions (tutorial 4)
- Plotting Event Related Potentials with an HMP decomposition (tutorial 5)
