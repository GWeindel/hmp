'''

'''

import os.path as op
import os
import numpy as np
import mne
from mne.datasets import sample

def available_source():
    '''
    list available sources for sample subject in MNE
    '''
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, 'subjects')
    subject = 'sample'
    labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir)
    named_labels = []
    for label in range(len(labels)):
        named_labels.append(labels[label].name)
    return named_labels

def simulation_sfreq():
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, 'subjects')
    subject = 'sample'
    # First, we get an info structure from the test subject.
    evoked_fname = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')
    info = mne.io.read_info(evoked_fname, verbose=False)
    return info['sfreq']
    
def simulate(sources, n_trials, n_jobs, file, n_subj=1, path='./', overwrite=False, verbose=False): 
    '''
    Simulates EEG n_trials using MNE's tools based on the specified sources
    
    Parameters
    ----------
    sources : list
        2D or 3D list with dimensions (n_subjects *) sources * source_parameters
        Source parameters should contain :
        - the name of the source (see the output of available_source())
        - the duration of the bump (in frequency, usually 10Hz)
        - the amplitude or strength of the signal from the source, expressed in volt (e.g. 1e-8 V)
        - the duration of the preceding stage as a list with a numpy rangom generator
            and two parameters (e.g. [np.random.gamma, shape, scale]). The durations are
            expected to be in milliseconds
    n_trials: int
        Number of trials
    n_jobs: int
        Number of jobs to use with MNE's function (multithreading)
    file: str
        Name of the file to be saved (number of the subject will be added)
    path: str
        path where to save the data
    overwrite: bool
        Whether to overwrite existing file
    verbose: bool
        Whether to display MNE's output
    
    Returns
    -------
    generating_events: ndarray
        Times of the simulated bumps used to test for accurate recovery compared to estimation
    files: list
        list of file names (file + number of subject)
    '''
    sources = np.array(sources, dtype=object)
    if len(np.shape(sources)) == 2:
        sources = [sources]#If only one subject
    if np.shape(sources)[0] != n_subj:
        raise ValueError('Number of subject is not coherent with the sources provided')
    #Infer max duration of a trial from the specified sources
    percentiles = np.empty(len(sources[0]))
    for source in range(len(sources[0])):
        stage_dur_fun = sources[0][source][-1]
        percentiles[source] = np.percentile(stage_dur_fun.rvs(size=n_trials), q=99)
    max_trial_length = np.sum(percentiles)+1000
    # Following code and comments largely comes from MNE examples (e.g. \
    # https://mne.tools/stable/auto_examples/simulation/simulated_raw_data_using_subject_anatomy.html)
    # It loads the data, info structure and forward solution for one example subject,
    # Note that all 'subject' will use this forward solution
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, 'subjects')
    subject = 'sample'
    # First, we get an info structure from the test subject.
    evoked_fname = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')
    info = mne.io.read_info(evoked_fname, verbose=verbose)
    tstep = 1. / info['sfreq']
    # To simulate sources, we also need a source space. It can be obtained from the
    # forward solution of the sample subject.
    fwd_fname = op.join(data_path, 'MEG', subject,'sample_audvis-meg-eeg-oct-6-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    src = fwd['src']
    
    #For each subject, simulate associated sources
    files = []
    for subj in range(n_subj):
        #Build simulator
        source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep, first_samp=0, \
                    duration=(2+1*n_trials+3)*max_trial_length*tstep)
        subj_file = file + f'_{subj}_raw.fif'
        if subj_file in os.listdir(path) and not overwrite:
            print(f'{subj_file} exists no new simulation performed')
            files.append(subj_file)
        else:
            subj_file = path+subj_file
            print(f'Simulating {subj_file}')
            sources_subj = sources[subj]
            # stim_onset occurs every x samples.
            events = np.zeros((n_trials, 3), int)
            stim_onsets =  2000+max_trial_length * np.arange(n_trials)#2000 = offset of first stim
            events[:,0] = stim_onsets#last event 
            events[:,2] = 1#trigger 1 = stimulus 

            #Fake source, actually stimulus onset
            selected_label = mne.read_labels_from_annot(
                    subject, regexp=sources_subj[0][0], subjects_dir=subjects_dir, verbose=verbose)[0]
            label = mne.label.select_sources(
                    subject, selected_label, location='center', extent=10,# Extent in mm of the region.
                    subjects_dir=subjects_dir)
            source_time_series = np.array([1e-99])#stim trigger
            source_simulator.add_data(label, source_time_series, events)
            source_simulator.add_data(label, source_time_series, events)

            trigger = 2
            random_source_times = []
            generating_events = events
            for source in sources_subj:
                selected_label = mne.read_labels_from_annot(
                    subject, regexp=source[0], subjects_dir=subjects_dir, verbose=verbose)[0]
                label = mne.label.select_sources(
                    subject, selected_label, location='center', extent=10,# Extent in mm of the region.
                    subjects_dir=subjects_dir)

                # Define the time course of the activity for each source of the region to
                # activate
                bump_duration = int(((1/source[1])/2)*info['sfreq'])
                source_time_series = np.sin(2. * np.pi * source[1] * np.arange(0,1000) * tstep)[:bump_duration]  * source[2]
                #adding source event
                events = events.copy()
                rand_i = source[-1].rvs(size=n_trials)/(tstep*1000)
                random_source_times.append(rand_i) #varying event 
                events[:, 0] = events[:,0] + random_source_times[-1] # Events sample.
                events[:, 2] = trigger  # All events have the sample id.
                trigger += 1
                generating_events = np.concatenate([generating_events, events])

                #add these events
                source_simulator.add_data(label, source_time_series, events)

            generating_events = generating_events[generating_events[:, 0].argsort()]
            # Project the source time series to sensor space and add some noise. The source
            # simulator can be given directly to the simulate_raw function.
            raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd, n_jobs=n_jobs,verbose=verbose)
            cov = mne.make_ad_hoc_cov(raw.info, verbose=verbose)
            mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04], verbose=verbose)

            raw.save(subj_file, overwrite=True)
            np.save(subj_file.split('.fif')[0]+'_generating_events.npy', generating_events)
            print(f'{subj_file} simulated')
            files.append(subj_file)
    if n_subj == 1:
        return files[0]
    else:
        return files