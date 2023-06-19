'''

'''

import os.path as op
import os
import numpy as np
import mne
from mne.datasets import sample
from warnings import warn


def available_sources(subselection=True):
    '''
    list available sources for sample subject in MNE
    '''
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, 'subjects')
    labels = mne.read_labels_from_annot('sample', subjects_dir=subjects_dir, verbose=False)
    named_labels = []
    for label in range(len(labels)):
        named_labels.append(labels[label].name)
    #Here we select sources with different topologies on HMP and relativ equal contribution on channels, not checked on MEG
    named_labels = np.array(named_labels)
    if subselection:
        named_labels = named_labels[[0,1,3,5,7,15,17,22,30,33,37,40,42,46,50,54,57,59,65,67]]
    return named_labels

def simulation_sfreq():
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, 'subjects')
    evoked_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
    info = mne.io.read_info(evoked_fname, verbose=False)
    return info['sfreq']

def simulation_positions():
    from mne import channels
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, 'subjects')
    subject = 'sample'
    evoked_fname = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')
    info = mne.io.read_info(evoked_fname, verbose=False)
    positions = np.delete(channels.layout._find_topomap_coords(info, 'eeg'),52,axis=0)#inferring channel location using MNE    
    return positions

def bump_shape(bump_width, bump_width_samples, steps):
    '''
    Computes the template of a half-sine (bump) with given frequency f and sampling frequency
    '''
    bump_idx = np.arange(bump_width_samples)*steps+steps/2
    bump_frequency = 1000/(bump_width*2)#gives bump frequency given that bumps are defined as half-sines
    template = np.sin(2*np.pi*bump_idx/1000*bump_frequency)#bump morph based on a half sine with given bump width and sampling frequency
    template = template/np.sum(template**2)#Weight normalized
    return template


def simulate(sources, n_trials, n_jobs, file, data_type='eeg', n_subj=1, path='./', overwrite=False, verbose=False, noise=True, times=None, location=1, seed=None): 
    '''
    Simulates n_trials of EEG and/or MEG using MNE's tools based on the specified sources
    
    Parameters
    ----------
    sources : list
        2D or 3D list with dimensions (n_subjects *) sources * source_parameters
        Source parameters should contain :
        - the name of the source (see the output of available_source())
        - the duration of the bump (in frequency, usually 10Hz)
        - the amplitude or strength of the signal from the source, expressed in volt (e.g. 1e-8 V)
        - the duration of the preceding stage as a scipy.stats distribution (e.g. scipy.stats.gamma(a, scale))
    n_trials: int
        Number of trials
    n_jobs: int
        Number of jobs to use with MNE's function (multithreading)
    file: str
        Name of the file to be saved (number of the subject will be added)
    data_type: str
        Whether to simulate "eeg" or "meg"
    n_subj: int
        How many subjects to simulate
    path: str
        path where to save the data
    overwrite: bool
        Whether to overwrite existing file
    verbose: bool
        Whether to display MNE's output
    noise: bool
        Adding noise to the simulated sources
    times: ndarray
        Deterministic simulation of event transitions times. Format is n_sources X n_trials
    location: float
        value in ms to add after a simulated event and before another one
    
    Returns
    -------
    generating_events: ndarray
        Times of the simulated bumps used to test for accurate recovery compared to estimation
    files: list
        list of file names (file + number of subject)
    '''
    if seed is not None:
        random_state = np.random.RandomState(seed)
    sources = np.array(sources, dtype=object)
    if len(np.shape(sources)) == 2:
        sources = [sources]#If only one subject
    if np.shape(sources)[0] != n_subj:
        raise ValueError('Number of subject is not coherent with the sources provided')
    #Infer max duration of a trial from the specified sources
    percentiles = np.empty(len(sources[0]))
    for source in range(len(sources[0])):
        if times is None:
            stage_dur_fun = sources[0][source][-1]
            percentiles[source] = np.percentile(stage_dur_fun.rvs(size=n_trials), q=99)
        else:
            percentiles[source] = np.max(times[:,source])
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
    if data_type == 'eeg':
        picked_type = mne.pick_types(info, meg=False, eeg=True)
    elif data_type == 'meg':
        picked_type = mne.pick_types(info, meg=True, eeg=False)
    elif data_type == 'eeg/meg':
        picked_type = mne.pick_types(info, meg=True, eeg=False)
    else:
        raise ValueError(f'Invalid data type {data_type}, expected "eeg", "meg" or "eeg/meg"')
    info = mne.pick_info(info, picked_type)
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
        files_subj = []
        source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep, first_samp=0, \
                    duration=(2+1*n_trials+3)*max_trial_length*tstep)
        if n_subj == 1: subj_file = file + f'_raw.fif'
        else: subj_file = file + f'_{subj}_raw.fif'
        if subj_file in os.listdir(path) and not overwrite:
            subj_file = path+subj_file
            warn(f'{subj_file} exists no new simulation performed', UserWarning)
            files_subj.append(subj_file)
            files_subj.append(subj_file.split('.fif')[0]+'_generating_events.npy')
            files.append(files_subj)
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
            label = mne.label.select_sources(subject, selected_label, subjects_dir=subjects_dir, random_state=random_state)
            source_time_series = np.array([1e-99])#stim trigger
            source_simulator.add_data(label, source_time_series, events)
            source_simulator.add_data(label, source_time_series, events)

            trigger = 2
            random_source_times = []
            generating_events = events
            for source in sources_subj:
                selected_label = mne.read_labels_from_annot(
                    subject, regexp=source[0], subjects_dir=subjects_dir, verbose=verbose)[0]
                label = mne.label.select_sources(subject, selected_label, subjects_dir=subjects_dir, location=0, grow_outside=False, random_state=random_state)
                #last two parameters ensure sources that are different enough
                # Define the time course of the activity for each source of the region to
                # activate
                bump_duration = int(((1/source[1])/2)*info['sfreq'])
                source_time_series = bump_shape(((1000/source[1])/2),bump_duration,1000/info['sfreq'])*source[2]

                #adding source event
                events = events.copy()
                if times is None:
                    rand_i = np.round(source[-1].rvs(size=n_trials, random_state=random_state)/(tstep*1000),decimals=0)
                else:
                    rand_i = times[:,trigger-2]
                if len(sources_subj)+1 > trigger > 2:
                    rand_i += location/(tstep*1000)
                else:
                    rand_i += 1
                if 0 in rand_i:
                    warn("0 stage duration found, add a location parameter to avoid this case", UserWarning)
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
            if data_type == 'eeg':
                raw = raw.pick_types(meg=False, eeg=True, stim=True)
            elif data_type == 'meg':
                raw = raw.pick_types(meg=True, eeg=False, stim=True)
            elif data_type == 'eeg/meg':
                raw = raw.pick_types(meg=True, eeg=True, stim=True)
            if noise:
                cov = mne.make_ad_hoc_cov(raw.info, verbose=verbose)
                mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04], verbose=verbose, random_state=random_state)
            data = raw.get_data()
            raw.save(subj_file, overwrite=True)
            files_subj.append(subj_file)
            np.save(subj_file.split('.fif')[0]+'_generating_events.npy', generating_events)
            files_subj.append(subj_file.split('.fif')[0]+'_generating_events.npy')
            files.append(files_subj)
            print(f'{subj_file} simulated')
    if n_subj == 1:
        return files[0]
    else:
        return files
    
