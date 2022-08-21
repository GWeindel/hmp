'''

'''

import os.path as op
import os
import numpy as np
import mne
from mne.datasets import sample
    
def simulate(sources, n_trials, max_trial_length, n_jobs, bump_frequency, file, path, overwrite=False):  
    if 'raw.fif' not in file:
        file = file + '_raw.fif'
        print("Aligning file name to MNE's convention")
    if file in os.listdir(path) and not overwrite:
        raw = mne.io.read_raw_fif(path+file, verbose=False)
        generating_events = np.load(path+file.split('.fif')[0]+'_generating_events.npy')
        print(f'Loading {file} no new simulation performed')
        return raw, generating_events
    else:
        print(f'Simulating {file} in {path}')
        file = path+file
        # Following code and comments largely comes from MNE examples (e.g. https://mne.tools/stable/auto_examples/simulation/simulated_raw_data_using_subject_anatomy.html)
        # For this example, we will be using the information of the sample subject.
        # This will download the data if it not already on your machine. We also set
        # the subjects directory so we don't need to give it to functions.
        data_path = sample.data_path()
        subjects_dir = op.join(data_path, 'subjects')
        subject = 'sample'

        # First, we get an info structure from the test subject.
        evoked_fname = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')
        info = mne.io.read_info(evoked_fname, verbose=False)
        #info = info.pick_channels(info.ch_names[-61:])
        tstep = 1. / info['sfreq']

        # To simulate sources, we also need a source space. It can be obtained from the
        # forward solution of the sample subject.
        fwd_fname = op.join(data_path, 'MEG', subject,
                            'sample_audvis-meg-eeg-oct-6-fwd.fif')
        fwd = mne.read_forward_solution(fwd_fname, verbose=False)
        #fwd = fwd.pick_channels(info.ch_names[-61:])
        src = fwd['src']

        source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep, first_samp=0, duration=(2+1*n_trials+3)*max_trial_length*tstep)

        # stim_onset occurs every x samples.
        events = np.zeros((n_trials, 3), int)
        stim_onsets =  2000+max_trial_length * np.arange(n_trials)#2000 = offset of first stim
        events[:,0] = stim_onsets#last event 

        trigger = 1
        random_source_times = []
        generating_events = events
        for source in sources:
            selected_label = mne.read_labels_from_annot(
                subject, regexp=source[0], subjects_dir=subjects_dir, verbose=False)[0]
            label = mne.label.select_sources(
                subject, selected_label, location='center', extent=10,# Extent in mm of the region.
                subjects_dir=subjects_dir)

            # Define the time course of the activity for each source of the region to
            # activate.
            source_time_series = np.sin(2. * np.pi * bump_frequency * np.arange(25) * tstep) * source[1]


            #adding source event
            events = events.copy()
            rand_i = source[2][0](source[2][1],source[2][2],n_trials)
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
        raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd, n_jobs=n_jobs,verbose=False)
        cov = mne.make_ad_hoc_cov(raw.info, verbose=False)
        mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04], verbose=False)

        raw.save(file, overwrite=True)
        np.save(file.split('.fif')[0]+'_generating_events.npy', generating_events)
        print(f'{file} simulated')
        return raw, generating_events